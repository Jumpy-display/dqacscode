import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
import torch
torch.set_num_threads(16)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pennylane as qml
import gymnasium as gym
import numpy as np
import time
import csv

import multiprocessing
from gymnasium.wrappers import FilterObservation, FlattenObservation
from gymnasium.wrappers import FrameStackObservation as FrameStack
from gymnasium.wrappers import RecordEpisodeStatistics

frame_stack_kwarg = 'stack_size'
NUM_CORES = 16
torch.set_num_threads(NUM_CORES)


class DiffQASLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except:
            print("Lightning not found, falling back to default (SLOW)")
            self.dev = qml.device("default.qubit", wires=n_qubits)

        self.encoding_ops = ['rx', 'ry'] 
        self.use_hadamards = [True, False]
        self.entanglement_ops = ['chain'] 
        self.variational_ops = ['rx', 'ry', 'rz']

        self.architectures = []
        for h in self.use_hadamards:
            for enc in self.encoding_ops:
                for ent in self.entanglement_ops:
                    for var in self.variational_ops:
                        self.architectures.append({
                            'hadamard': h,
                            'encoding_gate': enc,
                            'entanglement': ent,
                            'variational_gate': var
                        })

        self.n_candidates = len(self.architectures)
        print(f"DiffQAS initialized with {self.n_candidates} high-value architectures.")

        self.circuit_params = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits) 
        )
        
        self.struct_weights = nn.Parameter(torch.ones(self.n_candidates) / self.n_candidates)

        self.enc_scale = nn.Parameter(torch.ones(self.n_qubits))
        self.enc_bias = nn.Parameter(torch.zeros(self.n_qubits))

        self.readout_theta = nn.Parameter(torch.zeros(self.n_qubits))

        self.k_active = 2           
        self.k_active_start = min(6, self.n_candidates)
        self.k_active_mid = 2
        self.k_active_final = 1
        self.k_stage1_updates = 10
        self.k_stage2_updates = 100  
        self.lock_threshold = 0.95  
        self.lock_patience_updates = 3  
        self.lock_warmup_updates = 20   
        self.lock_check_every = 1        
        self.lock_patience = self.lock_patience_updates
        self.locked_idx = None
        self._lock_count = 0

        
        self.qnodes = []
        for i in range(self.n_candidates):
            self.qnodes.append(self.create_qnode(i))

    def create_qnode(self, arch_idx):
        arch = self.architectures[arch_idx]

        @qml.qnode(self.dev, interface='torch', diff_method="adjoint")
        def circuit(inputs, params, enc_scale, enc_bias, readout_theta):
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)

            if arch['hadamard']:
                for q in range(self.n_qubits):
                    qml.Hadamard(wires=q)

            enc_gate = arch['encoding_gate']
            for q in range(self.n_qubits):
                angle = enc_scale[q] * inputs[:, q] + enc_bias[q]
                if enc_gate == 'rx':
                    qml.RX(angle, wires=q)
                elif enc_gate == 'ry':
                    qml.RY(angle, wires=q)
                elif enc_gate == 'rz':
                    qml.RZ(angle, wires=q)

            for l in range(self.n_layers):
                layer_params = params[l]
                ent_type = arch['entanglement']

                if ent_type == 'chain':
                    for q in range(self.n_qubits - 1):
                        qml.CNOT(wires=[q, q+1])
                
                var_gate = arch['variational_gate']
                for q in range(self.n_qubits):
                    if var_gate == 'rx':
                        qml.RX(layer_params[q], wires=q)
                    elif var_gate == 'ry':
                        qml.RY(layer_params[q], wires=q)
                    elif var_gate == 'rz':
                        qml.RZ(layer_params[q], wires=q)

            
            for q in range(self.n_qubits):
                qml.RY(readout_theta[q], wires=q)

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        
        x_in = x

        
        if self.locked_idx is not None:
            qnode = self.qnodes[int(self.locked_idx)]
            circuit_out = qnode(x_in, self.circuit_params, self.enc_scale, self.enc_bias, self.readout_theta)
            if isinstance(circuit_out, (list, tuple)):
                circuit_out = torch.stack(circuit_out, dim=-1)
            if circuit_out.ndim == 1 and batch_size == 1:
                circuit_out = circuit_out.unsqueeze(0)
            return circuit_out.float()

        probs = torch.softmax(self.struct_weights, dim=0)

        
        k = int(min(self.k_active, self.n_candidates))
        active_indices = torch.topk(probs, k=k).indices

        

        ensemble_output = torch.zeros(batch_size, self.n_qubits, device=x.device)

        for i in active_indices:
            qnode = self.qnodes[int(i)]
            circuit_out = qnode(x_in, self.circuit_params, self.enc_scale, self.enc_bias, self.readout_theta)

            if isinstance(circuit_out, (list, tuple)):
                circuit_out = torch.stack(circuit_out, dim=-1)
            if circuit_out.ndim == 1 and batch_size == 1:
                circuit_out = circuit_out.unsqueeze(0)

            ensemble_output = ensemble_output + probs[i] * circuit_out.float()

        
        sum_active = probs[active_indices].sum()
        ensemble_output = ensemble_output / (sum_active + 1e-8)

        return ensemble_output


    def maybe_lock(self, update_idx: int):
        
        if self.locked_idx is not None:
            return

        
        if update_idx % int(self.lock_check_every) != 0:
            return

        
        if update_idx < int(self.lock_warmup_updates):
            self._lock_count = 0
            return

        with torch.no_grad():
            probs = torch.softmax(self.struct_weights, dim=0)
            max_prob, max_idx = torch.max(probs, dim=0)

            if float(max_prob) >= float(self.lock_threshold):
                self._lock_count += 1
                if self._lock_count >= int(self.lock_patience_updates):
                    self.locked_idx = int(max_idx.item())
                    print(f"[DiffQAS] Hard-locked architecture {self.locked_idx} at update {update_idx} (max_prob={float(max_prob):.4f})")
            else:
                self._lock_count = 0


    def update_search_schedule(self, update_idx: int) -> int:
        
        
        if self.locked_idx is not None:
            self.k_active = 1
            return self.k_active

        if update_idx < int(self.k_stage1_updates):
            k = self.k_active_start
        elif update_idx < int(self.k_stage2_updates):
            k = self.k_active_mid
        else:
            k = self.k_active_final

        self.k_active = int(max(1, min(int(k), int(self.n_candidates))))
        return self.k_active


class HybridQRLAgent(nn.Module):
    def __init__(self, input_dim=147, q_dim=8, action_dim=6, n_vqc_layers=2):
        super(HybridQRLAgent, self).__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, q_dim),
            nn.Tanh(),
        )
        self.q_layer = DiffQASLayer(n_qubits=q_dim, n_layers=n_vqc_layers)
        self.actor_head = nn.Linear(q_dim, action_dim)
        self.critic_head = nn.Linear(q_dim, 1)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.obs_encoder(x)
        q_out = self.q_layer(x)
        action_logits = self.actor_head(q_out)
        state_value = self.critic_head(q_out)
        return action_logits, state_value

    def get_action_and_value(self, state, action=None):
        
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
            
            if state.ndim == 1:
                state = state.unsqueeze(0)
        
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self, batch_size):
        
        states_arr = np.array(self.states).reshape(-1, *np.array(self.states).shape[2:])
        actions_arr = np.array(self.actions).flatten()
        probs_arr = np.array(self.probs).flatten()
        vals_arr = np.array(self.vals).flatten()
        rewards_arr = np.array(self.rewards).flatten()
        dones_arr = np.array(self.dones).flatten()
        
        n_states = len(states_arr)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]

        return states_arr, actions_arr, probs_arr, vals_arr, rewards_arr, dones_arr, batches

    def store_memory(self, state, action, probs, vals, reward, done):
        
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


def ppo_update(agent, optimizer, memory, next_value=None, next_done=None, batch_size=512, gamma=0.99, gae_lambda=0.95, 
               n_epochs=3, clip_range=0.2, lambda_arch=0.05, lambda_angle=0.01, ent_coef=0.01):
    
    
    states_arr, actions_arr, old_probs_arr, vals_arr, rewards_arr, dones_arr, batches = \
        memory.generate_batches(batch_size)

    states = torch.tensor(states_arr, dtype=torch.float)
    actions = torch.tensor(actions_arr, dtype=torch.long)
    old_probs = torch.tensor(old_probs_arr, dtype=torch.float)
    rewards = torch.tensor(rewards_arr, dtype=torch.float)
    dones = torch.tensor(dones_arr, dtype=torch.bool)
    values = torch.tensor(vals_arr, dtype=torch.float)

    
    assert next_value is not None and next_done is not None

    n_envs = int(next_value.shape[0])
    n_steps = rewards.shape[0] // n_envs

    rewards_mat = rewards.view(n_steps, n_envs)
    dones_mat   = dones.view(n_steps, n_envs)
    values_mat  = values.view(n_steps, n_envs)

    advantage_mat = torch.zeros_like(rewards_mat)
    last_advantage = torch.zeros(n_envs)

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            nextvalues = next_value
            nextnonterminal = 1.0 - next_done.float()
        else:
            nextvalues = values_mat[t + 1]
            nextnonterminal = 1.0 - dones_mat[t].float()

        delta = rewards_mat[t] + gamma * nextvalues * nextnonterminal - values_mat[t]
        last_advantage = delta + gamma * gae_lambda * nextnonterminal * last_advantage
        advantage_mat[t] = last_advantage

    advantage = advantage_mat.reshape(-1)
    returns = advantage + values



    
    for epoch in range(n_epochs):
        for i_batch, batch in enumerate(batches):
            state_batch = states[batch]
            action_batch = actions[batch]
            old_log_prob_batch = old_probs[batch]
            return_batch = returns[batch]
            advantage_batch = advantage[batch]

            
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

            
            _action, new_log_prob, entropy, value_pred = agent.get_action_and_value(state_batch, action_batch)
            value_pred = value_pred.squeeze()

            
            ratio = torch.exp(new_log_prob - old_log_prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(value_pred, return_batch)
            entropy_loss = -entropy.mean()

            
            probs = torch.softmax(agent.q_layer.struct_weights, dim=0)
            loss_arch = lambda_arch * -torch.sum(probs * torch.log(probs + 1e-10)) 
            
            circuit_params = agent.q_layer.circuit_params
            penalty_term = F.relu(circuit_params - torch.pi) + F.relu(-circuit_params - torch.pi)
            angle_penalty = torch.sum(penalty_term ** 2)
            loss_angle = lambda_angle * angle_penalty

            total_loss = actor_loss + 0.5 * critic_loss + ent_coef * entropy_loss + loss_arch + loss_angle

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    memory.clear_memory()

SEED=67
def make_env(seed: int, idx: int):
    def _init():
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  
        os.environ["BLIS_NUM_THREADS"] = "1"       
        
        env = gym.make("MiniGrid-Empty-8x8-v0", render_mode=None)
        
        # Seed per worker
        env.reset(seed=seed + idx) 

        env = FilterObservation(env, filter_keys=["image"])
        env = FrameStack(env, **{frame_stack_kwarg: 4})
        env = RecordEpisodeStatistics(env)
        return env
    return _init


def train():
    env_name = "MiniGrid-Empty-8x8-v0"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"{env_name}_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_file = os.path.join(run_dir, "training_log.csv")
    
    NUM_ENVS = 4
    print(f"Initializing {NUM_ENVS} parallel environments with Frame Stacking...")
    envs = gym.vector.AsyncVectorEnv([make_env(SEED, i) for i in range(NUM_ENVS)])
    
    dummy = make_env(SEED, 0)()
    obs_space = dummy.observation_space
    if isinstance(obs_space, gym.spaces.Dict):
         input_dim = int(np.prod(obs_space["image"].shape))
    else:
         input_dim = int(np.prod(obs_space.shape))    
    action_dim = dummy.action_space.n
    dummy.close()

    agent = HybridQRLAgent(input_dim, 8, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    memory = PPOMemory()

    
    MAX_UPDATES = 1000
    STEPS_PER_UPDATE = 1024 
    gamma=0.99
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    
    
    arch_configs = agent.q_layer.architectures
    arch_headers = []
    for i, conf in enumerate(arch_configs):
        h_str = "H" if conf['hadamard'] else "NoH"
        name = f"A{i}_{h_str}_{conf['encoding_gate']}_{conf['entanglement']}_{conf['variational_gate']}"
        arch_headers.append(name)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "update",
            "global_step",
            "rollout_reward_per_env",   
            "ep_return_mean",          
            "ep_len_mean",              
            "episodes_finished"         
        ] + arch_headers)

    
    obs, _ = envs.reset()
    next_obs = obs["image"].reshape(NUM_ENVS, -1)

    print(f"Starting Training: {NUM_ENVS} Envs | {STEPS_PER_UPDATE} Steps/Env | Batch Size {NUM_ENVS*STEPS_PER_UPDATE}")

    global_step = 0
    good_streak=0
    for update in range(1, MAX_UPDATES + 1):

        
        agent.q_layer.update_search_schedule(update)

        
        update_reward = 0

        
        ep_returns = []
        ep_lengths = []
        for step in range(STEPS_PER_UPDATE):
            global_step += NUM_ENVS

            
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs)

            
            real_action = action.cpu().numpy()
            obs, rewards, terminations, truncations, infos = envs.step(real_action)

            
            final_infos = infos.get("final_info", None)
            if final_infos is not None:
                
                for fi in final_infos:
                    if fi is None:
                        continue
                    ep = fi.get("episode", None)
                    if ep is not None:
                        ep_returns.append(float(ep["r"]))
                        ep_lengths.append(int(ep["l"]))
            elif "episode" in infos:
                
                ep = infos["episode"]
                r = np.asarray(ep["r"])
                l = np.asarray(ep["l"])

                
                mask = infos.get("_episode", None)
                if mask is None:
                    
                    mask = l > 0
                else:
                    mask = np.asarray(mask, dtype=bool)

                ep_returns.extend(r[mask].astype(np.float32).tolist())
                ep_lengths.extend(l[mask].astype(np.int32).tolist())
            
            dones = terminations | truncations
            timeouts = truncations & (~terminations)
                        
            shaped_rewards = np.full_like(rewards, -0.01, dtype=np.float32)
            shaped_rewards = np.where(rewards > 0, 10.0, shaped_rewards)
            shaped_rewards = np.where(terminations & (rewards <= 0), -1.0, shaped_rewards)
            
            memory.store_memory(
                next_obs,
                real_action,
                log_prob.cpu().numpy(),
                value.cpu().numpy().flatten(),
                shaped_rewards,
                dones,
            )

            
            next_obs = obs["image"].reshape(NUM_ENVS, -1)

            
            update_reward += np.sum(rewards)

            

        
        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(next_obs)
            next_value = next_value.squeeze(-1)

       
        next_done = torch.from_numpy(dones).bool()

       
        if len(ep_returns) > 0:
            print(f"--- Update {update} | Global Step {global_step} | Episode return mean (this update): {np.mean(ep_returns):.2f} | episodes: {len(ep_returns)}")
        else:
            print(f"--- Update {update} | Global Step {global_step} | Episode return mean (this update): n/a (no episodes ended)")
        
        
        ep_ret_mean = float(np.mean(ep_returns)) if len(ep_returns) > 0 else 0.0
        ep_count = int(len(ep_returns))
        

        if ep_count > 0 and ep_ret_mean >= 0.79:
            good_streak += 1
        else:
            good_streak = 0

        ent_coef = 0.01
        if good_streak >= 10:
            ent_coef = 0.001

        
                
        ppo_update(
            agent, optimizer, memory,
            batch_size=256, n_epochs=3,
            next_value=next_value,
            next_done=next_done,ent_coef=ent_coef
        )

        
        agent.q_layer.maybe_lock(update)

        
        with torch.no_grad():
            weights = agent.q_layer.struct_weights
            probs_t = torch.softmax(weights, dim=0)
            probs = probs_t.cpu().numpy().tolist()

            
            if update % 5 == 0:
                active_idx = torch.where(probs_t > 0.01)[0]
                print(f"  Active Architectures: {len(active_idx)} / {len(probs)}")

        rollout_reward_per_env = float(update_reward / NUM_ENVS)
        ep_ret_mean = float(np.mean(ep_returns)) if len(ep_returns) > 0 else 0.0
        ep_len_mean = float(np.mean(ep_lengths)) if len(ep_lengths) > 0 else 0.0
        ep_count = int(len(ep_returns))

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                update,
                global_step,
                rollout_reward_per_env,
                ep_ret_mean,
                ep_len_mean,
                ep_count,
            ] + probs)


        
        
        if update % 10 == 0:
            checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_update_{update}.pth")
            print(f"Saving checkpoint: {checkpoint_path}")
            torch.save({
                'update': update,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'arch_weights': agent.q_layer.struct_weights,
            }, checkpoint_path)

    
    print("Training Complete. Saving final model...")
    fin_path = os.path.join(ckpt_dir, f"final_model.pth")
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'arch_weights': agent.q_layer.struct_weights,
    }, fin_path)
    print("Model saved to ppo_agent_final.pth")

    envs.close()

if __name__ == '__main__':
    
    train()