import os
import time
import random
import csv
import sys
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import gymnasium as gym
import metaworld


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))


# =============================================================================
# Per-feature running mean / std for observation normalization
# =============================================================================
class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)


# =============================================================================
# Reward scaling (per-env, running discounted-return std)
# =============================================================================
class RewardScaler:
    def __init__(self, num_envs, gamma=0.99):
        self.gamma = gamma
        self.num_envs = num_envs
        self.running_return = np.zeros(num_envs, dtype=np.float64)
        self.variance = np.full(num_envs, 1.0, dtype=np.float64)

    def scale(self, rewards):
        self.running_return = self.running_return * self.gamma + rewards
        self.variance = 0.99 * self.variance + 0.01 * (self.running_return ** 2)
        return (rewards / (np.sqrt(self.variance) + 1e-8)).astype(np.float32)

    def reset(self, env_idx):
        self.running_return[env_idx] = 0.0


# =============================================================================
# GPU-native quantum simulator (unchanged)
# =============================================================================
class QuantumOps:
    def __init__(self, n_qubits, device):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.device = device
        indices = torch.arange(self.dim, device=device)

        self.cnot_perms = []
        for q in range(n_qubits - 1):
            control_bit = (indices >> (n_qubits - 1 - q)) & 1
            target_mask = 1 << (n_qubits - 1 - (q + 1))
            flipped = indices ^ target_mask
            self.cnot_perms.append(torch.where(control_bit.bool(), flipped, indices))

        self.pauli_z_signs = torch.zeros(n_qubits, self.dim, device=device)
        for q in range(n_qubits):
            bit = (indices >> (n_qubits - 1 - q)) & 1
            self.pauli_z_signs[q] = 1.0 - 2.0 * bit.float()

        self.H = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat, device=device) / (2 ** 0.5)

    def apply_gate(self, state, gate_2x2, qubit):
        batch = state.shape[0]
        a = 2 ** qubit
        b = 2 ** (self.n - qubit - 1)
        state = state.reshape(batch, a, 2, b)
        state = torch.einsum('bij,bkjl->bkil', gate_2x2, state)
        return state.reshape(batch, self.dim)

    def apply_cnot_chain(self, state):
        for perm in self.cnot_perms:
            state = state[:, perm]
        return state

    def measure_all_z(self, state):
        probs = state.real ** 2 + state.imag ** 2
        return (self.pauli_z_signs @ probs.T).T


def rx(theta):
    c = torch.cos(theta / 2).unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(theta / 2).unsqueeze(-1).unsqueeze(-1)
    return torch.cat([torch.cat([c, -1j * s], -1),
                      torch.cat([-1j * s, c], -1)], -2).to(torch.cfloat)

def ry(theta):
    c = torch.cos(theta / 2).unsqueeze(-1).unsqueeze(-1)
    s = torch.sin(theta / 2).unsqueeze(-1).unsqueeze(-1)
    return torch.cat([torch.cat([c, -s], -1),
                      torch.cat([s,  c], -1)], -2).to(torch.cfloat)

def rz(theta):
    p = theta / 2
    ep = torch.exp(1j * p).unsqueeze(-1).unsqueeze(-1)
    en = torch.exp(-1j * p).unsqueeze(-1).unsqueeze(-1)
    z = torch.zeros_like(ep)
    return torch.cat([torch.cat([en, z], -1),
                      torch.cat([z, ep], -1)], -2)

ROT_FNS = {'rx': rx, 'ry': ry, 'rz': rz}


# =============================================================================
# DiffQAS layer (Data Re-uploading) slower, more cautious search schedule
# =============================================================================
class BatchedDiffQASLayer(nn.Module):
    def __init__(self, n_qubits=16, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ops = None

        self.architectures = []
        for h in [True, False]:
            for enc in ['rx', 'ry']:
                for var in ['rx', 'ry', 'rz']:
                    self.architectures.append({
                        'hadamard': h, 'encoding_gate': enc, 'variational_gate': var,
                    })
        self.n_candidates = len(self.architectures)
        print(f"DiffQAS (Re-uploading) initialized with {self.n_candidates} candidate architectures.")

        self.circuit_params = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.enc_scale = nn.Parameter(torch.ones(n_layers, n_qubits))
        self.enc_bias = nn.Parameter(torch.zeros(n_layers, n_qubits))

        self.struct_weights = nn.Parameter(torch.ones(self.n_candidates) / self.n_candidates)
        self.readout_theta = nn.Parameter(torch.zeros(n_qubits))

        # Slower search schedule -- give the policy time to evaluate candidates
        self.k_active = 3
        self.k_active_start = min(3, self.n_candidates)
        self.k_active_mid = 2
        self.k_active_final = 1
        self.k_stage1_updates = 20
        self.k_stage2_updates = 80

        self.lock_threshold = 0.95
        self.lock_patience_updates = 3
        self.lock_warmup_updates = 100
        self.lock_check_every = 1
        self.locked_idx = None
        self._lock_count = 0

    def _ensure_ops(self, device):
        if self.ops is None or self.ops.device != device:
            self.ops = QuantumOps(self.n_qubits, device)

    def _run_single(self, x, arch):
        B = x.shape[0]
        ops = self.ops
        dev = x.device

        state = torch.zeros(B, ops.dim, dtype=torch.cfloat, device=dev)
        state[:, 0] = 1.0

        if arch['hadamard']:
            h_exp = ops.H.unsqueeze(0).expand(B, -1, -1)
            for q in range(self.n_qubits):
                state = ops.apply_gate(state, h_exp, q)

        enc_fn = ROT_FNS[arch['encoding_gate']]
        var_fn = ROT_FNS[arch['variational_gate']]

        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                angle = self.enc_scale[l, q] * x[:, q] + self.enc_bias[l, q]
                state = ops.apply_gate(state, enc_fn(angle), q)
            state = ops.apply_cnot_chain(state)
            for q in range(self.n_qubits):
                state = ops.apply_gate(
                    state, var_fn(self.circuit_params[l, q].expand(B)), q
                )

        for q in range(self.n_qubits):
            state = ops.apply_gate(state, ry(self.readout_theta[q].expand(B)), q)

        return ops.measure_all_z(state)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        self._ensure_ops(x.device)

        if self.locked_idx is not None:
            return self._run_single(x, self.architectures[int(self.locked_idx)])

        probs = torch.softmax(self.struct_weights, dim=0)
        k = min(self.k_active, self.n_candidates)
        active = torch.topk(probs, k=k).indices

        B = x.shape[0]
        out = torch.zeros(B, self.n_qubits, device=x.device)
        w_sum = torch.zeros((), device=x.device)
        for i in active:
            out = out + probs[i] * self._run_single(x, self.architectures[int(i)])
            w_sum = w_sum + probs[i]
        return out / (w_sum + 1e-8)

    def update_search_schedule(self, update_idx: int) -> int:
        if self.locked_idx is not None:
            self.k_active = 1
            return 1
        if update_idx < self.k_stage1_updates:
            k = self.k_active_start
        elif update_idx < self.k_stage2_updates:
            k = self.k_active_mid
        else:
            k = self.k_active_final
        self.k_active = max(1, min(k, self.n_candidates))
        return self.k_active

    def maybe_lock(self, update_idx: int):
        if self.locked_idx is not None:
            return
        if update_idx % self.lock_check_every != 0:
            return
        if update_idx < self.lock_warmup_updates:
            self._lock_count = 0
            return
        with torch.no_grad():
            probs = torch.softmax(self.struct_weights, dim=0)
            mp, mi = torch.max(probs, 0)
            if float(mp) >= self.lock_threshold:
                self._lock_count += 1
                if self._lock_count >= self.lock_patience_updates:
                    self.locked_idx = int(mi)
                    self.k_active = 1
                    print(f"[DiffQAS] Locked arch {self.locked_idx} @ update {update_idx} (p={float(mp):.4f})")
            else:
                self._lock_count = 0


# =============================================================================
# Helper for Orthogonal Initialization
# =============================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# Continuous hybrid agent: tanh-bounded angles + skip connection
# =============================================================================
class HybridContinuousAgent(nn.Module):
    def __init__(self, obs_dim=39, q_dim=16, action_dim=4, n_vqc_layers=3, hidden_dim=64):
        super().__init__()

        self.obs_dim = obs_dim
        self.q_dim = q_dim

        # Learnable projection 39 -> 16 (no bias, orthogonal init)
        self.state_proj = nn.Linear(obs_dim, q_dim, bias=False)
        torch.nn.init.orthogonal_(self.state_proj.weight, gain=1.0)

        # NOTE: frozen LayerNorm is GONE. We now use tanh()*pi which:
        #   - bounds angles to [-pi, pi] without wrap-around aliasing,
        #   - gives a smooth, non-saturating gradient through state_proj,
        #   - lets state_proj learn the right *scale* per feature.
        self.q_layer = BatchedDiffQASLayer(n_qubits=q_dim, n_layers=n_vqc_layers)

        # Skip connection: post_vqc sees [quantum_features || raw_normalized_obs]
        # so the policy is not bottlenecked through 16 bounded scalars.
        self.post_vqc = nn.Sequential(
            layer_init(nn.Linear(q_dim + obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        self.actor_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # Slightly higher exploration ceiling
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))
        self.min_logstd = -3.0

    def get_action_and_value(self, norm_state, action=None):
        # `norm_state` is already obs-normalized (mean 0 / std 1 per feature).
        projected = self.state_proj(norm_state)
        q_angles = torch.tanh(projected) * torch.pi

        q_out = self.q_layer(q_angles)

        # Concatenate quantum readout with the normalized observation (skip path)
        features = self.post_vqc(torch.cat([q_out, norm_state], dim=-1))

        action_mean = self.actor_mean(features)
        clamped_logstd = torch.clamp(self.actor_logstd, min=self.min_logstd)
        action_logstd = clamped_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        dist = Normal(action_mean, action_std)

        if action is None:
            # Sampling path (rollout): sample from Gaussian, then squash
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
        else:
            # Evaluation path (PPO update): recover the pre-tanh action
            # Clamp avoids atanh blowing up at exactly 1
            raw_action = torch.atanh(action.clamp(-0.999999, 0.999999))

        # Tanh-squashed Gaussian log-prob with Jacobian correction
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        # Entropy of the *pre-squash* Gaussian  used as an exploration bonus.
        # The true squashed entropy has no closed form; the Gaussian entropy is
        # the standard surrogate (SAC, etc.) and is fine for ENT_COEF weighting.
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic_head(features)
        return action, log_prob, entropy, value


# =============================================================================
# MetaWorld env factory
# =============================================================================
class MetaWorldGymAdapter(gym.Env):
    def __init__(self, env):
        super().__init__()
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        out = self._env.reset()
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

def make_metaworld_env(env_name, seed, idx):
    def thunk():
        ml1 = metaworld.ML1(env_name, seed=seed)
        env = ml1.train_classes[env_name]()
        rng = random.Random(seed + idx)
        env.set_task(rng.choice(ml1.train_tasks))
        adapter = MetaWorldGymAdapter(env)
        
        # Resample task on every reset
        original_reset = adapter.reset
        def reset_with_resample(*args, **kwargs):
            env.set_task(rng.choice(ml1.train_tasks))
            return original_reset(*args, **kwargs)
        adapter.reset = reset_with_resample
        
        return adapter
    return thunk

# =============================================================================
# Hyperparameters
# =============================================================================
SEED = 67
NUM_ENVS = 32
Q_DIM = 10                    
HIDDEN_DIM = 256
STEPS_PER_UPDATE = 1024
BATCH_SIZE = 256              # drop to 128 if OOM during k_active=3 search
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2

ENT_COEF = 0.001              # was 0.0  -- keep some exploration pressure
VF_COEF = 0.5                 # was implicitly 1.0 -- standard PPO value
LR = 3e-4
LAMBDA_ARCH = 0.02            # was 0.15 -- don't force premature arch lock-in
LAMBDA_ANGLE = 0.01

MAX_GRAD_NORM = 0.5
TARGET_KL = 0.05              
MAX_UPDATES = 1500
LR_END_FACTOR = 0.3
CHECKPOINT_EVERY = 100

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Main
# =============================================================================
def main():
    seed_everything(SEED)

    env_name = "faucet-open-v3"

    print(f"Initializing {NUM_ENVS} parallel MetaWorld envs ({env_name})...")
    envs = gym.vector.SyncVectorEnv(
        [make_metaworld_env(env_name, SEED, i) for i in range(NUM_ENVS)]
    )

    obs_dim = envs.single_observation_space.shape[0]
    print(f"obs_dim={obs_dim}, q_dim={Q_DIM}, post_vqc input={Q_DIM + obs_dim}")
    
    
    agent = HybridContinuousAgent(
        obs_dim=obs_dim, q_dim=Q_DIM, action_dim=4, n_vqc_layers=3, hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    arch_params = [agent.q_layer.struct_weights]
    base_params = [p for n, p in agent.named_parameters() if n != "q_layer.struct_weights"]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': LR},
        {'params': arch_params, 'lr': 1e-3},
    ], eps=1e-5)

    lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=LR_END_FACTOR, total_iters=MAX_UPDATES,
    )

    run_dir = os.path.join("runs", f"metaworld_{env_name}_v2_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "training_log.csv")

    arch_headers = [
        f"A{i}_{'H' if a['hadamard'] else 'NoH'}_{a['encoding_gate']}_chain_{a['variational_gate']}"
        for i, a in enumerate(agent.q_layer.architectures)
    ]

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "update", "global_step", "lr",
            "avg_reward", "rolling_mean_10",
            "avg_success", "rolling_success_10",
            "pg_loss", "v_loss", "arch_loss", "angle_loss",
            "mean_logstd",
            "k_active", "top_arch", "top_prob",
            "sps",
        ] + arch_headers)

    print(f"Run dir: {run_dir}")

    # Stat trackers
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    reward_scaler = RewardScaler(NUM_ENVS, gamma=GAMMA)

    print("\n--- Starting PPO training loop ---")
    t0 = time.time()
    global_step = 0
    reward_window = deque(maxlen=10)
    success_window = deque(maxlen=10)

    next_obs_raw, _ = envs.reset()

    for update in range(1, MAX_UPDATES + 1):
        agent.q_layer.update_search_schedule(update)

        # Buffer holds NORMALIZED observations (what the policy actually saw).
        b_states = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, obs_dim, device=DEVICE)
        b_actions = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, 4, device=DEVICE)
        b_logprobs = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=DEVICE)
        b_rewards = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=DEVICE)
        b_values = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=DEVICE)
        b_dones = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=DEVICE)

        raw_rewards_this_update = []
        raw_successes_this_update = []
        # Episode-level tracking (resets each update)
        ep_success_count = 0
        ep_total_count = 0
        # Per-env "did this episode see success" flag
        ep_succeeded = np.zeros(NUM_ENVS, dtype=bool)
        # --- Rollout ---
        for step in range(STEPS_PER_UPDATE):
            global_step += NUM_ENVS

            # 1) normalize with current stats, 2) update stats with this raw obs
            normed_np = obs_rms.normalize(next_obs_raw)
            obs_rms.update(next_obs_raw)
            norm_state = torch.tensor(normed_np, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(norm_state)

            action_np = action.cpu().numpy().astype(np.float32)

            next_obs_raw_new, rewards, terminations, truncations, infos = envs.step(action_np)
            

            # Track success robustly: gymnasium may put it under 'success' or
            # nest it inside 'final_info' on episode end.
            successes = np.zeros(NUM_ENVS, dtype=np.float32)
            if "success" in infos:
                successes = np.asarray(infos["success"], dtype=np.float32).reshape(-1)
            elif "final_info" in infos:
                fi = infos["final_info"]
                for i, info_i in enumerate(fi):
                    if info_i is not None and "success" in info_i:
                        successes[i] = float(info_i["success"])
            raw_successes_this_update.append(successes)
            if update < 600:
                success_bonus=1.0
            else:
                success_bonus=max(0.5, 3.0 * (1.0 - (update - 600) / 400.0))
            # ADD: success shaping bonus BEFORE reward scaling
            rewards_shaped = rewards.astype(np.float32) + success_bonus * successes
            raw_rewards_this_update.append(rewards.copy())  # log raw, not shaped
            raw_successes_this_update.append(successes)

            # Now scale the shaped rewards
            scaled_rewards = rewards_shaped * 0.1

            # Bootstrap on truncation (timeout, not termination)
            timeouts = truncations & (~terminations)
            if np.any(timeouts):
                final_obs = infos.get("final_observation", None)
                if final_obs is not None:
                    idxs = np.where(timeouts)[0]
                    valid = [i for i in idxs if final_obs[i] is not None]
                    if valid:
                        final_arr = np.stack([final_obs[i] for i in valid], axis=0)
                        # Normalize the bootstrap obs the same way
                        final_norm = obs_rms.normalize(final_arr)
                        final_tensor = torch.tensor(final_norm, dtype=torch.float32, device=DEVICE)
                        with torch.no_grad():
                            _, _, _, final_v = agent.get_action_and_value(final_tensor)
                            final_v = final_v.squeeze(-1).cpu().numpy()
                        scaled_rewards[valid] = scaled_rewards[valid] + GAMMA * final_v

            gae_dones = terminations.astype(np.float32)

            ep_succeeded |= (successes > 0)
            episode_over = terminations | truncations
            for i in np.where(episode_over)[0]:
                ep_total_count += 1
                if ep_succeeded[i]:
                    ep_success_count += 1
                ep_succeeded[i] = False  # reset for next episode in this env
                reward_scaler.reset(i)

            b_states[step] = norm_state
            b_actions[step] = action
            b_logprobs[step] = logprob
            b_rewards[step] = torch.from_numpy(scaled_rewards).to(DEVICE)
            b_values[step] = value.squeeze(-1)
            b_dones[step] = torch.from_numpy(gae_dones).to(DEVICE)

            next_obs_raw = next_obs_raw_new

        # --- GAE ---
        with torch.no_grad():
            normed_next = obs_rms.normalize(next_obs_raw)
            next_state_t = torch.tensor(normed_next, dtype=torch.float32, device=DEVICE)
            _, _, _, next_value = agent.get_action_and_value(next_state_t)
            next_value = next_value.squeeze(-1)

            advantages = torch.zeros_like(b_rewards)
            lastgaelam = torch.zeros(NUM_ENVS, device=DEVICE)
            for t in reversed(range(STEPS_PER_UPDATE)):
                if t == STEPS_PER_UPDATE - 1:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = b_values[t + 1]
                delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
                lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + b_values
        # Right after the GAE loop, before flattening
        print(f"  returns: u={returns.mean():.3f} o={returns.std():.3f} | "
              f"adv: u={advantages.mean():.3f} o={advantages.std():.3f}")
        f_states = b_states.reshape(-1, obs_dim)
        f_actions = b_actions.reshape(-1, 4)
        f_logprobs = b_logprobs.reshape(-1)
        f_advantages = advantages.reshape(-1)
        f_returns = returns.reshape(-1)
        N = f_states.shape[0]

        # Per-batch advantage normalization (more stable than per-minibatch)
        f_advantages = (f_advantages - f_advantages.mean()) / (f_advantages.std() + 1e-8)
        current_ent_coef = max(0.0, ENT_COEF * (1.0 - update / 800.0))
        # --- PPO update ---
        last_pg_loss = last_v_loss = last_arch_loss = last_angle_loss = 0.0
        stop_early = False

        for epoch in range(EPOCHS):
            if stop_early:
                break
            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = perm[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    f_states[mb_inds], f_actions[mb_inds]
                )

                with torch.no_grad():
                    approx_kl = (f_logprobs[mb_inds] - newlogprob).mean().item()
                if approx_kl > TARGET_KL:
                    stop_early = True
                    break

                logratio = newlogprob - f_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = f_advantages[mb_inds]
                
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.squeeze() - f_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                arch_probs = torch.softmax(agent.q_layer.struct_weights, dim=0)
                # NOTE: this is -entropy, so minimizing the loss DECREASES arch entropy
                # (i.e., pushes toward one-hot). LAMBDA_ARCH=0.02 makes this gentle.
                loss_arch = LAMBDA_ARCH * -torch.sum(
                    arch_probs * torch.log(arch_probs + 1e-10)
                )

                cp = agent.q_layer.circuit_params
                penalty = F.relu(cp - torch.pi) + F.relu(-cp - torch.pi)
                loss_angle = LAMBDA_ANGLE * torch.sum(penalty ** 2)

                loss = (
                    pg_loss
                    -current_ent_coef * entropy_loss
                    + VF_COEF * v_loss
                    + loss_arch
                    + loss_angle
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                last_pg_loss = pg_loss.item()
                last_v_loss = v_loss.item()
                last_arch_loss = loss_arch.item()
                last_angle_loss = loss_angle.item()

        agent.q_layer.maybe_lock(update)

        with torch.no_grad():
            probs_t = torch.softmax(agent.q_layer.struct_weights, dim=0)
            top_prob, top_idx = torch.max(probs_t, dim=0)
            arch_probs_list = probs_t.cpu().numpy().tolist()
            mean_logstd = float(agent.actor_logstd.mean().item())

        avg_reward = float(np.mean(np.stack(raw_rewards_this_update)))
        reward_window.append(avg_reward)
        rolling_mean = float(np.mean(reward_window))

        if ep_total_count > 0:
            avg_success = ep_success_count / ep_total_count
        else:
            avg_success = 0.0
        success_window.append(avg_success)
        rolling_success = float(np.mean(success_window))

        sps = global_step / (time.time() - t0)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Update: {update:03d} | "
            f"R: {avg_reward:.2f} (u10: {rolling_mean:.2f}) | "
            f"SR: {avg_success*100:.1f}% (u10: {rolling_success*100:.1f}%) | "
            f"PG: {last_pg_loss:+.4f} | "
            f"V: {last_v_loss:.2f} | "
            f"logstd: {mean_logstd:+.3f} | "
            f"k={agent.q_layer.k_active} | "
            f"top={int(top_idx)}@{float(top_prob):.2f} | "
            f"lr={current_lr:.2e} | "
            f"SPS={sps:.0f}"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                update, global_step, current_lr,
                avg_reward, rolling_mean,
                avg_success, rolling_success,
                last_pg_loss, last_v_loss, last_arch_loss, last_angle_loss,
                mean_logstd,
                agent.q_layer.k_active, int(top_idx), float(top_prob),
                sps,
            ] + arch_probs_list)

        lr_scheduler.step()
	# Inside the main loop, after lr_scheduler.step()
        if update % CHECKPOINT_EVERY == 0 or update == MAX_UPDATES:
    	    ckpt_path = os.path.join(run_dir, f'checkpoint_update_{update}.pt')
    	    torch.save({
                'update': update,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),  # for resuming training
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'locked_arch_idx': agent.q_layer.locked_idx,
                'obs_rms_mean': obs_rms.mean,
                'obs_rms_var': obs_rms.var,
                'obs_rms_count': obs_rms.count,
                'rolling_reward': rolling_mean,
                'rolling_sr': rolling_success,
    	    }, ckpt_path)

    envs.close()
    # At the end of main(), or periodically during training:
    checkpoint = {
        # Model
        'agent_state_dict': agent.state_dict(),
        
        # Quantum-specific state that isn't in standard params
        'locked_arch_idx': agent.q_layer.locked_idx,
        'locked_arch_config': agent.q_layer.architectures[agent.q_layer.locked_idx]
                              if agent.q_layer.locked_idx is not None else None,
        
        # Observation normalizer CRITICAL for inference
        'obs_rms_mean': obs_rms.mean,
        'obs_rms_var': obs_rms.var,
        'obs_rms_count': obs_rms.count,
        
        # Architecture hyperparameters needed to reconstruct the model
        'config': {
            'obs_dim': obs_dim,
            'q_dim': Q_DIM,
            'action_dim': 4,
            'n_vqc_layers': 3,
            'hidden_dim': HIDDEN_DIM,
        },
        
        # Optional but useful
        'training_metadata': {
            'env_name': env_name,
            'final_update': MAX_UPDATES,
            'final_reward': rolling_mean,
            'final_sr': rolling_success,
            'arch_probs': probs_t.cpu().numpy().tolist(),
        },
    }
    print(f"\nTraining complete. Log: {csv_path}")


if __name__ == "__main__":
    main()