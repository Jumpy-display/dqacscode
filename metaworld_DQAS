import os
import time
import random
import csv
from collections import deque

# Limit thread usage for scientific computing libraries to prevent CPU thread contention,
# ensuring the GPU does the heavy lifting for the simulation.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Force MuJoCo to use EGL for headless GPU-accelerated rendering
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import gymnasium as gym
import metaworld

# Auto-detect hardware for tensor operations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))


# =============================================================================
# Per-feature running mean / std for observation normalization
# =============================================================================
class RunningMeanStd:
    """
    Computes a running mean and standard deviation using Welford's online algorithm.
    Crucial for stabilizing PPO by ensuring neural network inputs remain standardized
    even as the agent explores new state distributions over time.
    """
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(eps) # Small epsilon to prevent division by zero early on

    def update(self, x):
        """Updates the running statistics given a new batch of observations."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Calculate batch statistics
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        
        # Welford's algorithm for merging two sets of statistics
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        
        # M2 is the sum of squared differences from the mean
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot
        
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x):
        """Standardizes the observation and clips it to [-10, 10] to prevent extreme outliers."""
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)


# =============================================================================
# Reward scaling (per-env, running discounted-return std)
# =============================================================================
class RewardScaler:
    """
    Scales rewards by the running standard deviation of the discounted returns.
    Unlike observation normalization, rewards are NOT mean-centered (which would alter 
    the RL objective). Scaling stabilizes the critic network's value targets.
    """
    def __init__(self, num_envs, gamma=0.99):
        self.gamma = gamma
        self.num_envs = num_envs
        self.running_return = np.zeros(num_envs, dtype=np.float64)
        self.variance = np.full(num_envs, 1.0, dtype=np.float64)

    def scale(self, rewards):
        """Updates the running return standard deviation and scales the immediate reward."""
        self.running_return = self.running_return * self.gamma + rewards
        # Exponential moving average of the variance
        self.variance = 0.99 * self.variance + 0.01 * (self.running_return ** 2)
        return (rewards / (np.sqrt(self.variance) + 1e-8)).astype(np.float32)

    def reset(self, env_idx):
        """Resets the running return for a specific environment upon episode termination."""
        self.running_return[env_idx] = 0.0


# =============================================================================
# GPU-native quantum simulator
# =============================================================================
class QuantumOps:
    """
    A highly optimized, batched GPU statevector simulator.
    Instead of relying on slow CPU-bound quantum simulation frameworks, this implements
    direct tensor contractions to simulate quantum circuits over mini-batches instantly.
    """
    def __init__(self, n_qubits, device):
        self.n = n_qubits
        self.dim = 2 ** n_qubits # Size of the Hilbert space
        self.device = device
        indices = torch.arange(self.dim, device=device)

        # Precompute CNOT permutations for a linear entanglement chain
        self.cnot_perms = []
        for q in range(n_qubits - 1):
            control_bit = (indices >> (n_qubits - 1 - q)) & 1
            target_mask = 1 << (n_qubits - 1 - (q + 1))
            flipped = indices ^ target_mask
            # If control bit is 1, swap target amplitudes; otherwise, keep indices
            self.cnot_perms.append(torch.where(control_bit.bool(), flipped, indices))

        # Precompute Pauli-Z measurement signs for expected value calculations
        self.pauli_z_signs = torch.zeros(n_qubits, self.dim, device=device)
        for q in range(n_qubits):
            bit = (indices >> (n_qubits - 1 - q)) & 1
            # Maps bit 0 -> +1, bit 1 -> -1
            self.pauli_z_signs[q] = 1.0 - 2.0 * bit.float()

        # Define the Hadamard gate matrix
        self.H = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat, device=device) / (2 ** 0.5)

    def apply_gate(self, state, gate_2x2, qubit):
        """
        Applies a 2x2 single-qubit gate to the batched statevector using Einstein summation.
        Reshapes the state to expose the target qubit dimension, applies the gate, and flattens back.
        """
        batch = state.shape[0]
        a = 2 ** qubit
        b = 2 ** (self.n - qubit - 1)
        state = state.reshape(batch, a, 2, b)
        # b: batch, i/j: leading dims, k: target qubit, l: trailing dims
        state = torch.einsum('bij,bkjl->bkil', gate_2x2, state)
        return state.reshape(batch, self.dim)

    def apply_cnot_chain(self, state):
        """Applies a linear chain of CNOT gates (entanglement) using precomputed permutations."""
        for perm in self.cnot_perms:
            state = state[:, perm]
        return state

    def measure_all_z(self, state):
        """Calculates the expected value of the Pauli-Z operator for all qubits."""
        probs = state.real ** 2 + state.imag ** 2 # Born rule: probabilities are magnitude squared
        return (self.pauli_z_signs @ probs.T).T


# Single-qubit rotation gate matrices. 
# Batch dimensions are prepended so they broadcast correctly in apply_gate.
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
    """
    Differentiable Quantum Architecture Search (DQACS) module.
    Maintains a set of candidate Variational Quantum Circuit (VQC) architectures.
    During early training, it acts as a mixture model. As training progresses, 
    it evaluates and gradually prunes candidates until "locking" onto the optimal topology.
    """
    def __init__(self, n_qubits=16, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ops = None

        # Build the discrete search space of VQC architectures
        self.architectures = []
        for h in [True, False]:
            for enc in ['rx', 'ry']:
                for var in ['rx', 'ry', 'rz']:
                    self.architectures.append({
                        'hadamard': h, 'encoding_gate': enc, 'variational_gate': var,
                    })
        self.n_candidates = len(self.architectures)
        print(f"DiffQAS (Re-uploading) initialized with {self.n_candidates} candidate architectures.")

        # Trainable parameters for the quantum circuit
        self.circuit_params = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.enc_scale = nn.Parameter(torch.ones(n_layers, n_qubits)) # Feature scaling
        self.enc_bias = nn.Parameter(torch.zeros(n_layers, n_qubits)) # Feature shifting

        # The learnable architecture distribution (logits)
        self.struct_weights = nn.Parameter(torch.ones(self.n_candidates) / self.n_candidates)
        self.readout_theta = nn.Parameter(torch.zeros(n_qubits))

        # Search schedule parameters (how many archs run simultaneously)
        self.k_active = 3
        self.k_active_start = min(3, self.n_candidates)
        self.k_active_mid = 2
        self.k_active_final = 1
        self.k_stage1_updates = 20
        self.k_stage2_updates = 80

        # Architecture locking parameters
        self.lock_threshold = 0.95
        self.lock_patience_updates = 3
        self.lock_warmup_updates = 100
        self.lock_check_every = 1
        self.locked_idx = None
        self._lock_count = 0

    def _ensure_ops(self, device):
        """Lazy initialization of QuantumOps to ensure it lives on the correct device."""
        if self.ops is None or self.ops.device != device:
            self.ops = QuantumOps(self.n_qubits, device)

    def _run_single(self, x, arch):
        """Runs a forward pass for a single architecture candidate using data re-uploading."""
        B = x.shape[0]
        ops = self.ops
        dev = x.device

        # Initialize |0...0> state
        state = torch.zeros(B, ops.dim, dtype=torch.cfloat, device=dev)
        state[:, 0] = 1.0

        # Optional initial superposition layer
        if arch['hadamard']:
            h_exp = ops.H.unsqueeze(0).expand(B, -1, -1)
            for q in range(self.n_qubits):
                state = ops.apply_gate(state, h_exp, q)

        enc_fn = ROT_FNS[arch['encoding_gate']]
        var_fn = ROT_FNS[arch['variational_gate']]

        # VQC Layers: Encoding (Data Re-uploading) -> Entanglement -> Variational
        for l in range(self.n_layers):
            # Data Encoding
            for q in range(self.n_qubits):
                angle = self.enc_scale[l, q] * x[:, q] + self.enc_bias[l, q]
                state = ops.apply_gate(state, enc_fn(angle), q)
            # Entanglement
            state = ops.apply_cnot_chain(state)
            # Variational / Parametrized layer
            for q in range(self.n_qubits):
                state = ops.apply_gate(
                    state, var_fn(self.circuit_params[l, q].expand(B)), q
                )

        # Final trainable measurement readout rotation
        for q in range(self.n_qubits):
            state = ops.apply_gate(state, ry(self.readout_theta[q].expand(B)), q)

        return ops.measure_all_z(state)

    def forward(self, x):
        """
        Forward pass. If locked, executes only the best architecture.
        If searching, runs the top-k active architectures and returns their weighted sum.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        self._ensure_ops(x.device)

        # Fast path if the optimal architecture has been decided
        if self.locked_idx is not None:
            return self._run_single(x, self.architectures[int(self.locked_idx)])

        # Softmax over structural weights to get probability distribution
        probs = torch.softmax(self.struct_weights, dim=0)
        k = min(self.k_active, self.n_candidates)
        active = torch.topk(probs, k=k).indices

        B = x.shape[0]
        out = torch.zeros(B, self.n_qubits, device=x.device)
        w_sum = torch.zeros((), device=x.device)
        
        # Weighted sum of the top-k candidate outputs
        for i in active:
            out = out + probs[i] * self._run_single(x, self.architectures[int(i)])
            w_sum = w_sum + probs[i]
        return out / (w_sum + 1e-8)

    def update_search_schedule(self, update_idx: int) -> int:
        """Decays the number of active architectures over time to speed up training."""
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
        """
        Checks if the policy has converged on a dominant architecture. 
        If a candidate's probability stays above a threshold, the architecture is 'locked' in.
        """
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
    """
    Initializes weights orthogonally. A proven standard in Deep RL to preserve 
    gradient norms and ensure stable training early on.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# =============================================================================
# Continuous hybrid agent: tanh-bounded angles + skip connection
# =============================================================================
class HybridContinuousAgent(nn.Module):
    """
    Bridging quantum circuits with continuous environments like MetaWorld.
    Projects classical observations into bounded quantum phase angles, processes them 
    via DQACS, and combines the readout with a classical skip connection to output continuous actions.
    """
    def __init__(self, obs_dim=39, q_dim=16, action_dim=4, n_vqc_layers=3, hidden_dim=64):
        super().__init__()

        self.obs_dim = obs_dim
        self.q_dim = q_dim

        # Learnable projection 39 -> 16 (no bias, orthogonal init)
        # Maps the larger continuous observation space to the smaller quantum dimension.
        self.state_proj = nn.Linear(obs_dim, q_dim, bias=False)
        torch.nn.init.orthogonal_(self.state_proj.weight, gain=1.0)

        # NOTE: frozen LayerNorm is GONE. We now use tanh()*pi which:
        #   - bounds angles to [-pi, pi] without wrap-around aliasing,
        #   - gives a smooth, non-saturating gradient through state_proj,
        #   - lets state_proj learn the right *scale* per feature.
        self.q_layer = BatchedDiffQASLayer(n_qubits=q_dim, n_layers=n_vqc_layers)

        # Skip connection: post_vqc sees [quantum_features || raw_normalized_obs]
        # Prevents the policy from losing critical environment state information that 
        # might be bottlenecked through the 16 bounded quantum scalars.
        self.post_vqc = nn.Sequential(
            layer_init(nn.Linear(q_dim + obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Actor head outputs the mean of the action distribution. std=0.01 forces 
        # actions to be near zero initially (safe exploration).
        self.actor_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        # Critic head predicts the value of the state.
        self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # Slightly higher exploration ceiling (parameterized log standard deviation)
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))
        self.min_logstd = -3.0

    def get_action_and_value(self, norm_state, action=None):
        """
        Forward pass for both Actor and Critic.
        Returns the action, its log probability, distribution entropy, and state value.
        """
        # `norm_state` is already obs-normalized (mean 0 / std 1 per feature).
        projected = self.state_proj(norm_state)
        # Bound angles to [-pi, pi] for quantum gate stability
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
            # Reparameterization trick (rsample) allows gradients to flow back
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
        else:
            # Evaluation path (PPO update): recover the pre-tanh action
            # Clamp avoids atanh blowing up at exactly 1 or -1
            raw_action = torch.atanh(action.clamp(-0.999999, 0.999999))

        # Tanh-squashed Gaussian log-prob with Jacobian correction
        # Adjusts the probability density to account for the tanh transformation
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
    """
    Seamlessly wraps MetaWorld environments into the standard Gymnasium API.
    Handles the difference in output formats between the two libraries.
    """
    def __init__(self, env):
        super().__init__()
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        out = self._env.reset()
        # Gymnasium expects a tuple (obs, info)
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

def make_metaworld_env(env_name: str, seed: int, idx: int):
    """Closure to initialize parallel MetaWorld instances with distinct seeds."""
    def _init():
        ml1 = metaworld.ML1(env_name, seed=seed + idx)
        env = ml1.train_classes[env_name]()
        rng = random.Random(seed + idx)
        # Selects a random task variant associated with the environment
        task = rng.choice(ml1.train_tasks)
        env.set_task(task)
        return MetaWorldGymAdapter(env)
    return _init


# =============================================================================
# Hyperparameters
# =============================================================================
SEED = 67
NUM_ENVS = 16
Q_DIM = 12                    # was 10. Expanding the Hilbert space dimension.
HIDDEN_DIM = 128
STEPS_PER_UPDATE = 512
BATCH_SIZE = 512              # drop to 128 if OOM during k_active=3 search
EPOCHS = 10
GAMMA = 0.99                  # Discount factor
GAE_LAMBDA = 0.95             # GAE bias-variance tradeoff parameter
CLIP_COEF = 0.2               # PPO trust region clipping coefficient

ENT_COEF = 0.001              # was 0.0  -- keep some exploration pressure active
VF_COEF = 0.5                 # was implicitly 1.0 -- standard PPO value for value loss weight
LR = 3e-4
LAMBDA_ARCH = 0.02            # was 0.15 -- don't force premature arch lock-in (entropy penalty weight)
LAMBDA_ANGLE = 0.01           # Penalty weight to keep phase angles bounded

MAX_GRAD_NORM = 0.5
TARGET_KL = 0.08              # was 0.03 -- looser trust region for early stopping
MAX_UPDATES = 1500
LR_END_FACTOR = 0.3           # Linear LR decay target
CHECKPOINT_EVERY = 100

def seed_everything(seed):
    """Sets random seeds across all libraries for reproducibility."""
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

    env_name = "reach-v3" # The specific continuous control task

    print(f"Initializing {NUM_ENVS} parallel MetaWorld envs ({env_name})...")
    # Vectorized environments run in parallel, dramatically speeding up rollout collection
    envs = gym.vector.SyncVectorEnv(
        [make_metaworld_env(env_name, SEED, i) for i in range(NUM_ENVS)]
    )

    obs_dim = envs.single_observation_space.shape[0]
    print(f"obs_dim={obs_dim}, q_dim={Q_DIM}, post_vqc input={Q_DIM + obs_dim}")

    agent = HybridContinuousAgent(
        obs_dim=obs_dim, q_dim=Q_DIM, action_dim=4, n_vqc_layers=3, hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    # Separate architecture structural weights from the base network parameters 
    # so we can train them with a distinct, higher learning rate (1e-3 vs 3e-4)
    arch_params = [agent.q_layer.struct_weights]
    base_params = [p for n, p in agent.named_parameters() if n != "q_layer.struct_weights"]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': LR},
        {'params': arch_params, 'lr': 1e-3},
    ], eps=1e-5)

    # Anneal learning rate over the course of training
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

    # Initialize logging
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
        # Update DQAS k-active architecture count
        agent.q_layer.update_search_schedule(update)

        # Transition buffers
        # Buffer holds NORMALIZED observations (what the policy actually saw).
        b_states = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, obs_dim, device=DEVICE)
        b_actions = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, 4, device=DEVICE)
        b_logprobs = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=DEVICE)
        b_rewards = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=DEVICE)
        b_values = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=DEVICE)
        b_dones = torch.zeros(STEPS_PER_UPDATE, NUM_ENVS, device=DEVICE)

        raw_rewards_this_update = []
        raw_successes_this_update = []

        # --- Rollout (Data Collection Phase) ---
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

            # ADD: success shaping bonus BEFORE reward scaling
            # Enhances sparse success signals for continuous domains
            rewards_shaped = rewards.astype(np.float32) + 3.0 * successes
            raw_rewards_this_update.append(rewards.copy())  # log raw, not shaped
            raw_successes_this_update.append(successes)

            # Now scale the shaped rewards for the critic
            scaled_rewards = reward_scaler.scale(rewards_shaped)

            # Bootstrap on truncation (timeout, not termination)
            # If an episode hit the time limit, the value is not strictly 0. 
            # We estimate the remaining value using the critic.
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
                        # Add discounted bootstrapped value to the final step's reward
                        scaled_rewards[valid] = scaled_rewards[valid] + GAMMA * final_v

            # Terminations represent true environment ends (success/failure)
            gae_dones = terminations.astype(np.float32)

            episode_over = terminations | truncations
            for i in np.where(episode_over)[0]:
                reward_scaler.reset(i) # Reset variance tracker per-env on completion

            # Save data to buffers
            b_states[step] = norm_state
            b_actions[step] = action
            b_logprobs[step] = logprob
            b_rewards[step] = torch.from_numpy(scaled_rewards).to(DEVICE)
            b_values[step] = value.squeeze(-1)
            b_dones[step] = torch.from_numpy(gae_dones).to(DEVICE)

            next_obs_raw = next_obs_raw_new

        # --- GAE (Generalized Advantage Estimation) ---
        # Calculates how much better an action was compared to the critic's baseline expectation.
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
                # TD Error
                delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
                # Exponential moving average of TD Errors
                lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + b_values

        # Right after the GAE loop, before flattening
        print(f"  returns: u={returns.mean():.3f} o={returns.std():.3f} | "
              f"adv: u={advantages.mean():.3f} o={advantages.std():.3f}")
        
        # Flatten the batch to prepare for network updates
        f_states = b_states.reshape(-1, obs_dim)
        f_actions = b_actions.reshape(-1, 4)
        f_logprobs = b_logprobs.reshape(-1)
        f_advantages = advantages.reshape(-1)
        f_returns = returns.reshape(-1)
        N = f_states.shape[0]

        # Per-batch advantage normalization (more stable than per-minibatch)
        f_advantages = (f_advantages - f_advantages.mean()) / (f_advantages.std() + 1e-8)

        # --- PPO update (Optimization Phase) ---
        last_pg_loss = last_v_loss = last_arch_loss = last_angle_loss = 0.0
        stop_early = False

        for epoch in range(EPOCHS):
            if stop_early:
                break
            perm = torch.randperm(N, device=DEVICE) # Shuffle data
            for start in range(0, N, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = perm[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    f_states[mb_inds], f_actions[mb_inds]
                )

                # KL Divergence early stopping (prevents catastrophic policy updates)
                with torch.no_grad():
                    approx_kl = (f_logprobs[mb_inds] - newlogprob).mean().item()
                if approx_kl > TARGET_KL:
                    stop_early = True
                    break

                # Probability ratio for PPO objective
                logratio = newlogprob - f_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = f_advantages[mb_inds]

                # PPO Clipped Surrogate Objective
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic Loss (Mean Squared Error)
                v_loss = 0.5 * ((newvalue.squeeze() - f_returns[mb_inds]) ** 2).mean()
                
                # Policy Entropy Loss (encourages exploration)
                entropy_loss = entropy.mean()

                # --- Architecture Search Loss ---
                arch_probs = torch.softmax(agent.q_layer.struct_weights, dim=0)
                # NOTE: this is -entropy, so minimizing the loss DECREASES arch entropy
                # (i.e., pushes toward one-hot distribution). LAMBDA_ARCH=0.02 makes this gentle.
                loss_arch = LAMBDA_ARCH * -torch.sum(
                    arch_probs * torch.log(arch_probs + 1e-10)
                )

                # --- Quantum Angle Penalty ---
                # Soft penalty to prevent trainable parameters from diverging far beyond +/- pi
                cp = agent.q_layer.circuit_params
                penalty = F.relu(cp - torch.pi) + F.relu(-cp - torch.pi)
                loss_angle = LAMBDA_ANGLE * torch.sum(penalty ** 2)

                # Combined Objective
                loss = (
                    pg_loss
                    - ENT_COEF * entropy_loss
                    + VF_COEF * v_loss
                    + loss_arch
                    + loss_angle
                )

                optimizer.zero_grad(set_to_none=True) # set_to_none=True is slightly faster
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM) # Exploding gradient protection
                optimizer.step()

                last_pg_loss = pg_loss.item()
                last_v_loss = v_loss.item()
                last_arch_loss = loss_arch.item()
                last_angle_loss = loss_angle.item()

        # Check if the architecture search should lock
        agent.q_layer.maybe_lock(update)

        # Logging stats
        with torch.no_grad():
            probs_t = torch.softmax(agent.q_layer.struct_weights, dim=0)
            top_prob, top_idx = torch.max(probs_t, dim=0)
            arch_probs_list = probs_t.cpu().numpy().tolist()
            mean_logstd = float(agent.actor_logstd.mean().item())

        avg_reward = float(np.mean(np.stack(raw_rewards_this_update)))
        reward_window.append(avg_reward)
        rolling_mean = float(np.mean(reward_window))

        avg_success = float(np.mean(np.stack(raw_successes_this_update)))
        success_window.append(avg_success)
        rolling_success = float(np.mean(success_window))

        # Steps per second
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
          # Save extensive checkpoint including normalization stats
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
    
    # At the end of main(), save the final production checkpoint
    checkpoint = {
        # Model
        'agent_state_dict': agent.state_dict(),
        
        # Quantum-specific state that isn't in standard params
        'locked_arch_idx': agent.q_layer.locked_idx,
        'locked_arch_config': agent.q_layer.architectures[agent.q_layer.locked_idx]
                              if agent.q_layer.locked_idx is not None else None,
        
        # Observation normalizer CRITICAL for inference
        # If these are lost, the agent will see completely alien states during testing.
        'obs_rms_mean': obs_rms.mean,
        'obs_rms_var': obs_rms.var,
        'obs_rms_count': obs_rms.count,
        
        # Architecture hyperparameters needed to reconstruct the model later
        'config': {
            'obs_dim': obs_dim,
            'q_dim': Q_DIM,
            'action_dim': 4,
            'n_vqc_layers': 3,
            'hidden_dim': HIDDEN_DIM,
        },
        
        # Optional but useful metadata
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
