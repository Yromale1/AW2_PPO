import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from advancewars_env import AdvanceWarsEnv
from model import PolicyNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Observation preprocessing
# -----------------------------
def preprocess_obs(obs):
    """
    Convert AdvanceWarsEnv obs dict into tensors for PolicyNet.
    """
    map_grid = torch.tensor(obs["map"], dtype=torch.long, device=device).unsqueeze(0)
    unit_grid = torch.tensor(obs["units"], dtype=torch.long, device=device).unsqueeze(0)
    extras = torch.tensor(obs["extras"], dtype=torch.float32, device=device).unsqueeze(0)
    return map_grid, unit_grid, extras

# -----------------------------
# Rollout namedtuple
# -----------------------------
Rollout = namedtuple('Rollout', [
    'obs', 'actions', 'log_probs', 'rewards', 'dones', 'values', 'advantages', 'returns'
])

# -----------------------------
# PPO Agent
# -----------------------------
class PPOTrainer:
    def __init__(self, model, env, lr=2.5e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                 epochs=4, minibatch_size=64, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.model = model.to(device)
        self.env = env
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.eps = 1e-8

    # -----------------------------
    # Collect trajectories
    # -----------------------------
    def collect_trajectories(self, horizon=128, render=False):
        obs_list, actions_list, logp_list, rewards_list, dones_list, values_list = [], [], [], [], [], []

        obs, info = self.env.reset()
        for t in range(horizon):
            if render:
                self.env.render()

            map_grid, unit_grid, extras = preprocess_obs(obs)
            action_np, log_prob, entropy, value = self.model.get_action_and_value(
                (map_grid, unit_grid, extras)
            )
            action = int(action_np[0])

            next_obs, reward, done, info = self.env.step(action)

            obs_list.append(obs)
            actions_list.append(action)
            logp_list.append(log_prob.detach().cpu().item())
            values_list.append(value.detach().cpu().item())
            rewards_list.append(float(reward))
            dones_list.append(float(done))

            obs = next_obs
            if done:
                obs, info = self.env.reset()

        # bootstrap last value
        map_grid, unit_grid, extras = preprocess_obs(obs)
        _, _, _, last_value = self.model.get_action_and_value((map_grid, unit_grid, extras))
        last_value = last_value.detach().cpu().numpy()[0]

        # Compute GAE
        T = len(rewards_list)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones_list[t]
            next_value = last_value if t == T - 1 else values_list[t + 1]
            delta = rewards_list[t] + self.gamma * next_value * mask - values_list[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values_list, dtype=np.float32)

        rollout = Rollout(
            obs=obs_list,
            actions=np.array(actions_list, dtype=np.int64),
            log_probs=np.array(logp_list, dtype=np.float32),
            rewards=np.array(rewards_list, dtype=np.float32),
            dones=np.array(dones_list, dtype=np.float32),
            values=np.array(values_list, dtype=np.float32),
            advantages=advantages,
            returns=returns
        )

        return rollout

    # -----------------------------
    # PPO update
    # -----------------------------
    def update(self, rollout: Rollout):
        # Convert obs to tensors
        map_grids = torch.stack([preprocess_obs(o)[0].squeeze(0) for o in rollout.obs]).to(device)
        unit_grids = torch.stack([preprocess_obs(o)[1].squeeze(0) for o in rollout.obs]).to(device)
        extras = torch.stack([preprocess_obs(o)[2].squeeze(0) for o in rollout.obs]).to(device)

        b_obs = (map_grids, unit_grids, extras)
        b_actions = torch.tensor(rollout.actions, dtype=torch.long, device=device)
        b_old_logprobs = torch.tensor(rollout.log_probs, dtype=torch.float32, device=device)
        b_returns = torch.tensor(rollout.returns, dtype=torch.float32, device=device)
        b_advantages = torch.tensor(rollout.advantages, dtype=torch.float32, device=device)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + self.eps)

        batch_size = b_actions.shape[0]

        for epoch in range(self.epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                mb_map = b_obs[0][mb_idx]
                mb_units = b_obs[1][mb_idx]
                mb_extras = b_obs[2][mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_logprobs = b_old_logprobs[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_advantages = b_advantages[mb_idx]

                new_logprob, entropy, value_pred = self.model.evaluate_actions(
                    (mb_map, mb_units, mb_extras), mb_actions
                )
                ratio = torch.exp(new_logprob - mb_old_logprobs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = F.mse_loss(value_pred.squeeze(-1), mb_returns)
                entropy_loss = torch.mean(entropy)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

# -----------------------------
# Training example
# -----------------------------
env = AdvanceWarsEnv()
action_dim = env.wrapped.action_space.n  # discrete actions
policy = PolicyNet(action_dim)
trainer = PPOTrainer(policy, env)

for update in range(1000):
    rollout = trainer.collect_trajectories(horizon=10192, render=True)
    p_loss, v_loss, e_loss = trainer.update(rollout)
    print(f"Update {update}: Policy {p_loss:.3f}, Value {v_loss:.3f}, Entropy {e_loss:.3f}")
