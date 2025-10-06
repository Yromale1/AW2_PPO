import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, action_dim, embed_dim=16, map_codes=64, units_codes=128):
        super().__init__()

        # Embeddings
        self.map_embed = nn.Embedding(map_codes, embed_dim)
        self.unit_embed = nn.Embedding(units_codes, embed_dim)

        # CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))  # fixed size regardless of map size
        )

        conv_out_dim = 64 * 5 * 5

        # Optional: extra scalar features (e.g. funds, turn, CO id, etc.)
        self.extra_fc = nn.Linear(1, 32)  # adjust if you add extra feats

        # Merge map + extras
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )

        # Policy & value heads
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)


    def forward(self, obs_tuple):
        """
        obs_tuple: (map_grid, unit_grid, extra_feats)
        map_grid: [B,H,W] LongTensor
        unit_grid: [B,H,W] LongTensor
        extra_feats: [B,N] FloatTensor
        """
        map_grid, unit_grid, extra_feats = obs_tuple

        # Embed
        m = self.map_embed(map_grid).permute(0,3,1,2)   # [B,embed,H,W]
        u = self.unit_embed(unit_grid).permute(0,3,1,2) # [B,embed,H,W]

        # Stack and conv
        x = torch.cat([m, u], dim=1)  # [B,2*embed,H,W]
        x = self.conv(x)
        x = x.flatten(1)  # [B,conv_out_dim]

        # Extras
        if extra_feats is not None:
            extra = F.relu(self.extra_fc(extra_feats))
            x = torch.cat([x, extra], dim=-1)

        # Dense trunk
        x = self.fc(x)

        # Heads
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs_tuple, deterministic=False):
        """
        obs_tuple: (map_grid, unit_grid, extra_feats)
          - map_grid: LongTensor [B,H,W]
          - unit_grid: LongTensor [B,H,W]
          - extra_feats: FloatTensor [B,extra_dim]

        Returns:
          action_numpy, log_prob, entropy, value
        """
        logits, value = self.forward(obs_tuple)  # forward expects the tuple
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.cpu().numpy(), log_prob, entropy, value

    def evaluate_actions(self, obs_tuple, actions):
        """
        obs_tuple: (map_grid, unit_grid, extra_feats)
        actions: LongTensor [B]
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long, device=device)

        logits, value = self.forward(obs_tuple)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value

