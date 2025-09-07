import torch.nn as nn
import torch

class Projector(nn.Module):
    def __init__(self, dim_struct=768, dim_sem=768, hidden1=1408, hidden2=2048, activation="gelu"):
        super().__init__()
        
        act_layer = nn.ReLU() if activation == "relu" else nn.GELU()

        # 768 -> 1408 -> 2048
        self.project_struct = nn.Sequential(
            nn.Linear(dim_struct, hidden1),
            act_layer,
            nn.Linear(hidden1, hidden2),
            act_layer
        )
        
        # 768 -> 1408 -> 2048
        self.project_sem = nn.Sequential(
            nn.Linear(dim_sem, hidden1),
            act_layer,
            nn.Linear(hidden1, hidden2),
            act_layer
        )

    def forward(self, struct_embed, sem_embed):
        s_proj = self.project_struct(struct_embed)
        m_proj = self.project_sem(sem_embed)
        return torch.cat([s_proj, m_proj], dim=-1)  