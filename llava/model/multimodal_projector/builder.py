import torch
from torch._C import dtype
import torch.nn as nn
import re
import torch.nn.functional as F

class SoftMoE(nn.Module):
    def __init__(self, patch_size:int,
                 patch_dim:int,
                 d:int, n:int, p:int,
                 experts:nn.ModuleList,
                 hidden_size:int):
        super().__init__()

        self.n = n
        self.patch_dim = patch_dim
        self.patch_size = patch_size
        self.p = p
        self.m = patch_size * patch_dim
        self.phi = nn.Linear(d, n*p)
        self.experts = experts
        self.hidden_size = hidden_size

    def forward(self, x):

        print("\n"*5)
        print("0-"*5)
        print("@ SoftMoE.forward(x):")
        print("  -> x.shape:", x.shape)
        print("  -> x.dtype:", x.dtype)

        if x.shape[0] < 30:
            repeat_num = self.patch_size - x.shape[0]
            last_slice = x[-1].unsqueeze(0).repeat(repeat_num,1,1)
            x = torch.cat([x, last_slice], dim=0)
            print("Amended.")

        x_phi = self.phi(x)
        # x_phi = x_phi.reshape(self.m,self.n,self.p)
        x_phi = x_phi.reshape(self.patch_size, self.patch_dim ,self.n,self.p)
        print("  -> x_phi.shape:", x_phi.shape)

        assert x_phi.shape[-2] == self.n

        # #[m,n,p]
        # D = x_phi.softmax(dim=2)
        # C = x_phi.softmax(dim=1)
        # # [n,p,m]
        # D.transpose_(0,-1)
        # D.transpose_(0,1)
        # # [n,p,m] X [m,d] -> [n, p, d]
        # X_tilde = D @ x
        # print(e)

        # t : patch_size
        # p : patch_dim
        # H : hidden_size

        # [patch_size, patch_dim ,n,p]
        D = x_phi.softmax(dim=3)
        C = x_phi.softmax(dim=2)


        X_tilde = torch.einsum("tznp,tzd->npd", D, x)
        print("  -> x_tilde:", X_tilde)
        print("  -> x_tilde.shape:", X_tilde.shape)
        torch.clamp_(X_tilde, -33000, 65000)
        X_tilde = F.layer_norm(X_tilde, [X_tilde.shape[-1]])
        print("  -> layer_norm(x_tilde):", X_tilde)
        X_tilde = F.tanh(X_tilde)
        print("  -> tanh(x_tilde):", X_tilde)

        print("  -> x_tilde.shape:", X_tilde.shape)
        print("  -> x_tilde.dtype:", X_tilde.dtype)

        # Train:
        Y_tilde = torch.ones([X_tilde.shape[0], X_tilde.shape[1], self.hidden_size], dtype=torch.float16).to(X_tilde.device)

        # Inference
        # Y_tilde = torch.ones([X_tilde.shape[0], X_tilde.shape[1], self.hidden_size], dtype=torch.float16).to(X_tilde.device)
        for i, expert in enumerate(self.experts):
            # print("Expert:", expert)
            Y_tilde[i, :, :] = expert(X_tilde[i,:,:]) #[n, p, d]


        print("  -> y_tilde.shape:", Y_tilde.shape)
        print("  -> y_tilde.dtype:", Y_tilde.dtype)
        print("  -> D.dtype:", D.dtype)
        print("  -> C.dtype:", C.dtype)

        # Y = torch.einsum('npd,mnp->md', Y_tilde, C)

        # [patch_size, patch_dim, n, p] X [n, p, hidden_size] -> [patch_size, patch_dim, hidden_size]
        Y = torch.einsum('tznp,npH->tzH', C, Y_tilde)
        print("  -> y.dtype:", Y.dtype)
        print("-0"*5)
        print("\n"*5)
        return Y

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def create_expert(mlp_depth, mm_hidden_size, hidden_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modules = [nn.Linear(mm_hidden_size, hidden_size)]
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())

        # Train:
        modules.append(nn.Linear(hidden_size, hidden_size).to(device))

        # Inference:
        # modules.append(nn.Linear(hidden_size, hidden_size).to_empty(device=device))
    return nn.Sequential(*modules)

def build_vision_projector(config, delay_load=False, **kwargs):

    projector_type = getattr(config, 'mm_projector_type', 'linear')
    is_soft_moe = getattr(config, 'soft_moe', False)
    print(config)
    print("is_soft_moe:",is_soft_moe)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:

        mlp_depth = int(mlp_gelu_match.group(1))
        mm_hidden_size = config.mm_hidden_size
        hidden_size = config.hidden_size

        if is_soft_moe:
            print("0-"*10)
            print("SoftMoE Layer Init.")
            print("  -> mm_hidden_size:",mm_hidden_size)
            print("  -> hidden_size:", hidden_size)
            print("-0"*10)

            experts_n = config.experts_n
            slots_n = config.slots_n
            experts = nn.ModuleList([create_expert(mlp_depth,mm_hidden_size, hidden_size) for _ in range(experts_n)])
            soft_moe = SoftMoE(patch_size=32,
                               patch_dim=576,
                               # d=hidden_size,
                               d=mm_hidden_size,
                               n=experts_n,
                               p=slots_n,
                               experts=experts,
                               hidden_size=hidden_size)
            print("\nRETURN SOFT MOE.\n")
            return soft_moe

        else:
            return create_expert(mlp_depth=mlp_depth,
                          mm_hidden_size=mm_hidden_size,
                          hidden_size=hidden_size)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
