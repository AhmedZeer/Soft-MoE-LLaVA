import torch
from torch._C import dtype
from torch.autograd import forward_ad
import torch.nn as nn
import re
import torch.nn.functional as F

class SoftMoE(nn.Module):
    def __init__(self, batch_size:int,
                 patch_dim:int,
                 d:int, n:int, p:int,
                 experts:nn.ModuleList,
                 hidden_size:int):
        super().__init__()

        self.n = n
        self.patch_dim = patch_dim
        # self.batch_size = batch_size
        self.p = p
        # self.m = batch_size * patch_dim
        self.phi = nn.Linear(d, n*p)
        self.experts = experts
        self.hidden_size = hidden_size

    def forward(self, x):

        # print("@ SoftMoE.forward(x)")
        # print("0-"*5)
        # print("  -> x.shape:", x.shape)
        # print("  -> self.patch_dim:", self.patch_dim)
        # print("  -> x.dtype:", x.dtype)

        # x : [8, 576, 1024]
        # y : [8, 576, 4096]

        # x : [batch_size, hidden_size, in_dim]
        # y : [batch_size, hidden_size, d]

        # if x.shape[0] < self.batch_size:
        #     repeat_num = self.batch_size - x.shape[0]
        #     last_slice = x[-1].unsqueeze(0).repeat(repeat_num,1,1)
        #     x = torch.cat([x, last_slice], dim=0)
        #     print("Amended.")

        x_phi = self.phi(x)
        # print("  -> x_phi.shape:", x_phi.shape)
        # x_phi = x_phi.reshape(self.m,self.n,self.p)
        x_phi = x_phi.reshape(-1, self.patch_dim ,self.n,self.p)
        # print("  -> x_phi.shape:", x_phi.shape)

        # print("0-"*5)
        # print("SoftMoE.forward(x) @")
        assert x_phi.shape[-2] == self.n

        # t : batch_size
        # p : patch_dim
        # H : hidden_size

        # [batch_size, patch_dim ,n,p]
        D = x_phi.softmax(dim=3)
        C = x_phi.softmax(dim=2)


        X_tilde = torch.einsum("tznp,tzd->npd", D, x)
        # print("  -> x_tilde:", X_tilde)
        # print("  -> x_tilde.shape:", X_tilde.shape)
        torch.clamp_(X_tilde, -33000, 65000)
        X_tilde = F.layer_norm(X_tilde, [X_tilde.shape[-1]])
        # print("  -> layer_norm(x_tilde):", X_tilde)
        X_tilde = F.tanh(X_tilde)
        # print("  -> tanh(x_tilde):", X_tilde)
        #
        # print("  -> x_tilde.shape:", X_tilde.shape)
        # print("  -> x_tilde.dtype:", X_tilde.dtype)

        # Train:
        Y_tilde = torch.ones([X_tilde.shape[0], X_tilde.shape[1], self.hidden_size], dtype=torch.float16).to(X_tilde.device)

        # Inference
        # Y_tilde = torch.ones([X_tilde.shape[0], X_tilde.shape[1], self.hidden_size], dtype=torch.float16).to(X_tilde.device)
        for i, expert in enumerate(self.experts):
            # print("Expert:", expert)
            Y_tilde[i, :, :] = expert(X_tilde[i,:,:]) #[n, p, d]


        # print("  -> y_tilde.shape:", Y_tilde.shape)
        # print("  -> y_tilde.dtype:", Y_tilde.dtype)
        # print("  -> D.dtype:", D.dtype)
        # print("  -> C.dtype:", C.dtype)

        # Y = torch.einsum('npd,mnp->md', Y_tilde, C)

        # [batch_size, patch_dim, n, p] X [n, p, hidden_size] -> [batch_size, patch_dim, hidden_size]
        Y = torch.einsum('tznp,npH->tzH', C, Y_tilde)
        # print("  -> y.dtype:", Y.dtype)
        # print("-0"*5)
        # print("\n"*5)
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

class TokenDropMLP(nn.Module):
    def __init__(self, mlp_depth, mm_hidden_size, hidden_size, is_token_drop):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, requires_grad=True))
        self.mlp = create_expert(mlp_depth, mm_hidden_size, hidden_size)
    def forward(self, x):
        # [batch_size, patch_num, patch_dim]
        print("@ TokenDropMLP")
        print("Shape Before Drop:", x.shape)
        last_hidden_state_normalized = torch.nn.functional.normalize(x, dim=-1)

        # Calculate cosine similarity using the normalized embeddings
        similarity = last_hidden_state_normalized @ last_hidden_state_normalized.transpose(-1,-2)
        sums = torch.sum(similarity, -1)
        probas = torch.softmax(sums, -1)
        self.alpha.data.clamp_(0.4, 1.0)
        _,indices = probas.topk((self.alpha * probas.shape[-1]).to(torch.int),largest=False)
        x = torch.index_select(x, -2, indices.squeeze())
        print("Shape After Drop:", x.shape)
        print("@ TokenDropMLP")
        return self.mlp(x)

def create_expert(mlp_depth, mm_hidden_size, hidden_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modules = [nn.Linear(mm_hidden_size, hidden_size)]
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())

        # Train:
        # modules.append(nn.Linear(hidden_size, hidden_size).to(device))

        # Inference:
        modules.append(nn.Linear(hidden_size, hidden_size).to_empty(device=device))
    return nn.Sequential(*modules)

def build_vision_projector(config, delay_load=False, **kwargs):

    projector_type = getattr(config, 'mm_projector_type', 'linear')
    is_soft_moe = getattr(config, 'soft_moe', False)
    batch_size = getattr(config, 'moe_batch_size', 1)
    is_token_drop = getattr(config, 'token_drop', False)
    vision_tower_name = getattr(config, 'mm_vision_tower', "").lower()
    print(config)
    print("is_soft_moe:",is_soft_moe)
    print("is_token_drop:",is_token_drop)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:

        mlp_depth = int(mlp_gelu_match.group(1))
        mm_hidden_size = config.mm_hidden_size
        hidden_size = config.hidden_size

        if is_soft_moe:
            # print("0-"*10)
            # print("SoftMoE Layer Init.")
            # print("  -> mm_hidden_size:",mm_hidden_size)
            # print("  -> hidden_size:", hidden_size)
            # print("-0"*10)

            experts_n = config.experts_n
            slots_n = config.slots_n
            experts = nn.ModuleList([create_expert(mlp_depth,mm_hidden_size, hidden_size) for _ in range(experts_n)])

            if "clip" in vision_tower_name:
                patch_dim = 576
            elif "dino" in vision_tower_name:
                patch_dim = 256
            elif "siglip" in vision_tower_name:
                patch_dim = 728
            else:
                raise Exception("Not Supported Vision Tower:", vision_tower_name)
            soft_moe = SoftMoE(batch_size=batch_size,
                               patch_dim=patch_dim,
                               d=mm_hidden_size, # Projector Hidden Size
                               n=experts_n,
                               p=slots_n,
                               experts=experts,
                               hidden_size=hidden_size)
            print("\nRETURN SOFT MOE.\n")
            return soft_moe

        elif is_token_drop:
            return TokenDropMLP(mlp_depth=mlp_depth,
                          mm_hidden_size=mm_hidden_size,
                          hidden_size=hidden_size,
                          is_token_drop = is_token_drop)
        else:
            return create_expert(mlp_depth=mlp_depth,
                          mm_hidden_size=mm_hidden_size,
                          hidden_size=hidden_size)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


"""
@ SoftMoE.forward(SIGLIP)
0-0-0-0-0-
  -> x.shape: torch.Size([8, 728, 1152])
  -> self.patch_dim: 728


@ SoftMoE.forward(CLIP)
0-0-0-0-0-
  -> x.shape: torch.Size([8, 576, 1024])
  -> self.patch_dim: 576

@ SoftMoE.forward(DINOv2)
0-0-0-0-0-
  -> x.shape: torch.Size([8, 256, 768])
  -> self.patch_dim: 256
"""
