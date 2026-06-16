from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from src.estgel_eam import build_nested_subgraph_tensor


class EAMOutput(NamedTuple):
    """Outputs from the Edge Attention Aggregation block at one timestep."""

    adjacency_refined: torch.Tensor
    layer_attention: torch.Tensor
    nested_subgraphs: torch.Tensor
    edge_index: torch.Tensor
    edge_weights: torch.Tensor


def adjacency_from_sparse_torch(
    edge_index: torch.Tensor,
    edge_t: torch.Tensor,
    t_idx: int,
    N: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Reconstruct dense A^t on the requested device."""
    device = device or edge_index.device
    A_t = torch.zeros(N, N, dtype=torch.float32, device=device)
    mask = edge_t == t_idx
    if mask.any():
        src = edge_index[0, mask]
        dst = edge_index[1, mask]
        A_t[src, dst] = 1.0
    return A_t


def nested_subgraph_tensor_torch(
    edge_index: torch.Tensor,
    edge_t: torch.Tensor,
    t_idx: int,
    N: int,
    K: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build nested subgraph tensor (N, N, K+1) from PyG sparse edges.

    Decomposition runs in NumPy; the returned tensor is placed on `device`.
    """
    device = device or edge_index.device
    edge_src = edge_index[0].detach().cpu().numpy()
    edge_dst = edge_index[1].detach().cpu().numpy()
    edge_t_np = edge_t.detach().cpu().numpy()
    nested_np = build_nested_subgraph_tensor(edge_src, edge_dst, edge_t_np, t_idx, N, K)
    return torch.from_numpy(nested_np).to(device=device, dtype=torch.float32)


def split_batch_graphs(batch: Batch) -> list[Data]:
    """Split a PyG Batch into individual Data objects."""
    graphs: list[Data] = []
    for i in range(batch.num_graphs):
        node_mask = batch.batch == i
        node_ids = node_mask.nonzero(as_tuple=True)[0]
        offset = int(node_ids.min().item())

        edge_mask = batch.batch[batch.edge_index[0]] == i
        edge_index = batch.edge_index[:, edge_mask] - offset

        graphs.append(
            Data(
                x=batch.x[node_mask],
                alive_mask=batch.alive_mask[node_mask],
                edge_index=edge_index,
                edge_t=batch.edge_t[edge_mask],
            )
        )
    return graphs


class EdgeAttentionAggregation(nn.Module):
    """
    ESTGEL Edge Attention Aggregation (Phase 2).

    For each edge (i, j), the module receives a K+1-dimensional signature
    (presence across nested subgraphs G^t_0 ... G^t_K) and learns softmax
    attention weights α over layers:

        Â^t[i, j] = Σ_k α_{ij,k} · nested[i, j, k]

    A dense forward pass materialises all N² pairs (ESTGEL baseline cost).
    A sparse forward pass scores only edges present in G^t_0.
    """

    def __init__(self, K: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if K < 1:
            raise ValueError("K must be >= 1.")
        self.K = K
        self.num_layers = K + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.num_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_layers),
        )

    def _layer_attention(self, edge_signatures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_signatures: (..., K+1) nested-layer values per edge/pair.

        Returns:
            weights: (...,) attention-refined edge strength
            alpha: (..., K+1) layer attention weights
        """
        logits = self.edge_mlp(edge_signatures)
        alpha = F.softmax(logits, dim=-1)
        weights = (alpha * edge_signatures).sum(dim=-1)
        return weights, alpha

    def forward_dense(self, nested: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dense ESTGEL-style aggregation over all node pairs.

        Args:
            nested: (N, N, K+1) or (B, N, N, K+1)

        Returns:
            adjacency_refined: (N, N) or (B, N, N)
            layer_attention: same leading dims + (K+1,)
        """
        if nested.dim() == 3:
            nested = nested.unsqueeze(0)

        B, N, _, L = nested.shape
        if L != self.num_layers:
            raise ValueError(f"Expected K+1={self.num_layers} layers, got {L}.")

        flat = nested.reshape(B, N * N, L)
        weights, alpha = self._layer_attention(flat)
        adjacency = weights.reshape(B, N, N)
        layer_attention = alpha.reshape(B, N, N, L)

        if B == 1:
            return adjacency.squeeze(0), layer_attention.squeeze(0)
        return adjacency, layer_attention

    def forward_sparse(
        self,
        nested: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse aggregation on edges present in G^t_0 (and their nested signatures).

        Args:
            nested: (N, N, K+1)
            edge_index: (2, E)

        Returns:
            edge_weights: (E,)
            layer_attention: (E, K+1)
        """
        src, dst = edge_index
        edge_signatures = nested[src, dst, :]
        return self._layer_attention(edge_signatures)

    def forward(
        self,
        nested: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        *,
        mode: str = "sparse",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "dense":
            return self.forward_dense(nested)
        if mode == "sparse":
            if edge_index is None:
                raise ValueError("edge_index is required for sparse mode.")
            return self.forward_sparse(nested, edge_index)
        raise ValueError(f"Unknown mode '{mode}'. Use 'dense' or 'sparse'.")


class ESTGELEdgeAttentionBlock(nn.Module):
    """
    Phase-2 block wired to EpicEmbryoDataset / PyG Data objects.

    Pipeline per timestep t:
      sparse edges -> nested subgraph tensor (N, N, K+1)
                   -> EdgeAttentionAggregation
                   -> refined adjacency + edge weights for downstream DRL
    """

    def __init__(self, K: int = 11, hidden_dim: int = 32) -> None:
        super().__init__()
        self.K = K
        self.eaa = EdgeAttentionAggregation(K=K, hidden_dim=hidden_dim)

    def forward_timestep(
        self,
        data: Data,
        t_idx: int,
        *,
        mode: str = "sparse",
        device: torch.device | None = None,
    ) -> EAMOutput:
        device = device or next(self.parameters()).device
        N = data.x.shape[0]
        edge_index = data.edge_index.to(device)
        edge_t = data.edge_t.to(device)

        nested = nested_subgraph_tensor_torch(edge_index, edge_t, t_idx, N, self.K, device=device)

        edges_at_t = edge_index[:, edge_t == t_idx]
        if edges_at_t.numel() == 0:
            empty_weights = torch.empty(0, device=device)
            empty_alpha = torch.empty(0, self.K + 1, device=device)
            return EAMOutput(
                adjacency_refined=torch.zeros(N, N, device=device),
                layer_attention=empty_alpha,
                nested_subgraphs=nested,
                edge_index=edges_at_t,
                edge_weights=empty_weights,
            )

        if mode == "dense":
            adjacency_refined, layer_attention = self.eaa.forward_dense(nested)
            edge_weights, edge_alpha = self.eaa.forward_sparse(nested, edges_at_t)
            return EAMOutput(
                adjacency_refined=adjacency_refined,
                layer_attention=layer_attention,
                nested_subgraphs=nested,
                edge_index=edges_at_t,
                edge_weights=edge_weights,
            )

        edge_weights, edge_alpha = self.eaa.forward_sparse(nested, edges_at_t)
        adjacency_refined = torch.zeros(N, N, device=device)
        adjacency_refined[edges_at_t[0], edges_at_t[1]] = edge_weights
        return EAMOutput(
            adjacency_refined=adjacency_refined,
            layer_attention=edge_alpha,
            nested_subgraphs=nested,
            edge_index=edges_at_t,
            edge_weights=edge_weights,
        )

    def forward_samples(
        self,
        samples: list[Data],
        t_idx: int,
        *,
        mode: str = "sparse",
        device: torch.device | None = None,
    ) -> list[EAMOutput]:
        """Run Phase 2 on a list of PyG Data objects (variable N/T safe)."""
        return [
            self.forward_timestep(sample, t_idx, mode=mode, device=device)
            for sample in samples
        ]

    def forward_batch(
        self,
        batch: Batch,
        t_idx: int,
        *,
        mode: str = "sparse",
        device: torch.device | None = None,
    ) -> list[EAMOutput]:
        """
        Run Phase 2 on a PyG Batch.

        Note: use ``use_global_index=True`` in EpicEmbryoDataset when batching,
        because local-mode embryos have different sequence lengths (T).
        """
        return self.forward_samples(split_batch_graphs(batch), t_idx, mode=mode, device=device)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Phase 3: Dynamic Relation Learning (DRL) — ESTGEL eq. (8)–(11)
# ---------------------------------------------------------------------------


def _bilinear_gate_field(
    A1: torch.Tensor,
    A2: torch.Tensor,
    r1: torch.Tensor,
    c1: torch.Tensor,
    r2: torch.Tensor,
    c2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Σ_n r1_n A1_{i,n} + Σ_n c1_n A1_{n,j} + Σ_n r2_n A2_{i,n} + Σ_n c2_n A2_{n,j}
    as an (N, N) matrix via outer-sum factorisation.
    """
    row = (A1 @ r1) + (A2 @ r2)
    col = (A1.T @ c1) + (A2.T @ c2)
    return row.unsqueeze(1) + col.unsqueeze(0)


class DynamicRelationLearning(nn.Module):
    """
    ESTGEL Dynamic Relation Learning module.

    Models spatio-temporal evolution of inter-node relations (INRs) with update (ZE),
    reset (RE), and candidate (HE) gates:

        Ā^{t-1} = (1/Lr) Σ RE_lr ⊙ Â^{t-1}
        Â^t = (1/Lz) Σ (1-ZE_lz) ⊙ Â^{t-1} + (1/Lz) Σ ZE_lz ⊙ (1/Lh) Σ HE_lh

    Learnable gate parameters follow the bilinear neighbourhood form in eq. (9)–(11).
    Parameter vectors are allocated for ``max_nodes`` and sliced per forward pass,
    matching the proposal's fixed-N tensor-masking strategy.
    """

    def __init__(
        self,
        max_nodes: int = 3072,
        Lz: int = 12,
        Lr: int = 12,
        Lh: int = 12,
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.Lz = Lz
        self.Lr = Lr
        self.Lh = Lh

        def _init_gate_params(channels: int) -> nn.Parameter:
            # (channels, 4, max_nodes) for r1, c1, r2, c2
            p = torch.randn(channels, 4, max_nodes) * 0.01
            return nn.Parameter(p)

        self.ZE_params = _init_gate_params(Lz)
        self.RE_params = _init_gate_params(Lr)
        self.HE_params = _init_gate_params(Lh)

    def _channel_gate(
        self,
        params: nn.Parameter,
        A1: torch.Tensor,
        A2: torch.Tensor,
        activation: str,
    ) -> torch.Tensor:
        N = A1.shape[0]
        gates = []
        for ch in range(params.shape[0]):
            r1, c1, r2, c2 = params[ch, 0, :N], params[ch, 1, :N], params[ch, 2, :N], params[ch, 3, :N]
            field = _bilinear_gate_field(A1, A2, r1, c1, r2, c2)
            if activation == "sigmoid":
                gates.append(torch.sigmoid(field))
            elif activation == "tanh":
                gates.append(torch.tanh(field))
            else:
                raise ValueError(f"Unknown activation '{activation}'")
        return torch.stack(gates, dim=0)

    def forward(
        self,
        A_tilde: torch.Tensor,
        A_hat_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            A_tilde: EAM-refined adjacency Ẫ^t, shape (N, N).
            A_hat_prev: Previous DRL output Â^{t-1}, shape (N, N).

        Returns:
            A_hat: Updated relation matrix Â^t, shape (N, N).
            gates: Intermediate gate tensors for explainability.
        """
        if A_tilde.shape != A_hat_prev.shape:
            raise ValueError("A_tilde and A_hat_prev must have the same shape.")
        if A_tilde.shape[0] > self.max_nodes:
            raise ValueError(f"N={A_tilde.shape[0]} exceeds max_nodes={self.max_nodes}.")

        RE = self._channel_gate(self.RE_params, A_hat_prev, A_tilde, "sigmoid")
        A_bar_prev = (RE * A_hat_prev.unsqueeze(0)).mean(dim=0)

        ZE = self._channel_gate(self.ZE_params, A_hat_prev, A_tilde, "sigmoid")
        HE = self._channel_gate(self.HE_params, A_bar_prev, A_tilde, "tanh")

        HE_mean = HE.mean(dim=0)
        retain = ((1.0 - ZE) * A_hat_prev.unsqueeze(0)).mean(dim=0)
        integrate = (ZE * HE_mean.unsqueeze(0)).mean(dim=0)
        A_hat = retain + integrate

        return A_hat, {
            "ZE": ZE,
            "RE": RE,
            "HE": HE,
            "A_bar_prev": A_bar_prev,
        }


class DRLOutput(NamedTuple):
    """Combined EAM + DRL output at one timestep."""

    A_tilde: torch.Tensor
    A_hat: torch.Tensor
    A_hat_prev: torch.Tensor
    eam: EAMOutput
    drl_gates: dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Phase 4: Dynamic Node Learning (DNL) — ESTGEL eq. (12)–(15)
# ---------------------------------------------------------------------------


class DynamicNodeLearning(nn.Module):
    """
    ESTGEL Dynamic Node Learning module.

    Refines node representations using gated graph convolution over Â^t:

        X̂^t = (1-ZN) ⊙ X̂^{t-1} + ZN ⊙ HN
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.W_gcn = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W_zn = nn.Linear(hidden_dim, in_dim)
        self.U_zn = nn.Linear(in_dim, in_dim)
        self.W_hn = nn.Linear(hidden_dim, in_dim)
        self.U_hn = nn.Linear(in_dim, in_dim)
        self.W_rn = nn.Linear(hidden_dim, in_dim)
        self.U_rn = nn.Linear(in_dim, in_dim)

    def _gcn(self, A_hat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        deg = A_hat.sum(dim=1, keepdim=True).clamp(min=1.0)
        A_norm = A_hat / deg
        return torch.relu(A_norm @ self.W_gcn(X))

    def forward(
        self,
        A_hat: torch.Tensor,
        X: torch.Tensor,
        X_hat_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            A_hat: DRL relation matrix (N, N).
            X: Raw node features at time t (N, d).
            X_hat_prev: Previous refined node states (N, d).

        Returns:
            X_hat: Updated node representations (N, d).
            gates: ZN, HN, RN tensors.
        """
        h = self._gcn(A_hat, X)
        ZN = torch.sigmoid(self.W_zn(h) + self.U_zn(X_hat_prev))
        RN = torch.sigmoid(self.W_rn(h) + self.U_rn(X_hat_prev))
        HN = torch.tanh(self.W_hn(h) + self.U_hn(RN * X_hat_prev))
        X_hat = (1.0 - ZN) * X_hat_prev + ZN * HN
        return X_hat, {"ZN": ZN, "RN": RN, "HN": HN, "gcn_hidden": h}


class DNLOutput(NamedTuple):
    """Full ESTGEL timestep output (EAM + DRL + DNL)."""

    A_tilde: torch.Tensor
    A_hat: torch.Tensor
    X_hat: torch.Tensor
    eam: EAMOutput
    drl_gates: dict[str, torch.Tensor]
    dnl_gates: dict[str, torch.Tensor]


class ESTGELTimestepBlock(nn.Module):
    """
    Single-timestep ESTGEL block: EAM → DRL → DNL.

    Aligns with GSoC proposal §1.2 (EAM, DRL, DNL) and the fixed-N masking
    strategy in §1.3.
    """

    def __init__(
        self,
        K: int = 11,
        in_dim: int = 5,
        eam_hidden: int = 32,
        dnl_hidden: int = 64,
        max_nodes: int = 3072,
        Lz: int = 12,
        Lr: int = 12,
        Lh: int = 12,
    ) -> None:
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.eam_block = ESTGELEdgeAttentionBlock(K=K, hidden_dim=eam_hidden)
        self.drl = DynamicRelationLearning(max_nodes=max_nodes, Lz=Lz, Lr=Lr, Lh=Lh)
        self.dnl = DynamicNodeLearning(in_dim=in_dim, hidden_dim=dnl_hidden)

    def _apply_alive_mask(
        self,
        matrix: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Zero rows/cols for inactive cells (growing-node masking)."""
        m = alive_mask.float()
        return matrix * m.unsqueeze(1) * m.unsqueeze(0)

    def forward_timestep(
        self,
        data: Data,
        t_idx: int,
        A_hat_prev: torch.Tensor | None = None,
        X_hat_prev: torch.Tensor | None = None,
        *,
        mode: str = "sparse",
        device: torch.device | None = None,
    ) -> DNLOutput:
        device = device or next(self.parameters()).device
        N = data.x.shape[0]
        d = data.x.shape[1]
        alive_t = data.alive_mask[:, t_idx].to(device)

        eam_out = self.eam_block.forward_timestep(data, t_idx, mode=mode, device=device)
        A_tilde = self._apply_alive_mask(eam_out.adjacency_refined, alive_t)

        if A_hat_prev is None:
            A_hat_prev = torch.zeros(N, N, device=device)
        else:
            A_hat_prev = self._apply_alive_mask(A_hat_prev.to(device), alive_t)

        A_hat, drl_gates = self.drl(A_tilde, A_hat_prev)

        X_t = data.x[:, :, t_idx].to(device)
        X_t = X_t * alive_t.unsqueeze(1).float()
        if X_hat_prev is None:
            X_hat_prev = torch.zeros(N, d, device=device)
        else:
            X_hat_prev = X_hat_prev.to(device) * alive_t.unsqueeze(1).float()

        X_hat, dnl_gates = self.dnl(A_hat, X_t, X_hat_prev)

        return DNLOutput(
            A_tilde=A_tilde,
            A_hat=A_hat,
            X_hat=X_hat,
            eam=eam_out,
            drl_gates=drl_gates,
            dnl_gates=dnl_gates,
        )

    def forward_sequence(
        self,
        data: Data,
        t_start: int = 0,
        t_end: int | None = None,
        timesteps: list[int] | None = None,
        *,
        mode: str = "sparse",
        device: torch.device | None = None,
        detach_state: bool = True,
    ) -> list[DNLOutput]:
        """Run ESTGEL recursively over a time window (temporal DRL/DNL state)."""
        device = device or next(self.parameters()).device
        T = int(data.T)
        if timesteps is None:
            t_end = T if t_end is None else min(t_end, T)
            timesteps = list(range(t_start, t_end))
        else:
            timesteps = [t for t in timesteps if 0 <= t < T]

        outputs: list[DNLOutput] = []
        A_hat_prev: torch.Tensor | None = None
        X_hat_prev: torch.Tensor | None = None

        for t_idx in timesteps:
            if not data.alive_mask[:, t_idx].any():
                continue
            out = self.forward_timestep(
                data,
                t_idx,
                A_hat_prev=A_hat_prev,
                X_hat_prev=X_hat_prev,
                mode=mode,
                device=device,
            )
            outputs.append(out)
            if detach_state:
                A_hat_prev = out.A_hat.detach()
                X_hat_prev = out.X_hat.detach()
            else:
                A_hat_prev = out.A_hat
                X_hat_prev = out.X_hat

        return outputs


def select_timesteps(
    alive_mask: torch.Tensor,
    *,
    max_steps: int = 8,
    stride: int = 15,
) -> list[int]:
    """Pick active, evenly spaced timesteps for tractable training."""
    T = alive_mask.shape[1]
    candidates = [t for t in range(0, T, max(stride, 1)) if alive_mask[:, t].any()]
    if not candidates:
        return [0]
    if len(candidates) <= max_steps:
        return candidates

    step = (len(candidates) - 1) / (max_steps - 1)
    picks = [candidates[int(round(i * step))] for i in range(max_steps)]
    return sorted(set(picks))


class ESTGELClassifier(nn.Module):
    """
    Full ESTGEL model with graph readout and classification head (eq. 12–16).

    Readout: per-timestep mean-pool node features to scalars (N,), concatenate over
    time, then two-layer MLP + logits.
    """

    def __init__(
        self,
        num_classes: int = 2,
        K: int = 11,
        in_dim: int = 5,
        max_timesteps: int = 8,
        time_stride: int = 15,
        window_size: int = 24,
        bptt_truncation: int = 12,
        readout_hidden: int = 128,
        dropout: float = 0.5,
        max_nodes: int = 3072,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_timesteps = max_timesteps
        self.time_stride = time_stride
        self.window_size = window_size
        self.bptt_truncation = bptt_truncation
        self.max_nodes = max_nodes
        self.in_dim = in_dim

        self.block = ESTGELTimestepBlock(
            K=K,
            in_dim=in_dim,
            max_nodes=max_nodes,
            **block_kwargs,
        )

        readout_dim = max_nodes * max_timesteps
        self.readout_mlp = nn.Sequential(
            nn.Linear(readout_dim, readout_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden, num_classes),
        )

    def _graph_readout(
        self,
        outputs: list[DNLOutput],
        timesteps: list[int],
        data: Data,
        device: torch.device,
    ) -> torch.Tensor:
        """Build concat( U_1, ..., U_T ) readout vector with alive-node masking."""
        N = data.x.shape[0]
        chunks: list[torch.Tensor] = []

        for out, t_idx in zip(outputs, timesteps):
            alive = data.alive_mask[:, t_idx].to(device).float()
            # ESTGEL: average-pool feature dim -> one scalar per node.
            u_t = out.X_hat.mean(dim=-1) * alive
            if N < self.max_nodes:
                u_t = F.pad(u_t, (0, self.max_nodes - N))
            elif N > self.max_nodes:
                u_t = u_t[: self.max_nodes]
            chunks.append(u_t)

        if not chunks:
            return torch.zeros(self.max_nodes * self.max_timesteps, device=device)

        while len(chunks) < self.max_timesteps:
            chunks.append(torch.zeros(self.max_nodes, device=device))

        return torch.cat(chunks[: self.max_timesteps], dim=0)

    def forward(
        self,
        data: Data,
        *,
        mode: str = "sparse",
        timesteps: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[DNLOutput]]:
        device = next(self.parameters()).device
        data = data.to(device)

        if timesteps is None:
            timesteps = select_timesteps(
                data.alive_mask,
                max_steps=self.max_timesteps,
                stride=self.time_stride,
            )
        if not timesteps:
            timesteps = [0]

        sample_set = set(timesteps)
        collected: list[DNLOutput] = []
        collected_ts: list[int] = []
        A_hat_prev: torch.Tensor | None = None
        X_hat_prev: torch.Tensor | None = None

        # Sliding window (ESTGEL paper: w=8, g=4); avoids backprop from t=0 every step.
        t_start = min(timesteps)
        t_stop = min(t_start + self.window_size, int(data.T))

        steps_since_detach = 0
        for t_idx in range(t_start, t_stop):
            if not data.alive_mask[:, t_idx].any():
                continue
            out = self.block.forward_timestep(
                data,
                t_idx,
                A_hat_prev=A_hat_prev,
                X_hat_prev=X_hat_prev,
                mode=mode,
                device=device,
            )
            if t_idx in sample_set:
                collected.append(out)
                collected_ts.append(t_idx)

            steps_since_detach += 1
            if self.training and steps_since_detach >= self.bptt_truncation:
                A_hat_prev = out.A_hat.detach()
                X_hat_prev = out.X_hat.detach()
                steps_since_detach = 0
            elif self.training:
                A_hat_prev = out.A_hat
                X_hat_prev = out.X_hat
            else:
                A_hat_prev = out.A_hat.detach()
                X_hat_prev = out.X_hat.detach()

        readout = self._graph_readout(collected, collected_ts, data, device)
        logits = self.readout_mlp(readout)
        return logits, collected
