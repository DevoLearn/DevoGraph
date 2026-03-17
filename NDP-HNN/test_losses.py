"""
Tests for NDP-HNN/losses.py — incidence_bce fix.

The bug: the old implementation constructed logits as a constant tensor from
ground-truth edge_index (no requires_grad), then computed BCE against
(logits > 0).float() — targets derived from the same constant.  Result:
loss ≈ 0.0 always, no gradient reached any model parameter.

These tests verify the three invariants that the fix must satisfy:
  1. loss.requires_grad is True  (gradient flows to model parameters)
  2. loss value is non-trivial    (not always ~0)
  3. backward() populates .grad  (optimizer actually updates weights)
  4. empty edge_index returns 0  (edge-case guard unchanged)
  5. LSTM state (h, c) tuple is handled correctly in train.py extraction
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pytest
import torch
from torch_geometric.data import Data
from losses import incidence_bce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(num_nodes=5, num_hyperedges=3, seed=0):
    """Synthetic PyG Data with a random incidence structure."""
    torch.manual_seed(seed)
    # Build a simple incidence: node i belongs to hyperedge i % num_hyperedges
    rows = torch.arange(num_nodes)
    cols = rows % num_hyperedges
    edge_index = torch.stack([rows, cols], dim=0)
    return Data(edge_index=edge_index, num_nodes=num_nodes)


def make_node_emb(num_nodes=5, hid_dim=16, requires_grad=True, seed=0):
    torch.manual_seed(seed)
    emb = torch.randn(num_nodes, hid_dim)
    emb.requires_grad_(requires_grad)
    return emb


# ---------------------------------------------------------------------------
# Test 1: loss must have requires_grad=True
# ---------------------------------------------------------------------------

def test_loss_has_grad():
    """
    The returned loss tensor must carry a grad_fn so that loss.backward()
    propagates gradients to model parameters.

    The old implementation returned a constant tensor (no grad_fn) because
    logits was constructed without any connection to learned parameters.
    """
    data = make_data()
    node_emb = make_node_emb(num_nodes=data.num_nodes)

    loss = incidence_bce(node_emb, data)

    assert loss.requires_grad, (
        "incidence_bce returned a tensor with requires_grad=False.\n"
        "The loss has no gradient path to model parameters — "
        "the reconstruction objective is a no-op."
    )


# ---------------------------------------------------------------------------
# Test 2: loss value is non-trivial (not always ~0)
# ---------------------------------------------------------------------------

def test_loss_value_is_nontrivial():
    """
    With random node embeddings the loss must be meaningfully above zero.

    The old implementation always returned BCE(±10, {0,1}) ≈ 0.0 because
    logits were hard-coded constants.  A correct implementation will return
    a value that depends on the embeddings and changes as they change.
    """
    data = make_data()
    node_emb = make_node_emb(num_nodes=data.num_nodes)

    loss = incidence_bce(node_emb, data)

    assert loss.item() > 1e-3, (
        f"incidence_bce returned {loss.item():.6f}, which is effectively zero.\n"
        "This indicates the loss is still a constant independent of node_emb."
    )


# ---------------------------------------------------------------------------
# Test 3: backward() populates gradients on node_emb
# ---------------------------------------------------------------------------

def test_backward_populates_grad():
    """
    After loss.backward(), node_emb.grad must be a non-zero tensor.

    This is the decisive test: it verifies that the optimizer will actually
    receive gradient signal from the reconstruction objective.
    """
    data = make_data()
    node_emb = make_node_emb(num_nodes=data.num_nodes)

    loss = incidence_bce(node_emb, data)
    loss.backward()

    assert node_emb.grad is not None, (
        "node_emb.grad is None after backward() — no gradient flowed from "
        "incidence_bce to the node embeddings."
    )
    assert node_emb.grad.abs().sum().item() > 0.0, (
        "node_emb.grad is all zeros — the loss is constant with respect to "
        "node_emb and provides no learning signal."
    )


# ---------------------------------------------------------------------------
# Test 4: loss changes when node_emb changes
# ---------------------------------------------------------------------------

def test_loss_depends_on_node_emb():
    """
    Two different node_emb tensors must yield different loss values.

    The old bug produced an identical constant loss regardless of embeddings.
    """
    data = make_data()
    emb_a = make_node_emb(num_nodes=data.num_nodes, seed=0)
    emb_b = make_node_emb(num_nodes=data.num_nodes, seed=99)

    loss_a = incidence_bce(emb_a, data).item()
    loss_b = incidence_bce(emb_b, data).item()

    assert abs(loss_a - loss_b) > 1e-6, (
        f"incidence_bce returned the same value ({loss_a:.6f}) for two "
        f"different node_emb tensors — loss is independent of embeddings."
    )


# ---------------------------------------------------------------------------
# Test 5: empty edge_index returns scalar 0.0 with no crash
# ---------------------------------------------------------------------------

def test_empty_edge_index_returns_zero():
    """
    When there are no hyperedges the function must return 0.0 without error.
    This is required for snapshots at early timesteps with no edges.
    """
    data = Data(edge_index=torch.zeros(2, 0, dtype=torch.long), num_nodes=4)
    node_emb = make_node_emb(num_nodes=4)

    loss = incidence_bce(node_emb, data)

    assert loss.item() == 0.0, (
        f"Expected 0.0 for empty edge_index, got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# Test 6: train.py state extraction — LSTM tuple is handled correctly
# ---------------------------------------------------------------------------

def test_lstm_state_extraction():
    """
    In train.py, state is (h, c) for LSTM and a plain tensor for GRU/RNN.
    The extraction  h = state[0] if isinstance(state, tuple) else state
    must yield a tensor that incidence_bce can use.

    This test replicates that extraction logic to confirm the fix is
    compatible with all three RNN variants.
    """
    data = make_data()
    N, D = data.num_nodes, 16

    # GRU / RNN: state is a plain tensor
    state_gru = torch.randn(N, D, requires_grad=True)
    h_gru = state_gru[0] if isinstance(state_gru, tuple) else state_gru
    loss_gru = incidence_bce(h_gru, data)
    assert loss_gru.requires_grad, "GRU state extraction broke gradient flow."

    # LSTM: state is (h, c)
    h_tensor = torch.randn(N, D, requires_grad=True)
    c_tensor = torch.randn(N, D)
    state_lstm = (h_tensor, c_tensor)
    h_lstm = state_lstm[0] if isinstance(state_lstm, tuple) else state_lstm
    loss_lstm = incidence_bce(h_lstm, data)
    assert loss_lstm.requires_grad, "LSTM state extraction broke gradient flow."
