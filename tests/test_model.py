import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from canosr.config import Cfg
from canosr.losses import adjacency_loss, node_recon_loss, supcon_loss
from canosr.models import GraphAE
from canosr.scoring import channel_scores, decide

MODEL_CFG = Cfg(
    {
        "encoder": "gatv2",
        "head": "prototype",
        "hidden": 8,
        "heads": 5,
        "layers": 2,
        "dropout": 0.0,
        "projection": 8,
        "readout": "attn",
        "id_embedding": False,
        "id_in_recon": False,
    }
)


def batch(n_graphs: int = 4, n_nodes: int = 6, in_dim: int = 13):
    graphs = []
    generator = torch.Generator().manual_seed(0)
    for index in range(n_graphs):
        x = torch.randn(n_nodes, in_dim, generator=generator)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        graphs.append(
            Data(
                x=x,
                edge_index=edges,
                edge_attr=torch.rand(edges.shape[1], 3, generator=generator),
                y=torch.tensor([float(index % 3)]),
            )
        )
    return next(iter(DataLoader(graphs, batch_size=n_graphs)))


def test_forward_shapes():
    data = batch()
    model = GraphAE(in_dim=13, n_classes=3, cfg=MODEL_CFG)
    out = model(data)
    assert out["h"].shape == (data.num_nodes, 40)
    assert out["z"].shape == (4, 8)
    assert torch.allclose(out["z"].norm(dim=1), torch.ones(4), atol=1e-5)
    assert out["node_hat"].shape == (data.num_nodes, 13)
    assert out["adj_logits"].shape == (4, 6, 6)


def test_losses_are_finite_and_differentiable():
    data = batch()
    model = GraphAE(in_dim=13, n_classes=3, cfg=MODEL_CFG)
    out = model(data)
    sup = supcon_loss(out["z"], data.y.long())
    node = node_recon_loss(out["node_hat"], data.x, skip_first=True)
    adj = adjacency_loss(out["adj_logits"], torch.zeros(4, 6, 6), pos_weight=torch.tensor([2.0]))
    total = sup + node + adj
    assert torch.isfinite(total)
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_channel_scores_and_decision():
    data = batch()
    model = GraphAE(in_dim=13, n_classes=3, cfg=MODEL_CFG)
    model.head.set_prototypes(torch.randn(3, 8), torch.tensor([0, 1, 2]))
    channels = channel_scores(model, data)
    for key in ("proto", "node", "adj", "msp", "energy"):
        assert channels[key].shape == (4,)
        assert torch.isfinite(channels[key]).all()
    pred = decide(channels["proto"], threshold=1e9, logits=model(data)["logits"], n_known=3)
    assert pred.max().item() <= 3
