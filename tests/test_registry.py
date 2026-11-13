import pytest

from canosr.registry import KINDS, build, names, register


def test_expected_entries_are_registered():
    import canosr.models  # noqa: F401
    import canosr.scoring  # noqa: F401

    assert {"gatv2", "gat", "sage", "gcn", "mlp"}.issubset(set(names("encoder")))
    assert {"prototype", "linear", "none"}.issubset(set(names("head")))
    assert {"fused", "msp", "energy", "recon", "proto_distance"}.issubset(set(names("scorer")))


def test_unknown_entry_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        build("encoder", "does-not-exist")


def test_duplicate_registration_is_rejected():
    @register("loss", "temp-loss")
    def _factory():
        return None

    with pytest.raises(ValueError):
        register("loss", "temp-loss")(lambda: None)


def test_unknown_kind_is_rejected():
    with pytest.raises(KeyError):
        register("banana", "x")
