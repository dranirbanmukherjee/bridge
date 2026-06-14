"""Smoke tests for the bridge package.

Fast sanity checks that the package imports cleanly and its core objects
construct and round-trip on tiny inputs. These do not train models, generate
embeddings, or make any external API / network calls.
"""
import bridge


def test_version_and_author():
    assert bridge.__version__ == "0.1.0"
    assert "Mukherjee" in bridge.__author__


def test_public_api_importable():
    # Every symbol advertised in __all__ resolves on the package.
    for name in bridge.__all__:
        assert hasattr(bridge, name), f"missing public symbol: {name}"
    # Spot-check the main entry points import directly.
    from bridge import (  # noqa: F401
        AttributeEncoder,
        BRIDGEConfig,
        BRIDGEModel,
        BRIDGEPipeline,
    )


def test_config_defaults_match_paper():
    cfg = bridge.BRIDGEConfig()
    # Fixed contrastive temperature/weight (tau = lambda = 0.1) per the paper.
    assert cfg.contrastive_temp == 0.1
    assert cfg.contrastive_weight == 0.1
    # OpenAI 3072-dim backend is the default.
    assert cfg.embedding_backend == "openai"
    assert cfg.embedding_dim == 3072


def test_config_presets():
    assert bridge.BRIDGEConfig.quick().epochs_final == 50
    assert bridge.BRIDGEConfig.standard().epochs_final == 200
    assert bridge.BRIDGEConfig.thorough().epochs_final == 1000
    gemma = bridge.BRIDGEConfig.for_gemma()
    assert gemma.embedding_backend == "gemma"
    assert gemma.embedding_dim == 768


def test_config_json_roundtrip(tmp_path):
    cfg = bridge.BRIDGEConfig(projection_units=64, mask_size=512)
    path = tmp_path / "cfg.json"
    cfg.save(str(path))
    loaded = bridge.BRIDGEConfig.load(str(path))
    assert loaded.to_dict() == cfg.to_dict()
    assert loaded.projection_units == 64


def test_pipeline_constructs(tmp_path):
    pipe = bridge.BRIDGEPipeline(
        attributes=["region", "varietal"],
        output_dir=str(tmp_path / "out"),
        verbose=False,
    )
    assert pipe.attributes == ["region", "varietal"]
    assert pipe.is_fitted is False
    assert pipe.config.contrastive_temp == 0.1
    assert (tmp_path / "out").is_dir()
