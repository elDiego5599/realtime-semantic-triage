import sys
import types

import numpy as np
import pytest

from semtriage.embed import BgeM3Embedder, Embedder, HashingEmbedder, l2_normalize


def test_l2_normalize_unit_rows_and_keeps_zero_rows():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, -2.0]])
    result = l2_normalize(matrix)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 0.0], [0.0, -1.0]], atol=1e-6)
    assert not np.isnan(result).any()


def test_l2_normalize_does_not_modify_input():
    matrix = np.array([[3.0, 4.0]])
    l2_normalize(matrix)
    np.testing.assert_array_equal(matrix, [[3.0, 4.0]])


def test_hashing_embedder_shape_dtype_and_name():
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.embed(["buffer overflow in parser", "sql injection"])
    assert embedder.dim == 64
    assert embedder.name == "hashing-64"
    assert vectors.shape == (2, 64)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-6)


def test_l2_normalize_float32_input_is_not_modified_in_place():
    matrix = np.array([[3.0, 4.0]], dtype=np.float32)
    result = l2_normalize(matrix)
    assert result is not matrix
    np.testing.assert_array_equal(matrix, [[3.0, 4.0]])


def test_l2_normalize_large_values_do_not_overflow_to_zero():
    result = l2_normalize(np.array([[1e30, 1e30]]))
    np.testing.assert_allclose(result, [[2**-0.5, 2**-0.5]], atol=1e-6)


def test_l2_normalize_rejects_non_2d_input():
    with pytest.raises(ValueError):
        l2_normalize(np.array([3.0, 4.0]))


def test_hashing_embedder_rejects_bare_string():
    with pytest.raises(TypeError):
        HashingEmbedder(dim=8).embed("not a list")


def test_hashing_embedder_default_dim():
    assert HashingEmbedder().dim == 256


def test_hashing_embedder_is_deterministic_across_instances():
    text = "Remote code execution via crafted YAML"
    first = HashingEmbedder(dim=128).embed([text])
    second = HashingEmbedder(dim=128).embed([text])
    np.testing.assert_array_equal(first, second)


def test_hashing_embedder_known_buckets():
    # Pinned so the hash cannot silently change between runs or Python processes.
    import hashlib

    dim = 32
    tokens = ["xss", "in", "login"]
    expected = np.zeros(dim)
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        expected[int.from_bytes(digest, "little") % dim] += 1.0
    expected /= np.linalg.norm(expected)
    result = HashingEmbedder(dim=dim).embed(["XSS, in login!"])[0]
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_hashing_embedder_case_and_punctuation_insensitive():
    embedder = HashingEmbedder(dim=64)
    a, b = embedder.embed(["Path Traversal!", "path traversal"])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_hashing_embedder_similar_texts_score_higher():
    embedder = HashingEmbedder(dim=512)
    base, close, far = embedder.embed(
        [
            "heap buffer overflow in the png decoder",
            "buffer overflow in png decoder heap",
            "cross site scripting in admin dashboard",
        ]
    )
    assert float(base @ close) > float(base @ far)


def test_hashing_embedder_empty_and_punctuation_text_is_zero_row():
    vectors = HashingEmbedder(dim=16).embed(["", "!!! ---"])
    assert vectors.shape == (2, 16)
    np.testing.assert_array_equal(vectors, np.zeros((2, 16), dtype=np.float32))


def test_hashing_embedder_empty_batch():
    vectors = HashingEmbedder(dim=16).embed([])
    assert vectors.shape == (0, 16)
    assert vectors.dtype == np.float32


@pytest.mark.parametrize("dim", [0, -1])
def test_hashing_embedder_rejects_non_positive_dim(dim):
    with pytest.raises(ValueError):
        HashingEmbedder(dim=dim)


def test_hashing_embedder_satisfies_protocol():
    embedder: Embedder = HashingEmbedder(dim=8)
    assert isinstance(embedder.name, str)
    assert isinstance(embedder.dim, int)


class _FakeBGEM3FlagModel:
    instances: list["_FakeBGEM3FlagModel"] = []

    def __init__(self, model_name, use_fp16=True, **kwargs):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.kwargs = kwargs
        self.calls = []
        _FakeBGEM3FlagModel.instances.append(self)

    def encode(self, sentences, **kwargs):
        self.calls.append((sentences, kwargs))
        rows = [[float(len(text)), 0.0] + [0.0] * 1022 for text in sentences]
        for row in rows:
            row[1] = 1.0
        return {"dense_vecs": np.array(rows, dtype=np.float64)}


@pytest.fixture
def fake_flag_embedding(monkeypatch):
    _FakeBGEM3FlagModel.instances = []
    module = types.ModuleType("FlagEmbedding")
    module.BGEM3FlagModel = _FakeBGEM3FlagModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", module)
    return _FakeBGEM3FlagModel


def test_bge_m3_defaults_and_model_construction(fake_flag_embedding):
    embedder = BgeM3Embedder()
    assert embedder.dim == 1024
    assert embedder.name == "BAAI/bge-m3"
    (model,) = fake_flag_embedding.instances
    assert model.model_name == "BAAI/bge-m3"
    assert model.use_fp16 is False


def test_bge_m3_custom_model_and_fp16(fake_flag_embedding):
    embedder = BgeM3Embedder("local/bge-m3", use_fp16=True)
    assert embedder.name == "local/bge-m3"
    (model,) = fake_flag_embedding.instances
    assert model.model_name == "local/bge-m3"
    assert model.use_fp16 is True


def test_bge_m3_encode_arguments_and_normalized_output(fake_flag_embedding):
    embedder = BgeM3Embedder(batch_size=8, max_length=256)
    vectors = embedder.embed(("abc", "abcd"))
    (model,) = fake_flag_embedding.instances
    (sentences, kwargs) = model.calls[0]
    assert sentences == ["abc", "abcd"]
    assert kwargs["batch_size"] == 8
    assert kwargs["max_length"] == 256
    assert kwargs["return_dense"] is True
    assert kwargs["return_sparse"] is False
    assert kwargs["return_colbert_vecs"] is False
    assert vectors.shape == (2, 1024)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(vectors[0, :2], np.array([3.0, 1.0]) / np.sqrt(10.0), atol=1e-6)


def test_bge_m3_default_batch_and_length(fake_flag_embedding):
    BgeM3Embedder().embed(["x"])
    (model,) = fake_flag_embedding.instances
    kwargs = model.calls[0][1]
    assert kwargs["batch_size"] == 32
    assert kwargs["max_length"] == 512


def test_bge_m3_empty_batch_does_not_call_model(fake_flag_embedding):
    vectors = BgeM3Embedder().embed([])
    assert vectors.shape == (0, 1024)
    assert vectors.dtype == np.float32
    assert fake_flag_embedding.instances[0].calls == []


def test_bge_m3_rejects_bare_string(fake_flag_embedding):
    with pytest.raises(TypeError):
        BgeM3Embedder().embed("not a list")


def test_bge_m3_rejects_unexpected_output_shape(fake_flag_embedding, monkeypatch):
    def short_encode(self, sentences, **kwargs):
        return {"dense_vecs": np.ones((len(sentences), 768))}

    monkeypatch.setattr(fake_flag_embedding, "encode", short_encode)
    with pytest.raises(ValueError, match="1024"):
        BgeM3Embedder().embed(["x"])


def test_bge_m3_missing_package_raises_helpful_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    with pytest.raises(ImportError, match=r"uv sync --group embed") as excinfo:
        BgeM3Embedder()
    assert excinfo.value.__cause__ is not None


def test_importing_embed_module_does_not_import_flag_embedding():
    import importlib

    import semtriage.embed

    sys.modules.pop("FlagEmbedding", None)
    importlib.reload(semtriage.embed)
    assert "FlagEmbedding" not in sys.modules
