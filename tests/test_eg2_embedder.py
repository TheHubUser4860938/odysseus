"""The embeddinggemma-2 client: fastembed's catalog does not carry
google/embeddinggemma-2, so Odysseus drives the graph itself and lands its
vectors in the same 768-d space as the Strata store.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from src.eg2_embedder import Eg2Embedder


class _Feat:
    """One tokenized string, the way the `tokenizers` batch result looks."""

    def __init__(self, text):
        self.tokens = text.split()
        self.ids = [1] + [7] * len(self.tokens) + [2]
        self.attention_mask = [1] * len(self.ids)


class _Tok:
    def __init__(self):
        self.seen = []
        self.truncated_to = None
        self.padding = None

    def enable_truncation(self, max_length):
        self.truncated_to = max_length

    def enable_padding(self, pad_id, pad_token):
        self.padding = (pad_id, pad_token)

    def token_to_id(self, token):
        return 0 if token == "<pad>" else 42

    def encode_batch(self, texts):
        self.seen = list(texts)
        return [_Feat(t) for t in texts]


class _IO:
    def __init__(self, name):
        self.name = name


class _Sess:
    """Deterministic stand-in for the ONNX graph: row i of the batch is the
    vector (i+1) * [1, 2, 3, ...], so normalization is checkable by hand."""

    def __init__(self, path, options, providers=None):
        self.path = path
        self.options = options
        self.providers = providers
        self.feeds = []

    def get_inputs(self):
        return [_IO("input_ids"), _IO("attention_mask")]

    def get_outputs(self):
        return [_IO("sentence_embedding")]

    def run(self, output_names, feed):
        assert output_names == ["sentence_embedding"]
        self.feeds.append(feed)
        n = feed["input_ids"].shape[0]
        base = np.arange(1, 769, dtype=np.float32)
        return [np.stack([base * (i + 1) for i in range(n)])]


def _prepare(tmp_path, monkeypatch, outputs=None):
    """Stage a model directory and swap in the ONNX/tokenizer doubles."""
    (tmp_path / "onnx").mkdir(parents=True, exist_ok=True)
    (tmp_path / "onnx" / "model_quantized.onnx").write_bytes(b"graph")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    sess = _Sess(None, None)
    if outputs is not None:
        sess.get_outputs = lambda: [_IO(name) for name in outputs]
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda p, o, providers=None: sess)
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda p: _Tok())
    return sess


def _embedder(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    return Eg2Embedder(tmp_path, threads=2), None


def test_model_identity_is_the_store_stamp(tmp_path, monkeypatch):
    client, _ = _embedder(tmp_path, monkeypatch)

    assert client.model == "embeddinggemma-2"
    assert client.url == "local://embeddinggemma-2"
    assert client.dims == 768
    assert client.get_sentence_embedding_dimension() == 768


def test_documents_and_queries_use_the_models_own_recipes(tmp_path, monkeypatch):
    """embeddinggemma-2 is a prefix-conditioned model: the query recipe and the
    document recipe are what make its two sides comparable."""
    _prepare(tmp_path, monkeypatch)
    client = Eg2Embedder(tmp_path, threads=2)

    list(client.embed(["a passage"]))
    assert client._tok.seen == ["title: none | text: a passage"]

    list(client.query_embed(["a question"]))
    assert client._tok.seen == ["task: search result | query: a question"]


def test_vectors_are_l2_normalized(tmp_path, monkeypatch):
    client, _ = _embedder(tmp_path, monkeypatch)

    vecs = list(client.embed(["one passage", "two passages"]))

    for v in vecs:
        assert pytest.approx(float(np.linalg.norm(v)), abs=1e-5) == 1.0
    # Row 2 was built 2x row 1, so after normalizing both must point the same way.
    assert pytest.approx(float(np.dot(vecs[0], vecs[1])), abs=1e-5) == 1.0


def test_a_graph_without_the_pinned_output_refuses_to_guess(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch, outputs=["last_hidden_state"])

    with pytest.raises(ValueError, match="sentence_embedding"):
        Eg2Embedder(tmp_path, threads=1)


def test_the_sha256_pin_is_enforced(tmp_path, monkeypatch):
    sess = _prepare(tmp_path, monkeypatch)
    (tmp_path / "SHA256SUMS").write_text(
        "0" * 64 + "  onnx/model_quantized.onnx\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="model_quantized.onnx"):
        Eg2Embedder(tmp_path, threads=1)


def test_the_model_directory_is_named_not_guessed(monkeypatch, tmp_path):
    monkeypatch.delenv("EMBEDDINGGEMMA_MODEL_DIR", raising=False)

    with pytest.raises(ValueError, match="EMBEDDINGGEMMA_MODEL_DIR"):
        Eg2Embedder(threads=1)

    monkeypatch.setenv("EMBEDDINGGEMMA_MODEL_DIR", str(tmp_path / "nope"))
    with pytest.raises(ValueError, match="nope"):
        Eg2Embedder(threads=1)


def test_fastembed_client_routes_the_gemma_name_to_the_local_graph(tmp_path, monkeypatch):
    """FASTEMBED_MODEL=embeddinggemma-2 must not reach for the fastembed registry,
    which does not list it."""
    import src.embeddings as embeddings

    built = {}

    class Gemme:
        model = "embeddinggemma-2"
        dims = 768

        def __init__(self, model_dir=None, threads=None):
            built["threads"] = threads

        def embed(self, texts):
            return iter([np.ones(768, dtype=np.float32) for _ in texts])

        def query_embed(self, texts):
            return iter([np.zeros(768, dtype=np.float32) for _ in texts])

    monkeypatch.setattr(embeddings, "Eg2Embedder", Gemme)
    import fastembed

    monkeypatch.setattr(
        fastembed, "TextEmbedding", lambda **kw: pytest.fail("the fastembed registry was used")
    )

    client = embeddings.FastEmbedClient("embeddinggemma-2")

    assert client.model == "embeddinggemma-2"
    assert client.get_sentence_embedding_dimension() == 768
    assert built["threads"] is None


def test_queries_use_the_query_recipe_documents_the_plain_one(tmp_path, monkeypatch):
    """One recipe for both sides is what makes a hybrid search quietly worse: the
    store's rows are embedded under the document prefix, the query under the
    query prefix."""
    import src.embeddings as embeddings

    calls = []

    class Recording:
        model = "BAAI/bge-large-en-v1.5"

        def embed(self, texts):
            calls.append(("embed", list(texts)))
            return iter([np.ones(1024, dtype=np.float32) for _ in texts])

        def query_embed(self, texts):
            calls.append(("query_embed", list(texts)))
            return iter([np.ones(1024, dtype=np.float32) for _ in texts])

    client = embeddings.FastEmbedClient.__new__(embeddings.FastEmbedClient)
    client.model = Recording.model
    client._embedding = Recording()
    client._dim = None
    client.url = "local://fastembed"
    client.threads = None

    client.encode(["passage a", "passage b"])
    client.encode(["what is federalism"], query=True)

    assert calls == [
        ("embed", ["passage a", "passage b"]),
        ("query_embed", ["what is federalism"]),
    ]


def test_lanes_ask_for_the_query_recipe_when_searching(monkeypatch):
    """query_lanes embeds the query, not a document; the lane must say so, and the
    document path must stay on the plain recipe."""
    import src.embedding_lanes as lanes
    import src.embeddings as embeddings
    from tests.helpers.embedding_lanes import FakeChroma, patch_chroma

    fake = FakeChroma()
    patch_chroma(monkeypatch, fake)
    seen = []

    class Recorder:
        model = "recorder-model"
        url = "local://fastembed"

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, normalize_embeddings=True, query=False):
            seen.append(query)
            return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(embeddings, "FastEmbedClient", Recorder)

    def no_http_lane():
        raise RuntimeError("HTTP embedding lane unavailable")

    monkeypatch.setattr(lanes, "_build_custom_client", no_http_lane)

    built = lanes.build_embedding_lanes("odysseus_rag")
    lane = built[0]
    lane.collection.add(ids=["a"], embeddings=[[1.0] * 4], documents=["a doc"], metadatas=[{}])
    seen.clear()

    lanes.query_lanes([lane], "a query", n_results=lambda l: 3, include=["documents"])

    assert seen == [True]


EG2_DIR = os.getenv("EMBEDDINGGEMMA_MODEL_DIR", "/home/eugene/Strata-data/embeddings/embeddinggemma-2")


@pytest.mark.skipif(
    not (Path(EG2_DIR) / "onnx" / "model_quantized.onnx").is_file(),
    reason="the store's embeddinggemma-2 weights are not on this box",
)
def test_the_real_weights_embed_in_the_stores_space():
    """The doubles prove the recipe; this proves the graph. Same weights, same
    prefixes, same output tensor as the Strata store runs."""
    emb = Eg2Embedder(EG2_DIR, threads=4)
    doc = next(emb.embed(["The federal power of enlarging the sphere of the society"]))
    query = next(emb.query_embed(["what is federalism"]))

    assert emb.dims == 768
    assert doc.shape == (768,)
    assert query.shape == (768,)
    assert pytest.approx(float(np.linalg.norm(doc)), abs=1e-3) == 1.0
    assert 0.0 < float(np.dot(doc, query)) < 1.0
