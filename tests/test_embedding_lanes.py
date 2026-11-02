import os

import pytest

from src.embedding_lanes import (
    LANE_CUSTOM,
    LANE_FASTEMBED,
    build_embedding_lanes,
)
from tests.helpers.embedding_lanes import (
    FakeChroma,
    FakeEmbedder,
    FailingEmbedder,
    patch_chroma,
)


def test_build_embedding_lanes_keeps_custom_and_fastembed_dimensions_separate(monkeypatch):
    fake = FakeChroma()
    patch_chroma(monkeypatch, fake)

    import src.embedding_lanes as lanes

    monkeypatch.setattr(
        lanes,
        "_build_custom_client",
        lambda: FakeEmbedder(768, "nomic-embed-text", "http://embeddings/v1"),
    )
    monkeypatch.setattr(
        lanes,
        "_build_fastembed_client",
        lambda: FakeEmbedder(384, "sentence-transformers/all-MiniLM-L6-v2", "local://fastembed"),
    )

    built = build_embedding_lanes("odysseus_memories")

    assert [lane.name for lane in built] == [LANE_CUSTOM, LANE_FASTEMBED]
    assert built[0].collection_name == "odysseus_memories_custom"
    assert built[0].dimension == 768
    assert built[1].collection_name == "odysseus_memories_fastembed"
    assert built[1].dimension == 384

    built[0].collection.add(ids=["custom"], embeddings=built[0].encode(["a"]), documents=["a"])
    built[1].collection.add(ids=["fast"], embeddings=built[1].encode(["a"]), documents=["a"])

    with pytest.raises(RuntimeError, match="dimension"):
        built[0].collection.query(query_embeddings=built[1].encode(["bad"]), n_results=1)


def test_build_embedding_lanes_recreates_only_custom_when_fingerprint_changes(monkeypatch):
    fake = FakeChroma()
    old_custom = fake.get_or_create_collection(
        "odysseus_rag_custom",
        metadata={
            "embedding_lane": "custom",
            "embedding_dimension": 768,
            "embedding_fingerprint": "old",
        },
    )
    old_custom.add(ids=["old"], embeddings=[[0.0] * 768], documents=["old"])
    fast = fake.get_or_create_collection(
        "odysseus_rag_fastembed",
        metadata={
            "embedding_lane": "fastembed",
            "embedding_dimension": 384,
        },
    )
    fast.add(ids=["fast"], embeddings=[[0.0] * 384], documents=["fast"])
    patch_chroma(monkeypatch, fake)

    import src.embedding_lanes as lanes

    monkeypatch.setattr(lanes, "_build_custom_client", lambda: FakeEmbedder(1024, "bge-large", "http://embeddings/v1"))
    monkeypatch.setattr(lanes, "_build_fastembed_client", lambda: FakeEmbedder(384, "sentence-transformers/all-MiniLM-L6-v2", "local://fastembed"))

    built = build_embedding_lanes("odysseus_rag")

    assert "odysseus_rag_custom" in fake.deleted
    assert fake.collections["odysseus_rag_custom"].count() == 1
    assert len(fake.collections["odysseus_rag_custom"].rows["old"]["embedding"]) == 1024
    assert fake.collections["odysseus_rag_fastembed"].count() == 1
    assert built[0].dimension == 1024


def test_lane_reset_reembeds_existing_documents_on_fingerprint_change(monkeypatch):
    fake = FakeChroma()
    old_custom = fake.get_or_create_collection(
        "odysseus_memories_custom",
        metadata={
            "embedding_lane": "custom",
            "embedding_dimension": 384,
            "embedding_fingerprint": "old",
        },
    )
    old_custom.add(
        ids=["existing-memory"],
        embeddings=[[0.0] * 384],
        documents=["existing custom memory"],
        metadatas=[{"source": "memory"}],
    )
    patch_chroma(monkeypatch, fake)

    import src.embedding_lanes as lanes

    monkeypatch.setattr(lanes, "_build_custom_client", lambda: FakeEmbedder(768, "nomic", "http://embeddings/v1"))

    def fail_fastembed():
        raise RuntimeError("fastembed missing")

    monkeypatch.setattr(lanes, "_build_fastembed_client", fail_fastembed)

    built = build_embedding_lanes("odysseus_memories")

    assert [lane.name for lane in built] == [LANE_CUSTOM]
    assert "odysseus_memories_custom" in fake.deleted
    rebuilt = fake.collections["odysseus_memories_custom"]
    assert rebuilt.count() == 1
    assert rebuilt.get()["ids"] == ["existing-memory"]
    assert len(rebuilt.rows["existing-memory"]["embedding"]) == 768


def test_lane_reset_keeps_existing_collection_when_reembed_fails(monkeypatch):
    fake = FakeChroma()
    old_custom = fake.get_or_create_collection(
        "odysseus_memories_custom",
        metadata={
            "embedding_lane": "custom",
            "embedding_dimension": 384,
            "embedding_fingerprint": "old",
        },
    )
    old_custom.add(
        ids=["existing-memory"],
        embeddings=[[0.0] * 384],
        documents=["existing custom memory"],
        metadatas=[{"source": "memory"}],
    )
    patch_chroma(monkeypatch, fake)

    import src.embedding_lanes as lanes

    monkeypatch.setattr(lanes, "_build_custom_client", lambda: FailingEmbedder(768, "nomic", "http://embeddings/v1"))
    monkeypatch.setattr(lanes, "_build_fastembed_client", lambda: FakeEmbedder(384, "mini", "local://fastembed"))

    built = build_embedding_lanes("odysseus_memories")

    assert [lane.name for lane in built] == [LANE_FASTEMBED]
    assert "odysseus_memories_custom" not in fake.deleted
    assert fake.collections["odysseus_memories_custom"].count() == 1
    assert len(fake.collections["odysseus_memories_custom"].rows["existing-memory"]["embedding"]) == 384


def test_lane_reset_keeps_existing_collection_when_preserve_read_fails(monkeypatch):
    fake = FakeChroma()
    old_custom = fake.get_or_create_collection(
        "odysseus_memories_custom",
        metadata={
            "embedding_lane": "custom",
            "embedding_dimension": 384,
            "embedding_fingerprint": "old",
        },
    )
    old_custom.add(
        ids=["existing-memory"],
        embeddings=[[0.0] * 384],
        documents=["existing custom memory"],
        metadatas=[{"source": "memory"}],
    )

    def fail_get(*_args, **_kwargs):
        raise RuntimeError("chroma read failed")

    old_custom.get = fail_get
    patch_chroma(monkeypatch, fake)

    import src.embedding_lanes as lanes

    monkeypatch.setattr(lanes, "_build_custom_client", lambda: FakeEmbedder(768, "nomic", "http://embeddings/v1"))

    def fail_fastembed():
        raise RuntimeError("fastembed missing")

    monkeypatch.setattr(lanes, "_build_fastembed_client", fail_fastembed)

    built = build_embedding_lanes("odysseus_memories")

    assert built == []
    assert "odysseus_memories_custom" not in fake.deleted
    assert "odysseus_memories_custom" in fake.collections


def test_lane_reset_restores_existing_collection_when_rewrite_fails(monkeypatch):
    fake = FakeChroma()
    old_custom = fake.get_or_create_collection(
        "odysseus_memories_custom",
        metadata={
            "embedding_lane": "custom",
            "embedding_dimension": 384,
            "embedding_fingerprint": "old",
        },
    )
    old_custom.add(
        ids=["existing-memory"],
        embeddings=[[0.0] * 384],
        documents=["existing custom memory"],
        metadatas=[{"source": "memory"}],
    )
    fake.fail_next_add_for["odysseus_memories_custom"] = 1
    patch_chroma(monkeypatch, fake)

    import src.embedding_lanes as lanes

    monkeypatch.setattr(lanes, "_build_custom_client", lambda: FakeEmbedder(768, "nomic", "http://embeddings/v1"))

    def fail_fastembed():
        raise RuntimeError("fastembed missing")

    monkeypatch.setattr(lanes, "_build_fastembed_client", fail_fastembed)

    built = build_embedding_lanes("odysseus_memories")

    assert built == []
    restored = fake.collections["odysseus_memories_custom"]
    assert restored.count() == 1
    assert restored.get()["ids"] == ["existing-memory"]
    assert len(restored.rows["existing-memory"]["embedding"]) == 384


def test_build_embedding_lanes_uses_fastembed_when_custom_unavailable(monkeypatch):
    fake = FakeChroma()
    patch_chroma(monkeypatch, fake)

    import src.embedding_lanes as lanes

    def fail_custom():
        raise RuntimeError("down")

    monkeypatch.setattr(lanes, "_build_custom_client", fail_custom)
    monkeypatch.setattr(lanes, "_build_fastembed_client", lambda: FakeEmbedder(384, "mini", "local://fastembed"))

    built = build_embedding_lanes("odysseus_tool_index")

    assert [lane.name for lane in built] == [LANE_FASTEMBED]
    assert built[0].collection_name == "odysseus_tool_index_fastembed"


def test_custom_lane_preserves_default_embedding_client_probe(monkeypatch):
    import src.embedding_lanes as lanes
    import src.embeddings as embeddings

    embeddings.reset_http_embed_state()
    monkeypatch.setattr(lanes, "_load_custom_endpoint", lambda: {})

    calls = []

    class DefaultClient(FakeEmbedder):
        def __init__(self, url=None, model=None, api_key=None):
            calls.append({"url": url, "model": model, "api_key": api_key})
            super().__init__(768, model or "all-minilm:l6-v2", url or "http://localhost:11434/v1/embeddings")

    monkeypatch.setattr(embeddings, "EmbeddingClient", DefaultClient)

    client = lanes._build_custom_client()

    assert calls == [{"url": None, "model": None, "api_key": None}]
    assert client.url == "http://localhost:11434/v1/embeddings"
    embeddings.reset_http_embed_state()


def test_custom_lane_uses_http_down_latch(monkeypatch):
    import src.embedding_lanes as lanes
    import src.embeddings as embeddings

    embeddings.reset_http_embed_state()
    calls = []

    class DownClient:
        def __init__(self, url=None, model=None, api_key=None):
            calls.append({"url": url, "model": model, "api_key": api_key})

        def get_sentence_embedding_dimension(self):
            raise RuntimeError("endpoint down")

    class LocalFastEmbed(FakeEmbedder):
        def __init__(self):
            super().__init__(384, "mini", "local://fastembed")

    monkeypatch.setattr(embeddings, "EmbeddingClient", DownClient)
    monkeypatch.setattr(embeddings, "FastEmbedClient", LocalFastEmbed)

    with pytest.raises(RuntimeError, match="HTTP embedding lane unavailable"):
        lanes._build_custom_client()
    with pytest.raises(RuntimeError, match="HTTP embedding lane unavailable"):
        lanes._build_custom_client()

    assert calls == [{"url": None, "model": None, "api_key": None}]
    embeddings.reset_http_embed_state()


def test_fastembed_clients_are_shared_across_lanes(monkeypatch):
    """ONNX weights are heap copies (~1.2GB each for bge-large), so loading one
    model once per consumer is what pushed the server to ~4.4GB resident. Every
    lane must reuse a single client."""
    import src.embedding_lanes as lanes
    import src.embeddings as embeddings

    fake = FakeChroma()
    patch_chroma(monkeypatch, fake)
    embeddings.reset_http_embed_state()

    constructed = []

    class OneModelFastEmbed:
        def __init__(self, model=None):
            constructed.append(model)
            self.model = model or "sentence-transformers/all-MiniLM-L6-v2"
            self.url = "local://fastembed"

        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, normalize_embeddings=True, query=False):
            return [[float(i + 1)] * 384 for i, _ in enumerate(texts)]

    monkeypatch.setattr(embeddings, "FastEmbedClient", OneModelFastEmbed)
    monkeypatch.setattr(embeddings, "_DEFAULT_FASTEMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    def fake_http_client():
        raise RuntimeError("HTTP embedding lane unavailable")

    monkeypatch.setattr(lanes, "_build_custom_client", fake_http_client)

    memories = build_embedding_lanes("odysseus_memories")
    docs = build_embedding_lanes("odysseus_rag")

    assert len(constructed) == 1, "ONNX model was loaded once per consumer"
    assert memories[0].client is docs[0].client
    assert embeddings.get_fastembed_client() is memories[0].client
    embeddings.reset_http_embed_state()


def test_reset_reloads_fastembed_after_model_switch(monkeypatch):
    """Each distinct model keeps its own cached client, and resetting the cache
    lets a model/config change pick up new weights without a restart."""
    import src.embeddings as embeddings

    loaded = []

    class ModelAwareFastEmbed:
        def __init__(self, model=None):
            loaded.append(model)
            self.model = model or "sentence-transformers/all-MiniLM-L6-v2"
            self.url = "local://fastembed"

        def get_sentence_embedding_dimension(self):
            return 384 if "MiniLM" in self.model else 1024

    monkeypatch.setattr(embeddings, "FastEmbedClient", ModelAwareFastEmbed)

    small = embeddings.get_fastembed_client("sentence-transformers/all-MiniLM-L6-v2")
    assert embeddings.get_fastembed_client("sentence-transformers/all-MiniLM-L6-v2") is small
    large = embeddings.get_fastembed_client("BAAI/bge-large-en-v1.5")

    assert large is not small
    assert large.get_sentence_embedding_dimension() == 1024

    embeddings.reset_fastembed_client_cache()
    reloaded = embeddings.get_fastembed_client("BAAI/bge-large-en-v1.5")

    assert reloaded is not large
    assert loaded == [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ]
    embeddings.reset_fastembed_client_cache()


def test_thread_setting_replaces_the_cached_client(monkeypatch):
    """FASTEMBED_THREADS sizes the ONNX thread pool when the client is built, so a
    client cached under one setting must not be served to a process reconfigured
    to another."""
    import src.embeddings as embeddings

    built = []

    class ThreadsAwareFastEmbed:
        def __init__(self, model=None):
            raw = os.environ.get("FASTEMBED_THREADS", "").strip()
            self.threads = int(raw) if raw else None
            built.append(self.threads)
            self.model = model or embeddings._DEFAULT_FASTEMBED_MODEL
            self.url = "local://fastembed"

        def get_sentence_embedding_dimension(self):
            return 384

    monkeypatch.setattr(embeddings, "FastEmbedClient", ThreadsAwareFastEmbed)
    monkeypatch.delenv("FASTEMBED_CLIENT_CACHE_SIZE", raising=False)

    monkeypatch.setenv("FASTEMBED_THREADS", "4")
    four = embeddings.get_fastembed_client()
    monkeypatch.setenv("FASTEMBED_THREADS", "16")
    sixteen = embeddings.get_fastembed_client()

    assert sixteen is not four, "a threads change reused the client's old thread pool"
    assert built == [4, 16]
    assert embeddings.get_fastembed_client() is sixteen  # cached per setting


def test_cache_keeps_only_the_newest_client_by_default(monkeypatch):
    """Every resident model is ~1.2GB of private heap, so a model swap must not
    stack: by default the cache holds the newest client only."""
    import src.embeddings as embeddings

    loaded = []

    class TwoModelFastEmbed:
        def __init__(self, model=None):
            self.model = model or embeddings._DEFAULT_FASTEMBED_MODEL
            self.url = "local://fastembed"
            loaded.append(self.model)

        def get_sentence_embedding_dimension(self):
            return 384 if "MiniLM" in self.model else 1024

    monkeypatch.setattr(embeddings, "FastEmbedClient", TwoModelFastEmbed)
    monkeypatch.delenv("FASTEMBED_CLIENT_CACHE_SIZE", raising=False)

    small = embeddings.get_fastembed_client("sentence-transformers/all-MiniLM-L6-v2")
    large = embeddings.get_fastembed_client("BAAI/bge-large-en-v1.5")

    assert small is not large
    assert embeddings.fastembed_client_cache_size() == 1, "the previous model stayed resident"
    assert embeddings.get_fastembed_client("BAAI/bge-large-en-v1.5") is large
    assert loaded == ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5"]


def test_cache_size_setting_keeps_two_models_resident(monkeypatch):
    """FASTEMBED_CLIENT_CACHE_SIZE=2 is the escape hatch for a setup that flips
    between two models and wants both weight copies kept."""
    import src.embeddings as embeddings

    loaded = []

    class TwoModelFastEmbed:
        def __init__(self, model=None):
            self.model = model or embeddings._DEFAULT_FASTEMBED_MODEL
            self.url = "local://fastembed"
            loaded.append(self.model)

        def get_sentence_embedding_dimension(self):
            return 384 if "MiniLM" in self.model else 1024

    monkeypatch.setattr(embeddings, "FastEmbedClient", TwoModelFastEmbed)
    monkeypatch.setenv("FASTEMBED_CLIENT_CACHE_SIZE", "2")

    small = embeddings.get_fastembed_client("sentence-transformers/all-MiniLM-L6-v2")
    large = embeddings.get_fastembed_client("BAAI/bge-large-en-v1.5")

    assert embeddings.fastembed_client_cache_size() == 2
    assert embeddings.get_fastembed_client("sentence-transformers/all-MiniLM-L6-v2") is small
    assert embeddings.get_fastembed_client("BAAI/bge-large-en-v1.5") is large
    assert loaded == ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5"]


def test_two_threads_load_the_model_once(monkeypatch):
    """The RAG, memory and tool-index lanes are built seconds apart at boot and can
    race the very first load; the lock must yield one weight copy, not two."""
    import threading
    import time

    import src.embeddings as embeddings

    start = threading.Barrier(2)
    loaded = []

    class SlowFastEmbed:
        def __init__(self, model=None):
            time.sleep(0.05)  # widen the window the two threads can race through
            self.model = model or embeddings._DEFAULT_FASTEMBED_MODEL
            self.url = "local://fastembed"
            loaded.append(self.model)

        def get_sentence_embedding_dimension(self):
            return 384

    monkeypatch.setattr(embeddings, "FastEmbedClient", SlowFastEmbed)
    clients = []

    def worker():
        start.wait()
        clients.append(embeddings.get_fastembed_client())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(loaded) == 1, f"the model was loaded {len(loaded)} times"
    assert clients[0] is clients[1]
