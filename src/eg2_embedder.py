"""
eg2_embedder.py

A local client for google/embeddinggemma-2 — the embedder the Strata store runs.

fastembed's catalog (41 models in 0.9.1) does not carry embeddinggemma-2, so this
module drives the ONNX graph directly with onnxruntime + tokenizers. The recipe is
the one serve/embedder.py uses on the Strata side — same prefixes, same output
tensor, same padding, same L2 normalization — so a passage embedded here lands in
the same 768-d space as the store's rows and the two applications' vectors are
directly comparable.

Configure with EMBEDDINGGEMMA_MODEL_DIR, the directory holding
onnx/model_quantized.onnx, tokenizer.json and (optionally) SHA256SUMS. Unset names
no model and raises; a SHA256SUMS entry that does not match the file on disk
raises, because a silently swapped model silently changes every ranking.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "embeddinggemma-2"
MODEL_DIR_ENV = "EMBEDDINGGEMMA_MODEL_DIR"


class Eg2Embedder:
    """embeddinggemma-2, text-only, community q8 ONNX (the repo's own precision
    table puts worst-case cosine to fp32 at 0.9997). 768-d native, stored at full
    dimension so an MRL slice later needs no re-embed.

    Duck-types the slice of fastembed's TextEmbedding that src/embeddings.py uses:
    embed() for documents, query_embed() for queries, and the dimension probe."""

    DIMS = 768
    FILE = "onnx/model_quantized.onnx"
    MAX_SEQ = 8192
    OUT = "sentence_embedding"          # the graph pools; no numpy mean here
    QUERY_PREFIX = "task: search result | query: "
    DOC_PREFIX = "title: none | text: "
    PAD_TOKEN, PAD_ID = "<pad>", 0
    BATCH = 64                           # bounds the arena on big backfills

    def __init__(self, model_dir: str | Path | None = None, threads: int | Optional[int] = None):
        import onnxruntime as ort          # lazy: opt-in model, not a base import
        from tokenizers import Tokenizer   # ditto

        raw = str(model_dir or os.getenv(MODEL_DIR_ENV, "")).strip()
        if not raw:
            raise ValueError(
                f"{MODEL_NAME} needs a model directory: set {MODEL_DIR_ENV} to the folder "
                f"holding {self.FILE} and tokenizer.json"
            )
        d = Path(raw).expanduser()
        if not d.is_dir():
            raise ValueError(f"{MODEL_DIR_ENV}={d} is not a directory")
        self._verify_pinned(d)

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads)) if threads else 0
        options.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(d / self.FILE), options, providers=["CPUExecutionProvider"]
        )
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(max_length=self.MAX_SEQ)
        pad_id = self._tok.token_to_id(self.PAD_TOKEN)
        self._tok.enable_padding(pad_id=self.PAD_ID if pad_id is None else pad_id,
                                 pad_token=self.PAD_TOKEN)

        inputs = {i.name for i in self._sess.get_inputs()}
        self._input_names = inputs
        outputs = {o.name for o in self._sess.get_outputs()}
        if self.OUT not in outputs:
            raise ValueError(
                f"{MODEL_NAME} graph has no output {self.OUT!r} (has: {sorted(outputs)}) "
                "— refusing to guess a pooling path"
            )
        self.model = MODEL_NAME
        self.model_name = MODEL_NAME       # the store's stamp name
        self.url = f"local://{MODEL_NAME}"
        self.dims = self.DIMS
        self.threads = threads
        self.max_seq = self.MAX_SEQ
        self.model_dir = d
        logger.info("embeddinggemma-2 loaded from %s (dims=%d, threads=%s)",
                    d, self.dims, threads if threads else "default")

    @staticmethod
    def _verify_pinned(model_dir: Path) -> None:
        """SHA256SUMS is the model's identity, not a nicety: write the file next to
        the weights and a silent re-download cannot change the vector space."""
        pins = model_dir / "SHA256SUMS"
        if not pins.is_file():
            return
        for line in pins.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            digest, _, rel = line.partition("  ")
            rel = (rel or "").strip()
            target = model_dir / rel
            if not target.is_file():
                raise ValueError(f"{MODEL_NAME} pin names a missing file: {target}")
            hasher = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(block)
            got = hasher.hexdigest()
            if got != digest.strip():
                raise ValueError(
                    f"{MODEL_NAME} model changed: {rel} is {got}, SHA256SUMS says {digest.strip()}"
                )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DIMS), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), self.BATCH):
            encoded = self._tok.encode_batch(list(texts[start : start + self.BATCH]))
            ids = np.fromiter(
                (t for e in encoded for t in e.ids), np.int64,
                count=sum(len(e.ids) for e in encoded),
            ).reshape(len(encoded), -1)
            mask = np.fromiter(
                (t for e in encoded for t in e.attention_mask), np.int64, count=ids.size
            ).reshape(ids.shape)
            feed: dict = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            for feature in ("image_features", "video_features", "audio_features"):
                # The multimodal export demands these inputs for pure text too; an
                # empty tensor is the honest "no media here".
                if feature in self._input_names:
                    feed[feature] = np.zeros((0, 512), np.float32)
            out = self._sess.run([self.OUT], feed)[0]
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            batches.append((out / np.clip(norms, 1e-9, None)).astype(np.float32))
        return np.vstack(batches)

    def embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        """Documents, under the model's document recipe."""
        return iter(self._encode([self.DOC_PREFIX + t for t in texts]))

    def query_embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        """Queries, under the retrieval recipe the model was trained with."""
        return iter(self._encode([self.QUERY_PREFIX + t for t in texts]))

    def passage_embed(self, texts: Iterable[str], max_length: Optional[int] = None) -> Iterator[np.ndarray]:
        return self.embed(texts)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dims
