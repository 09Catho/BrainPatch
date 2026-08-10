"""Text-corpus ingestion for activation extraction.

Turns a Hugging Face dataset into a deterministic stream of fixed-length token
blocks. Determinism matters more than it might seem: two extraction runs with
the same seed must produce byte-identical shards, otherwise "resume" is not
resume, and an SAE cannot be attributed to a specific corpus.

Chunking strategy
-----------------
Documents are tokenized individually and split into non-overlapping blocks of
exactly ``sequence_length`` tokens. Trailing remainders shorter than
``min_block_tokens`` are discarded rather than padded. The result is that every
stored activation corresponds to a real token in real context -- there is no
padding to mask out, and no boundary artifacts from concatenating unrelated
documents inside one block.

Licensing note
--------------
The default corpus (``Salesforce/wikitext``) is CC BY-SA 3.0, derived from
Wikipedia. Derived numerical artifacts (SAE weights, activation statistics) are
publishable; verbatim text is redistributed only as short attributed snippets
inside feature contexts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator

#: Small, permissively-licensed default. Good enough to exercise the pipeline;
#: not claimed to be the right corpus for behaviour-relevant features.
DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"
DEFAULT_SPLIT = "train"


@dataclass
class TokenBlock:
    """One fixed-length block of tokens with provenance back to its document."""

    example_index: int
    input_ids: list[int]
    text: str
    source_doc: int
    char_offset: int


@dataclass
class CorpusConfig:
    """How to turn a dataset into token blocks."""

    dataset: str = DEFAULT_DATASET
    config: str | None = DEFAULT_CONFIG
    split: str = DEFAULT_SPLIT
    text_column: str = "text"
    sequence_length: int = 256
    min_block_tokens: int = 64
    min_doc_chars: int = 200
    seed: int = 0
    #: Documents drawn from the head of the split before shuffling. Bounding
    #: this keeps a smoke run from streaming the whole dataset.
    max_documents: int = 20_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "text_column": self.text_column,
            "sequence_length": self.sequence_length,
            "min_block_tokens": self.min_block_tokens,
            "min_doc_chars": self.min_doc_chars,
            "seed": self.seed,
            "max_documents": self.max_documents,
        }

    @property
    def dataset_id(self) -> str:
        """Human-readable identifier recorded in the manifest."""
        return f"{self.dataset}:{self.config}" if self.config else self.dataset


def load_documents(cfg: CorpusConfig) -> list[str]:
    """Load and deterministically shuffle raw documents.

    Short documents are dropped first, so the shuffle operates on the set that
    will actually be used and the seed therefore selects the same documents
    regardless of how many are ultimately consumed.
    """
    from datasets import load_dataset

    dataset = load_dataset(cfg.dataset, cfg.config, split=cfg.split)

    docs: list[str] = []
    for i, row in enumerate(dataset):
        if i >= cfg.max_documents:
            break
        text = row.get(cfg.text_column)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < cfg.min_doc_chars:
            continue
        docs.append(text)

    random.Random(cfg.seed).shuffle(docs)
    return docs


def iter_token_blocks(
    tokenizer: Any,
    cfg: CorpusConfig,
    *,
    documents: list[str] | None = None,
    start_example: int = 0,
) -> Iterator[TokenBlock]:
    """Yield fixed-length token blocks, deterministically.

    Parameters
    ----------
    start_example:
        Skip this many blocks before yielding. Used on resume so a restarted
        run reproduces exactly the block sequence it would have produced had it
        never stopped.
    """
    docs = documents if documents is not None else load_documents(cfg)
    example_index = 0

    for doc_index, text in enumerate(docs):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n_blocks = len(ids) // cfg.sequence_length
        remainder = len(ids) - n_blocks * cfg.sequence_length

        spans: list[tuple[int, int]] = [
            (b * cfg.sequence_length, (b + 1) * cfg.sequence_length) for b in range(n_blocks)
        ]
        if remainder >= cfg.min_block_tokens:
            spans.append((n_blocks * cfg.sequence_length, len(ids)))

        for start, end in spans:
            block_ids = ids[start:end]
            if len(block_ids) < cfg.min_block_tokens:
                continue
            if example_index >= start_example:
                yield TokenBlock(
                    example_index=example_index,
                    input_ids=block_ids,
                    text=tokenizer.decode(block_ids),
                    source_doc=doc_index,
                    char_offset=start,
                )
            example_index += 1


def batched(iterator: Iterator[TokenBlock], batch_size: int) -> Iterator[list[TokenBlock]]:
    """Group blocks into batches, yielding a short final batch if needed."""
    batch: list[TokenBlock] = []
    for block in iterator:
        batch.append(block)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
