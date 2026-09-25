import os

import lancedb
import torch
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DB_PATH
from logger import logger

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Avoid accidental CUDA probes in downstream libs when we already know we want CPU.
# (Only set when CUDA is unavailable so real GPU machines are unaffected.)
if not torch.cuda.is_available():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def get_embedding_device() -> str:
    """Pick embedding device once, up front. Never fall through to a CUDA retry loop."""
    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    logger.info(
        f"Embedding device selection: torch.cuda.is_available()={cuda_available} "
        f"-> device={device}"
    )
    return device


def _force_table_embedding_device(tbl, runtime_device: str) -> None:
    """
    Why this exists (and must be reapplied, not just once at startup):

    LanceDB persists the embedding function config (including device='cuda') in
    table schema metadata. On every open_table() — and again when table.add() /
    table.search() rehydrate the embedding function for auto-embed — LanceDB
    rebuilds that function from the *persisted* metadata, independent of any
    device we set when constructing our process-local `model`. If metadata still
    says cuda on a CPU-only torch build, LanceDB's retry_with_exponential_backoff
    can hang for minutes on 'Torch not compiled with CUDA enabled'.

    Therefore: after every open_table()/create_table(), and immediately before
    every table.search() / table.add(), force the restored embedding function(s)
    onto get_embedding_device() and disable retries for permanent device failures.
    """
    funcs = tbl.embedding_functions or {}
    for vector_col, conf in funcs.items():
        fn = conf.function
        old_device = getattr(fn, "device", None)
        if old_device != runtime_device:
            logger.warning(
                f"Overriding stale embedding device on column={vector_col!r}: "
                f"{old_device!r} -> {runtime_device!r} (from table metadata)"
            )
            try:
                fn.device = runtime_device
            except Exception as e:
                logger.error(f"Failed to set embedding device={runtime_device}: {e}")
                raise

        # max_retries=0 disables LanceDB's exponential backoff for embed failures.
        if runtime_device == "cpu" and getattr(fn, "max_retries", None) != 0:
            fn.max_retries = 0

        # Drop any cached SentenceTransformer built for the wrong device.
        getter = getattr(fn, "get_embedding_model", None)
        if getter is not None:
            cache_clear = getattr(getter, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()


# Module-level handles filled during init below.
db = None
device = None
model = None
splitter = None
table = None
table_name = "words"


def ensure_table_ready(tbl=None):
    """
    Single shared gate for every table use site (open, search, add).

    Re-applies the runtime embedding device (from get_embedding_device() /
    module `device`) so LanceDB cannot use stale schema-metadata device='cuda'.
    Call this right after open_table()/create_table() and immediately before
    table.search() / table.add() — not only once at module import.
    """
    target = table if tbl is None else tbl
    if target is None:
        raise ValueError("Table cannot be None")
    if device is None:
        raise RuntimeError("Embedding device not initialized yet")
    _force_table_embedding_device(target, device)
    return target


def ping_db() -> bool:
    """Return True if LanceDB is reachable."""
    db.table_names()
    return True


# Connect to LanceDB
try:
    logger.info("Connecting to LanceDB...")
    db = lancedb.connect(DB_PATH)
    logger.info("Connected to LanceDB successfully")
except Exception as e:
    error_msg = f"Failed to connect to LanceDB: {str(e)}"
    logger.error(error_msg)
    raise Exception(error_msg)

# Set up embedding model
try:
    device = get_embedding_device()
    logger.info(f"Setting up embedding model on device={device}...")
    # max_retries=0: do not sleep/retry on permanent CUDA/device errors.
    model = (
        get_registry()
        .get("sentence-transformers")
        .create(name=EMBEDDING_MODEL_NAME, device=device, max_retries=0)
    )
    logger.info(
        f"Embedding model initialized successfully "
        f"(name={EMBEDDING_MODEL_NAME}, device={model.device}, max_retries={model.max_retries})"
    )
except Exception as e:
    error_msg = f"Failed to initialize embedding model: {str(e)}"
    logger.error(error_msg)
    raise Exception(error_msg)

# Define text splitter
try:
    logger.info("Initializing text splitter...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    logger.info("Text splitter initialized successfully")
except Exception as e:
    error_msg = f"Failed to initialize text splitter: {str(e)}"
    logger.error(error_msg)
    raise Exception(error_msg)


# Define LanceDB Schema
class Words(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()
    source_id: str


# Create or open table
try:
    if table_name not in db.table_names():
        logger.info(f"Table '{table_name}' does not exist. Creating new table...")
        table = db.create_table(table_name, schema=Words)
        logger.info("Table created successfully")
    else:
        logger.info(f"Opening existing table '{table_name}'...")
        table = db.open_table(table_name)
        logger.info("Table opened successfully")

    # open_table()/create_table() rebuild embedders from persisted metadata —
    # reapply runtime device immediately (not only once "at startup" in spirit:
    # every open path must do this).
    ensure_table_ready(table)
    for col, conf in (table.embedding_functions or {}).items():
        logger.info(
            f"Active embedding for {col}: device={conf.function.device}, "
            f"max_retries={conf.function.max_retries}"
        )
except Exception as e:
    error_msg = f"Failed to create or open table '{table_name}': {str(e)}"
    logger.error(error_msg)
    raise Exception(error_msg)


def insert_text_into_db(text_dict):
    """
    Takes a dictionary of source_id -> text mappings,
    splits the text into chunks and inserts each chunk into the vector database.

    Embeddings are computed with the process-local model on `device`. We also
    call ensure_table_ready() before every table.add() so LanceDB cannot rebuild
    a stale cuda embedder from schema metadata during add.
    """
    total_chunks = 0
    logger.info("[insert] START text insertion")

    try:
        if not isinstance(text_dict, dict):
            error_msg = "Input must be a dictionary mapping source_id to text"
            logger.error(error_msg)
            raise TypeError(error_msg)

        ready = ensure_table_ready(table)

        for source_id, text in text_dict.items():
            try:
                logger.info(f"[insert] Processing source_id={source_id}")
                if not isinstance(text, str):
                    error_msg = f"Text for source_id {source_id} must be a string"
                    logger.error(error_msg)
                    raise TypeError(error_msg)

                chunks = splitter.split_text(text)
                if not chunks:
                    logger.warning(f"[insert] No chunks for source_id={source_id}")
                    continue

                logger.info(
                    f"[insert] BEFORE CPU embed chunks={len(chunks)} "
                    f"device={model.device}"
                )
                vectors = model.compute_source_embeddings(chunks)
                logger.info(f"[insert] AFTER CPU embed vectors={len(vectors)}")

                records = [
                    {
                        "text": chunk,
                        "source_id": source_id,
                        "vector": vector,
                    }
                    for chunk, vector in zip(chunks, vectors)
                ]
                # Re-force immediately before add: add() can rehydrate embedders
                # from schema metadata independently of our process-local model.
                ensure_table_ready(ready)
                logger.info(f"[insert] BEFORE table.add rows={len(records)}")
                ready.add(records)
                logger.info(f"[insert] AFTER table.add rows={len(records)}")
                total_chunks += len(records)
                logger.info(
                    f"[insert] Inserted {len(records)} chunks for source_id={source_id}"
                )
            except Exception as e:
                error_msg = f"Failed to process text for source_id {source_id}: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)

        logger.info(
            f"[insert] DONE total_chunks={total_chunks} sources={len(text_dict)}"
        )
    except Exception as e:
        if not isinstance(e, TypeError):
            error_msg = f"Failed during text insertion process: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        raise
