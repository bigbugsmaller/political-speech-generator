import time

from database import ensure_table_ready, insert_text_into_db
from logger import logger
from serper_api import fetch_additional_results


def search_with_threshold(
    table,
    query,
    threshold=0.85,
    metric="cosine",
    limit=5,
    recursion_depth=0,
    max_recursion=3,
):
    """
    Searches for the query in the LanceDB table and filters results based on a similarity threshold.
    If filtered results are fewer than 2, it fetches additional results using the Serper API.
    """
    overall_started = time.perf_counter()
    try:
        # Use `is None` — empty Lance tables can be falsy via __len__ == 0.
        if table is None:
            raise ValueError("Table cannot be None")
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if threshold <= 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        # table.search() may rehydrate embedders from schema metadata (stale cuda).
        table = ensure_table_ready(table)

        logger.info(
            f"[retrieve] START query={query!r} threshold={threshold} "
            f"limit={limit} recursion={recursion_depth}/{max_recursion}"
        )

        try:
            logger.info("[retrieve] BEFORE vector search")
            search_started = time.perf_counter()
            results_df = table.search(query).metric(metric).limit(limit).to_pandas()
            logger.info(
                f"[retrieve] AFTER vector search rows={len(results_df)} "
                f"elapsed_ms={(time.perf_counter() - search_started) * 1000:.1f}"
            )
        except Exception as e:
            logger.error(f"[retrieve] Database search failed: {e}")
            raise Exception(f"Failed to search database: {str(e)}")

        try:
            if metric == "cosine":
                results_df["similarity_score"] = 1 - results_df["_distance"]
            elif metric == "l2":
                results_df["similarity_score"] = 1 / (1 + results_df["_distance"])
            elif metric == "dot":
                results_df["similarity_score"] = results_df["_distance"]
            elif metric == "ip":
                results_df["similarity_score"] = -results_df["_distance"]
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        except KeyError as e:
            logger.error(f"[retrieve] Missing expected column in results: {e}")
            raise Exception(f"Database results format error: {str(e)}")
        except Exception as e:
            logger.error(f"[retrieve] Error processing similarity scores: {e}")
            raise

        filtered_results = results_df[results_df["similarity_score"] >= threshold]
        logger.info(
            f"[retrieve] Filtered hits={len(filtered_results)} (threshold={threshold})"
        )

        if len(filtered_results) < 2 and recursion_depth < max_recursion:
            logger.info(
                f"[retrieve] BEFORE web fallback "
                f"(recursion {recursion_depth + 1}/{max_recursion})"
            )
            try:
                fallback_started = time.perf_counter()
                additional_results = fetch_additional_results(
                    table, query, min_results=5
                )
                logger.info(
                    f"[retrieve] AFTER web fallback "
                    f"sources={len(additional_results or {})} "
                    f"elapsed_ms={(time.perf_counter() - fallback_started) * 1000:.1f}"
                )
            except Exception as e:
                logger.error(f"[retrieve] Error fetching additional results: {e}")
                raise Exception(f"Failed to fetch additional results: {str(e)}")

            if additional_results:
                logger.info(
                    f"[retrieve] BEFORE insert_text_into_db "
                    f"sources={len(additional_results)}"
                )
                try:
                    insert_started = time.perf_counter()
                    insert_text_into_db(additional_results)
                    logger.info(
                        f"[retrieve] AFTER insert_text_into_db "
                        f"elapsed_ms={(time.perf_counter() - insert_started) * 1000:.1f}"
                    )
                except Exception as e:
                    logger.error(f"[retrieve] Database insertion error: {e}")
                    raise Exception(f"Database insertion error: {str(e)}")

                return search_with_threshold(
                    table,
                    query,
                    threshold=threshold,
                    metric=metric,
                    limit=limit,
                    recursion_depth=recursion_depth + 1,
                    max_recursion=max_recursion,
                )
            logger.warning("[retrieve] No new sources from web fallback")

        if len(filtered_results) == 0:
            logger.info(
                f"[retrieve] DONE no hits "
                f"elapsed_ms={(time.perf_counter() - overall_started) * 1000:.1f}"
            )
            return ""

        try:
            joined = "\n".join(filtered_results["text"])
            logger.info(
                f"[retrieve] DONE hits={len(filtered_results)} chars={len(joined)} "
                f"elapsed_ms={(time.perf_counter() - overall_started) * 1000:.1f}"
            )
            return joined
        except KeyError as e:
            logger.error(f"[retrieve] Missing 'text' column: {e}")
            raise Exception("Results format error: missing 'text' column")

    except Exception as e:
        logger.error(f"[retrieve] Error in search_with_threshold: {e}")
        raise
