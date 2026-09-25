import json
import time

import requests

from config import SERPER_API_HOST, SERPER_API_KEY
from logger import logger
from scraper import extract_p_tags

REQUEST_TIMEOUT_SECONDS = 10
SERPER_MAX_ATTEMPTS = 3


def serper_search(query, num_results=1, pages=1):
    """Fetches search results from Serper API via requests (with timeout)."""
    started = time.perf_counter()
    url = f"https://{SERPER_API_HOST}/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "gl": "in",
        "num": num_results,
        "page": pages,
        "type": "search",
    }

    try:
        logger.info(
            f"[serper] BEFORE POST timeout={REQUEST_TIMEOUT_SECONDS}s "
            f"query={query!r} url={url}"
        )
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        logger.info(
            f"[serper] AFTER POST status={response.status_code} "
            f"elapsed_ms={(time.perf_counter() - started) * 1000:.1f} "
            f"bytes={len(response.content)}"
        )

        if response.status_code != 200:
            logger.error(
                f"[serper] Non-200 status={response.status_code} body={response.text[:300]!r}"
            )
            return {}

        parsed_data = response.json()
        organic = parsed_data.get("organic", [])
        logger.info(f"[serper] Parsed organic_results={len(organic)}")
        return parsed_data
    except requests.exceptions.Timeout:
        logger.error(
            f"[serper] TIMEOUT after {REQUEST_TIMEOUT_SECONDS}s query={query!r}"
        )
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"[serper] Request error: {e}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"[serper] JSON parse error: {e}")
        return {}
    except Exception as e:
        logger.error(f"[serper] Unexpected error: {e}", exc_info=True)
        return {}


def fetch_additional_results(table, query, min_results=4):
    """Fetches additional results using Serper API if needed."""
    overall_started = time.perf_counter()
    try:
        logger.info(
            f"[web_fetch] START query={query!r} min_results={min_results} "
            f"max_attempts={SERPER_MAX_ATTEMPTS}"
        )
        data = {}

        for attempt in range(1, SERPER_MAX_ATTEMPTS + 1):
            logger.info(
                f"[web_fetch] Serper attempt {attempt}/{SERPER_MAX_ATTEMPTS}"
            )
            data = serper_search(query, num_results=min_results)

            if data.get("organic"):
                logger.info("[web_fetch] Got organic results")
                break
            logger.warning(
                f"[web_fetch] No organic results on attempt {attempt}, retrying..."
            )
            time.sleep(1)

        result_count = len(data.get("organic", []))
        logger.info(f"[web_fetch] organic_count={result_count}")

        if result_count == 0:
            logger.warning("[web_fetch] No results after all attempts")
            return {}

        results = {}

        for idx, item in enumerate(data.get("organic", []), start=1):
            try:
                link = item.get("link", "")
                if not link:
                    logger.warning(f"[web_fetch] Skipping result #{idx} with no link")
                    continue

                source_id = link
                # Skip expensive vector existence checks; duplicate inserts are acceptable.
                logger.info(f"[web_fetch] BEFORE scrape #{idx} url={link}")
                scrape_started = time.perf_counter()
                extracted_text = extract_p_tags(link)
                logger.info(
                    f"[web_fetch] AFTER scrape #{idx} chars={len(extracted_text or '')} "
                    f"elapsed_ms={(time.perf_counter() - scrape_started) * 1000:.1f}"
                )
                if extracted_text:
                    results[link] = extracted_text
                else:
                    logger.warning(f"[web_fetch] No text extracted from {link}")
            except Exception as e:
                logger.error(f"[web_fetch] Error processing result #{idx}: {e}")
                continue

        logger.info(
            f"[web_fetch] DONE scraped_urls={len(results)} "
            f"elapsed_ms={(time.perf_counter() - overall_started) * 1000:.1f}"
        )
        return results
    except Exception as e:
        logger.error(f"[web_fetch] Unexpected error: {e}", exc_info=True)
        return {}
