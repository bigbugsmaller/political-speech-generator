import time

import requests
from bs4 import BeautifulSoup

from logger import logger

REQUEST_TIMEOUT_SECONDS = 10


def extract_p_tags(url):
    """Fetch and extract all <p> text from a given URL."""
    try:
        logger.info(f"[scrape] START url={url}")
        started = time.perf_counter()

        if not url or not isinstance(url, str):
            logger.error(f"[scrape] Invalid URL provided: {url}")
            return ""

        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            logger.info(
                f"[scrape] BEFORE requests.get timeout={REQUEST_TIMEOUT_SECONDS}s url={url}"
            )
            page = requests.get(
                url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
            )
            logger.info(
                f"[scrape] AFTER requests.get status={page.status_code} "
                f"elapsed_ms={(time.perf_counter() - started) * 1000:.1f} url={url}"
            )

            if page.status_code != 200:
                logger.warning(
                    f"[scrape] Non-200 status ({page.status_code}) from {url}"
                )
        except requests.exceptions.Timeout:
            logger.error(
                f"[scrape] TIMEOUT after {REQUEST_TIMEOUT_SECONDS}s url={url}"
            )
            return ""
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[scrape] Connection error url={url}: {e}")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"[scrape] Request failed url={url}: {e}")
            return ""

        try:
            soup = BeautifulSoup(page.text, "html.parser")
            p_tags = soup.find_all("p")
            paragraphs = " ".join(p.get_text() for p in p_tags)
            logger.info(
                f"[scrape] DONE chars={len(paragraphs)} p_tags={len(p_tags)} "
                f"elapsed_ms={(time.perf_counter() - started) * 1000:.1f} url={url}"
            )
            return paragraphs
        except Exception as e:
            logger.error(f"[scrape] HTML parsing error for {url}: {e}")
            return page.text if hasattr(page, "text") else ""

    except Exception as e:
        logger.error(f"[scrape] Unexpected error for {url}: {e}", exc_info=True)
        return ""
