import logging
from duckduckgo_search import DDGS

def search_images(params):
    """
    Enhanced image search:
    - Appends 'diagram' for scientific queries (anatomy, botany, zoology, biology)
    - Falls back to plain search if no results
    - Returns richer metadata (url, title, source)
    """
    query = params.get("query", "").lower().strip()
    logging.info(f"search_images called with query={query}")

    # Add 'diagram' only for science/anatomy/botany/zoology keywords
    science_terms = [
    "anatomy", "photosynthesis", "biology", "botany", "zoology",
    "heart", "brain", "leaf", "flower", "skeleton", "kidney",
    "liver", "eye", "root", "stem", "plant", "animal",
    "reproductive system", "digestive system"
]

    if any(term in query for term in science_terms) and "diagram" not in query:
        query = query + " diagram"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=5))

            # Fallback: retry without 'diagram' if no results
            if not results and "diagram" in query:
                query = query.replace("diagram", "").strip()
                results = list(ddgs.images(query, max_results=5))

            if results:
                logging.info(f"DuckDuckGo returned {len(results)} images for query={query}")
                return {
                    "images": [
                        {
                            "url": r.get("image"),
                            "title": r.get("title"),
                            "source": r.get("source"),
                            "thumbnail": r.get("thumbnail")
                        }
                        for r in results
                    ]
                }
    except Exception as e:
        logging.error(f"DuckDuckGo image search error: {e}")

    return {"images": []}


"""
import logging
from duckduckgo_search import DDGS

def search_images(params):

    query = params.get("query", "").lower().strip()
    logging.info(f"search_images called with query={query}")

    # Always force diagram context
    if "diagram" not in query:
        query = query + " diagram"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=5))
            if results:
                logging.info(f"DuckDuckGo returned {len(results)} images for query={query}")
                return {"images": [{"url": r["image"]} for r in results]}
    except Exception as e:
        logging.error(f"DuckDuckGo image search error: {e}")

    return {"images": []}
"""