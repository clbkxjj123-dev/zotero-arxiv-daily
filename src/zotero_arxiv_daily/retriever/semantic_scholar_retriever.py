from datetime import date, timedelta
from time import sleep
from typing import Any

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper

S2_BULK_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "paperId,title,abstract,authors,url,venue,publicationDate,externalIds"
# The free API key allows 1 request per second across all endpoints
S2_REQUEST_INTERVAL = 1.1


@register_retriever("semantic_scholar")
class SemanticScholarRetriever(BaseRetriever):
    """Keyword-based sweep over Semantic Scholar's bulk search API.

    Complements arXiv/Crossref with conference papers and venues that are
    not covered by the tracked journals.
    """

    def __init__(self, config):
        super().__init__(config)
        if not self.retriever_config.queries:
            raise ValueError("source.semantic_scholar.queries must be a list of search queries")

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        days_back = self.retriever_config.get("days_back", 3)
        since = (date.today() - timedelta(days=days_back)).isoformat()
        headers = {}
        if api_key := self.retriever_config.get("api_key"):
            headers["x-api-key"] = api_key
        items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for query in self.retriever_config.queries:
            params = {
                "query": query,
                "publicationDateOrYear": f"{since}:",
                "fields": S2_FIELDS,
            }
            try:
                response = requests.get(S2_BULK_SEARCH_API, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                for item in response.json().get("data", []):
                    paper_id = item.get("paperId")
                    if paper_id and paper_id not in seen_ids:
                        seen_ids.add(paper_id)
                        items.append(item)
            except Exception as e:
                # A failing query must not kill the whole daily run
                logger.error(f"Semantic Scholar query failed ({query}): {e}")
            sleep(S2_REQUEST_INTERVAL)
        if self.config.executor.debug:
            items = items[:10]
        return items

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        title = raw_paper.get("title")
        if not title:
            return None
        # Fall back to the title so abstract-less papers can still be scored
        abstract = raw_paper.get("abstract") or title
        authors = [a.get("name") for a in raw_paper.get("authors", []) if a.get("name")]
        url = raw_paper.get("url") or ""
        if venue := raw_paper.get("venue"):
            title = f"[{venue}] {title}"
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
        )
