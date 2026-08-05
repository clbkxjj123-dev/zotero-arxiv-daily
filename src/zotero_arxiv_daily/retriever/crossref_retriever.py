from datetime import date, timedelta
from time import sleep
from typing import Any

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from .html_text import strip_html
from ..protocol import Paper

CROSSREF_API = "https://api.crossref.org/works"


@register_retriever("crossref")
class CrossrefRetriever(BaseRetriever):
    """Track newly indexed journal articles by ISSN via the Crossref REST API."""

    def __init__(self, config):
        super().__init__(config)
        if not self.retriever_config.issn:
            raise ValueError("source.crossref.issn must be a list of journal ISSNs")

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        days_back = self.retriever_config.get("days_back", 2)
        since = (date.today() - timedelta(days=days_back)).isoformat()
        filters = [f"issn:{i}" for i in self.retriever_config.issn]
        filters += [f"from-index-date:{since}", "type:journal-article"]
        rows = self.retriever_config.get("rows", 100)
        params = {
            "filter": ",".join(filters),
            "rows": rows,
            "select": "DOI,title,author,abstract,URL,container-title,created",
            "cursor": "*",
        }
        if mailto := self.retriever_config.get("mailto"):
            params["mailto"] = mailto
        items: list[dict[str, Any]] = []
        try:
            while True:
                response = requests.get(CROSSREF_API, params=params, timeout=30)
                response.raise_for_status()
                message = response.json()["message"]
                batch = message.get("items", [])
                items.extend(batch)
                if len(batch) < rows or not message.get("next-cursor"):
                    break
                params["cursor"] = message["next-cursor"]
                sleep(1)
        except Exception as e:
            # A single unavailable source must not kill the whole daily run
            logger.error(f"Failed to retrieve Crossref papers: {e}")
        if self.config.executor.debug:
            items = items[:10]
        return items

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        titles = raw_paper.get("title") or []
        if not titles:
            return None
        title = strip_html(titles[0])
        journal = (raw_paper.get("container-title") or [""])[0]
        authors = [
            " ".join(part for part in [a.get("given"), a.get("family")] if part)
            for a in raw_paper.get("author", [])
        ]
        authors = [a for a in authors if a]
        abstract = strip_html(raw_paper.get("abstract"))
        if not abstract:
            # Some publishers do not deposit abstracts; fall back to the title
            # so the reranker can still score the paper instead of dropping it.
            abstract = title
        url = raw_paper.get("URL") or f"https://doi.org/{raw_paper['DOI']}"
        display_title = f"[{journal}] {title}" if journal else title
        return Paper(
            source=self.name,
            title=display_title,
            authors=authors,
            abstract=abstract,
            url=url,
        )
