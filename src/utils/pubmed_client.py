from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.kaggle_env import KaggleEnv
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_TOOL_NAME = "MiniMedPrime"
PUBMED_DATABASE = "pubmed"
PUBMED_SEARCH_ENDPOINT = "esearch.fcgi"
PUBMED_FETCH_ENDPOINT = "efetch.fcgi"
DEFAULT_SEARCH_RETMAX = 50
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_CACHE_PATH = KaggleEnv.ensure_writeable(KaggleEnv.path("data/pubmed_cache.jsonl"))
DEFAULT_RATE_LIMIT_WITHOUT_API_KEY = 0.34
DEFAULT_RATE_LIMIT_WITH_API_KEY = 0.11
DEFAULT_USER_AGENT = "MiniMedPrime/0.1 (+https://github.com/openai/minimed-prime)"
DEFAULT_RETRY_TOTAL = 3
DEFAULT_RETRY_BACKOFF_FACTOR = 0.5
DEFAULT_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


@dataclass(slots=True)
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    publication_year: str | None = None
    mesh_terms: tuple[str, ...] = ()

    @classmethod
    def from_cache_record(cls, payload: dict[str, Any]) -> PubMedArticle:
        return cls(
            pmid=str(payload["pmid"]),
            title=str(payload.get("title", "")),
            abstract=str(payload.get("abstract", "")),
            journal=payload.get("journal"),
            publication_year=payload.get("publication_year"),
            mesh_terms=tuple(payload.get("mesh_terms", ())),
        )


@dataclass(slots=True)
class PubMedClientConfig:
    api_key: str | None = None
    cache_path: Path = field(default_factory=lambda: DEFAULT_CACHE_PATH)
    base_url: str = PUBMED_BASE_URL
    search_retmax: int = DEFAULT_SEARCH_RETMAX
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    tool_name: str = PUBMED_TOOL_NAME
    rate_limit_seconds: float = DEFAULT_RATE_LIMIT_WITHOUT_API_KEY


class PubMedClient:
    """Thin wrapper around NCBI E-utilities with JSONL query caching."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: Path | None = None,
        session: requests.Session | None = None,
        search_retmax: int = DEFAULT_SEARCH_RETMAX,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        resolved_cache_path = cache_path or DEFAULT_CACHE_PATH
        resolved_cache_path = KaggleEnv.ensure_writeable(resolved_cache_path if resolved_cache_path.is_absolute() else KaggleEnv.path(resolved_cache_path))
        rate_limit_seconds = DEFAULT_RATE_LIMIT_WITH_API_KEY if api_key else DEFAULT_RATE_LIMIT_WITHOUT_API_KEY
        self.config = PubMedClientConfig(
            api_key=api_key,
            cache_path=resolved_cache_path,
            search_retmax=search_retmax,
            timeout_seconds=timeout_seconds,
            rate_limit_seconds=rate_limit_seconds,
        )
        self.session = session or self._build_session()
        self._cache: dict[str, list[PubMedArticle]] = self._load_cache()
        self._next_request_time = 0.0
        self.structured_logger = StructuredLogger("pubmed_client", DEFAULT_LOG_DIR)

    def search(self, query: str, retmax: int | None = None) -> list[str]:
        """Search PubMed and return PMID strings ordered by NCBI relevance."""

        effective_retmax = retmax or self.config.search_retmax
        started_at = time.perf_counter()
        response = self._request(
            PUBMED_SEARCH_ENDPOINT,
            {
                "db": PUBMED_DATABASE,
                "term": query,
                "retmax": effective_retmax,
                "retmode": "json",
                "sort": "relevance",
            },
        )
        payload = response.json()
        id_list = payload.get("esearchresult", {}).get("idlist", [])
        pmids = [str(pmid) for pmid in id_list if str(pmid).strip()]
        self.structured_logger.log_event(
            "pubmed_search",
            {
                "query": query,
                "retmax": effective_retmax,
                "k_returned": len(pmids),
                "backend_used": "ncbi_eutils",
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )
        return pmids

    def fetch_details(self, pmids: Sequence[str]) -> list[PubMedArticle]:
        """Fetch article metadata and abstracts for concrete PMIDs."""

        normalized_pmids = [str(pmid) for pmid in pmids if str(pmid).strip()]
        if not normalized_pmids:
            return []

        started_at = time.perf_counter()
        response = self._request(
            PUBMED_FETCH_ENDPOINT,
            {
                "db": PUBMED_DATABASE,
                "id": ",".join(normalized_pmids),
                "retmode": "xml",
            },
        )
        articles = self._parse_fetch_response(response.text)
        self.structured_logger.log_event(
            "pubmed_fetch",
            {
                "pmid_count": len(normalized_pmids),
                "articles_returned": len(articles),
                "backend_used": "ncbi_eutils",
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )
        return articles

    def search_and_fetch(self, query: str, retmax: int | None = None, use_cache: bool = True) -> list[PubMedArticle]:
        """Resolve a free-text query into cached article records."""

        effective_retmax = retmax or self.config.search_retmax
        cache_key = self._cache_key(query=query, retmax=effective_retmax)
        if use_cache and cache_key in self._cache:
            self.structured_logger.log_event(
                "pubmed_cache_hit",
                {
                    "query": query,
                    "retmax": effective_retmax,
                    "cached_articles": len(self._cache[cache_key]),
                    "backend_used": "jsonl_cache",
                    "latency_ms": 0.0,
                },
            )
            return list(self._cache[cache_key])

        started_at = time.perf_counter()
        pmids = self.search(query=query, retmax=effective_retmax)
        articles = self.fetch_details(pmids)
        if use_cache:
            self._cache[cache_key] = list(articles)
            self._append_cache_entry(query=query, retmax=effective_retmax, articles=articles)
        self.structured_logger.log_event(
            "pubmed_search_and_fetch",
            {
                "query": query,
                "retmax": effective_retmax,
                "articles_returned": len(articles),
                "backend_used": "ncbi_eutils",
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )
        return list(articles)

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
        retry = Retry(
            total=DEFAULT_RETRY_TOTAL,
            read=DEFAULT_RETRY_TOTAL,
            connect=DEFAULT_RETRY_TOTAL,
            backoff_factor=DEFAULT_RETRY_BACKOFF_FACTOR,
            status_forcelist=DEFAULT_RETRY_STATUS_CODES,
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _load_cache(self) -> dict[str, list[PubMedArticle]]:
        cache: dict[str, list[PubMedArticle]] = {}
        cache_path = self.config.cache_path
        if not cache_path.exists():
            return cache

        with cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed PubMed cache line in %s", cache_path)
                    self.structured_logger.log_event(
                        "pubmed_cache_malformed_line",
                        {
                            "cache_path": str(cache_path),
                            "backend_used": "jsonl_cache",
                            "latency_ms": 0.0,
                        },
                    )
                    continue

                query = payload.get("query")
                retmax = payload.get("retmax", self.config.search_retmax)
                articles_payload = payload.get("articles", [])
                if not isinstance(query, str):
                    continue
                cache_key = self._cache_key(query=query, retmax=int(retmax))
                cache[cache_key] = [
                    PubMedArticle.from_cache_record(article_payload) for article_payload in articles_payload
                ]
        return cache

    def _append_cache_entry(self, query: str, retmax: int, articles: Sequence[PubMedArticle]) -> None:
        cache_path = self.config.cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query": query,
            "retmax": retmax,
            "timestamp": int(time.time()),
            "articles": [asdict(article) for article in articles],
        }
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.structured_logger.log_event(
            "pubmed_cache_append",
            {
                "cache_path": str(cache_path),
                "articles_cached": len(list(articles)),
                "backend_used": "jsonl_cache",
                "latency_ms": 0.0,
            },
        )

    def _request(self, endpoint: str, params: dict[str, Any]) -> requests.Response:
        self._respect_rate_limit()
        request_params = {
            "tool": self.config.tool_name,
            **params,
        }
        if self.config.api_key:
            request_params["api_key"] = self.config.api_key

        response = self.session.get(
            urljoin(self.config.base_url, endpoint),
            params=request_params,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response

    def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        if now < self._next_request_time:
            time.sleep(self._next_request_time - now)
        self._next_request_time = time.monotonic() + self.config.rate_limit_seconds

    def _parse_fetch_response(self, xml_payload: str) -> list[PubMedArticle]:
        articles: list[PubMedArticle] = []
        root = ET.fromstring(xml_payload)
        for article_node in root.findall(".//PubmedArticle"):
            citation_node = article_node.find("./MedlineCitation")
            article_detail_node = citation_node.find("./Article") if citation_node is not None else None
            pmid = self._extract_text(citation_node.find("./PMID")) if citation_node is not None else ""
            title = self._extract_text(article_detail_node.find("./ArticleTitle")) if article_detail_node is not None else ""
            abstract_sections = []
            if article_detail_node is not None:
                for abstract_node in article_detail_node.findall("./Abstract/AbstractText"):
                    abstract_sections.append(self._extract_text(abstract_node))
            abstract = " ".join(section for section in abstract_sections if section).strip()
            journal = (
                self._extract_text(article_detail_node.find("./Journal/Title")) if article_detail_node is not None else None
            )
            publication_year = (
                self._extract_text(article_detail_node.find("./Journal/JournalIssue/PubDate/Year"))
                if article_detail_node is not None
                else None
            )
            mesh_terms = tuple(
                self._extract_text(mesh_heading.find("./DescriptorName"))
                for mesh_heading in article_node.findall("./MedlineCitation/MeshHeadingList/MeshHeading")
                if self._extract_text(mesh_heading.find("./DescriptorName"))
            )
            if pmid:
                articles.append(
                    PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        journal=journal or None,
                        publication_year=publication_year or None,
                        mesh_terms=mesh_terms,
                    )
                )
        return articles

    def _cache_key(self, query: str, retmax: int) -> str:
        return f"{query.strip().lower()}::{retmax}"

    @staticmethod
    def _extract_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return "".join(node.itertext()).strip()


__all__ = ["PubMedArticle", "PubMedClient", "PubMedClientConfig"]
