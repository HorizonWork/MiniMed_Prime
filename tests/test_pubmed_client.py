from __future__ import annotations

from pathlib import Path

from src.utils.pubmed_client import PubMedArticle, PubMedClient


class StubPubMedClient(PubMedClient):
    def __init__(self, cache_path: Path) -> None:
        super().__init__(cache_path=cache_path)
        self.search_calls = 0
        self.fetch_calls = 0

    def search(self, query: str, retmax: int | None = None) -> list[str]:
        self.search_calls += 1
        return ["12345678"]

    def fetch_details(self, pmids: list[str]) -> list[PubMedArticle]:
        self.fetch_calls += 1
        return [
            PubMedArticle(
                pmid="12345678",
                title="Aspirin in stroke prevention",
                abstract="Aspirin reduces recurrent stroke in selected populations.",
            )
        ]


def test_pubmed_client_uses_jsonl_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "pubmed_cache.jsonl"
    client = StubPubMedClient(cache_path=cache_path)

    first_result = client.search_and_fetch("aspirin recurrent stroke")
    second_result = client.search_and_fetch("aspirin recurrent stroke")

    assert [article.pmid for article in first_result] == ["12345678"]
    assert [article.pmid for article in second_result] == ["12345678"]
    assert client.search_calls == 1
    assert client.fetch_calls == 1
    assert cache_path.exists()

    reloaded_client = PubMedClient(cache_path=cache_path)
    cached_result = reloaded_client.search_and_fetch("aspirin recurrent stroke")
    assert [article.pmid for article in cached_result] == ["12345678"]
