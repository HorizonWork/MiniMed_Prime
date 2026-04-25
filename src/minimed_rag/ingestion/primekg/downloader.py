"""PrimeKG downloader: stream the Harvard Dataverse CSV to disk."""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path
from urllib.error import URLError

from minimed_rag.ingestion.base import RawFile

logger = logging.getLogger(__name__)


DEFAULT_URL = "https://dataverse.harvard.edu/api/access/datafile/6180626"


class PrimeKGDownloader:
    """Idempotent downloader.

    ``ensure_download(release)`` is the main entry point used by the
    streaming pipeline. ``download_release(release)`` is retained for the
    legacy ``SourceIngestionPipeline.download()`` hook but should not be
    used for the production CSV (it materialises the full 1.5 GB file as
    bytes).
    """

    def __init__(
        self,
        data_dir: str | Path = "data/primekg",
        url: str = DEFAULT_URL,
        force: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.url = url
        self.force = force

    def local_path(self, release: str) -> Path:
        return self.data_dir / release / "kg.csv"

    def ensure_download(self, release: str) -> Path:
        dest = self.local_path(release)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not self.force:
            logger.info("primekg: %s already present (%d bytes)", dest, dest.stat().st_size)
            return dest

        logger.info("primekg: downloading from %s -> %s", self.url, dest)
        tmp = dest.with_suffix(".download")
        try:
            with urllib.request.urlopen(self.url, timeout=300) as response:
                with tmp.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
        except URLError as exc:
            raise RuntimeError(f"PrimeKG download failed: {exc}") from exc
        tmp.replace(dest)
        logger.info("primekg: saved %d bytes", dest.stat().st_size)
        return dest

    def download_release(self, release: str) -> list[RawFile]:
        path = self.ensure_download(release)
        return [RawFile(name=path.name, bytes=path.read_bytes(), release=release)]
