"""Scispacy NER + UMLS entity linker.

Imports ``spacy`` / ``scispacy`` lazily so projects without the ``kg`` extra
keep importable. The first call to ``link`` / ``nlp`` materialises the
model. Pass ``nlp_model`` explicitly to inject a pre-loaded pipeline (used
by tests and by callers that want to share a model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from minimed_rag.common.ids import make_concept_id
from minimed_rag.nlp.linking.entity_linker import LinkedMention

logger = logging.getLogger(__name__)


@dataclass
class ScispacyEntityLinker:
    model_name: str = "en_core_sci_lg"
    use_umls_linker: bool = True
    threshold: float = 0.7
    top_k_candidates: int = 10
    resolve_abbreviations: bool = True
    nlp_model: Any = None
    _loaded: Any = field(default=None, init=False, repr=False)

    def _load(self) -> Any:
        if self.nlp_model is not None:
            return self.nlp_model
        if self._loaded is not None:
            return self._loaded
        try:
            import spacy  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised by integration tests
            raise RuntimeError(
                "spacy / scispacy not installed. Install with: "
                "`uv sync --extra kg` and then run "
                "`bash scripts/download_scispacy_models.sh`."
            ) from exc

        logger.info("scispacy: loading %s", self.model_name)
        nlp = spacy.load(self.model_name)

        if self.resolve_abbreviations and "abbreviation_detector" not in nlp.pipe_names:
            try:  # pragma: no cover - optional component
                from scispacy.abbreviation import AbbreviationDetector  # noqa: F401

                nlp.add_pipe("abbreviation_detector")
            except Exception as exc:  # pragma: no cover
                logger.warning("scispacy: abbreviation_detector unavailable (%s)", exc)

        if self.use_umls_linker and "scispacy_linker" not in nlp.pipe_names:
            try:  # pragma: no cover - requires ~1 GB KB download on first use
                from scispacy.linking import EntityLinker  # noqa: F401

                nlp.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": "umls"},
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "scispacy: UMLS linker unavailable (%s); "
                    "falling back to NER-only mode. "
                    "The Postgres synonym fallback will still link mentions.",
                    exc,
                )
                self.use_umls_linker = False

        self._loaded = nlp
        return nlp

    @property
    def nlp(self) -> Any:
        return self._load()

    def link(self, text: str) -> list[LinkedMention]:
        nlp = self._load()
        doc = nlp(text)
        return [self._entity_to_mention(text, ent) for ent in doc.ents]

    def _entity_to_mention(self, context: str, ent: Any) -> LinkedMention:
        surface = getattr(ent, "text", "")
        label = getattr(ent, "label_", None)
        kb_ents: list[tuple[str, float]] = []
        if self.use_umls_linker:
            ent_underscore = getattr(ent, "_", None)
            if ent_underscore is not None:
                kb_ents = list(getattr(ent_underscore, "kb_ents", []) or [])

        if kb_ents and kb_ents[0][1] >= self.threshold:
            best_cui, best_score = kb_ents[0]
            candidates = [cui for cui, _ in kb_ents[: self.top_k_candidates]]
            return LinkedMention(
                mention_text=surface,
                expanded_text=surface,
                entity_id="",
                concept_id=make_concept_id(best_cui),
                cui=best_cui,
                semantic_type=label,
                confidence=float(best_score),
                candidates=candidates,
            )
        return LinkedMention(
            mention_text=surface,
            expanded_text=surface,
            entity_id="",
            concept_id=None,
            cui=None,
            semantic_type=label,
            confidence=0.0,
            candidates=[],
        )
