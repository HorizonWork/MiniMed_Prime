# MiniMed Research Roadmap

This document defines the agile plan for turning the current MiniMed scaffold into a biomedical KG-RAG research prototype optimized for benchmark performance.

## Product Direction

MiniMed is a research prototype, not a production medical product yet. The near-term goal is to improve benchmark performance on biomedical and medical QA tasks through retrieval, knowledge graph construction, reasoning paths, and low-cost local models.

Primary constraints:

- Research-first: optimize experiments, metrics, ablations, and reproducibility.
- Benchmark-first: every major feature should improve or explain benchmark results.
- Free/local-first: prefer open datasets, local inference, open-weight models, and no paid APIs by default.
- No PHI: only public biomedical/medical datasets are in scope for now.
- Productization later: clinical safety, compliance, PHI handling, and user-facing product flows are deferred.

## Benchmark Strategy

The primary benchmark should be medical MIRAGE, meaning Medical Information Retrieval-Augmented Generation Evaluation. It is a strong fit because it evaluates RAG in a realistic zero-shot, multiple-choice, question-only retrieval setting. The medical MIRAGE benchmark contains five component datasets: MMLU-Med, MedQA-US, MedMCQA, PubMedQA*, and BioASQ-Y/N.

Primary benchmark:

- MIRAGE medical aggregate score.
- Component scores: MMLU-Med, MedQA-US, MedMCQA, PubMedQA*, BioASQ-Y/N.
- Report both exact answer accuracy and retrieval diagnostics.

Secondary benchmarks:

- MedQA-US for USMLE-style clinical reasoning.
- MedMCQA for large-scale medical exam coverage.
- PubMedQA for biomedical research QA.
- BioASQ-Y/N for biomedical literature QA.
- MMLU medical subsets for broad medical knowledge.

Stretch benchmark:

- MedHopQA for multi-step disease, gene, and chemical reasoning once graph/path reasoning is ready.

Important rule: do not tune directly on benchmark test labels. Use development splits, local validation slices, and ablations. Treat official test scores as final evaluation, not training feedback.

## Free/Low-Cost Model Policy

Default path:

- Run all retrieval, indexing, evaluation, and most generation locally.
- Use BM25 as the first retrieval baseline because it is cheap, deterministic, and strong on exams.
- Add open biomedical dense retrievers only after BM25 metrics are visible.
- Use local open-weight LLMs through Ollama, llama.cpp, vLLM, or SGLang.

Model candidates by hardware:

- CPU or small GPU: quantized 7B-8B instruct model for pipeline debugging.
- 12-16 GB VRAM: quantized Qwen/Gemma-class 12B-14B model.
- 24 GB VRAM: quantized 27B-32B model or small MoE model.
- Multi-GPU: larger open-weight reasoning model for final benchmark runs.

Recommended local baselines:

- Fast baseline: BM25 + small local instruct model.
- Stronger low-cost baseline: BM25 + dense biomedical retriever + reranker + 12B-32B local instruct model.
- Reasoning experiment: retrieval + KG paths + local reasoning model, compared against retrieval-only.

Paid API usage should be optional and isolated behind a provider interface. It can be used for calibration or upper-bound experiments, but the main project should remain runnable without paid services.

## North-Star Metrics

Benchmark metrics:

- Accuracy per benchmark and macro-average across tasks.
- Answer extraction validity: model must output one of the allowed options.
- Refusal rate for insufficient evidence when running open-ended RAG tests.

Retrieval metrics:

- Recall@k where gold support is available.
- MRR and nDCG for ranked evidence.
- Corpus hit rate by source: PubMed, StatPearls, UMLS, PrimeKG, textbooks if added.
- Question-only retrieval performance, without answer choices.

KG metrics:

- Entity linking precision on validation examples.
- Assertion coverage by benchmark domain.
- Evidence-backed assertion ratio.
- Duplicate/conflict rate.
- Path coverage for reasoning questions.

Experiment metrics:

- Cost per run.
- Runtime per benchmark.
- Index build time.
- Reproducibility: fixed configs, seeds, dataset versions, and graph versions.

## Agile Operating Model

Use one-week or two-week sprints. Each sprint should ship one measurable vertical slice.

Every story should include:

- Hypothesis: what score or quality issue this might improve.
- Implementation: code/config/data changes.
- Evaluation: exact command and metric.
- Decision: keep, revert, or revise based on evidence.

Definition of Done for research work:

- Code path is runnable from CLI.
- Config is checked into `configs/`.
- Metrics are written to `artifacts/reports/`.
- At least one focused test or smoke test exists.
- Result is recorded in an experiment log.

## Phase 0 - Research Baseline

Goal: establish a reproducible benchmark harness before optimizing anything.

Build:

- Dataset adapters for MIRAGE component tasks.
- Unified multiple-choice evaluator.
- Prompt templates for zero-shot answer selection.
- Local model provider interface.
- Result writer for JSONL and Markdown reports.

Acceptance criteria:

- A command can run a small benchmark slice end-to-end.
- The report includes accuracy, invalid-output rate, latency, and config hash.
- Baseline uses no paid API.

Suggested command:

```bash
minimed evaluate benchmark --suite mirage --limit 50 --model local
```

## Phase 1 - Repository Stabilization

Goal: make the scaffold reliable enough for repeated experiments.

Build:

- Finish package rename consistency: `biomed_kg` to `minimed_rag`.
- Add pytest, lint, format, and smoke checks.
- Add CLI smoke tests for `minimed --help`.
- Add API health test.
- Normalize Makefile targets for local research.

Acceptance criteria:

- `make test` passes.
- `uv run minimed --help` passes.
- `make up` starts local dependencies.
- No stale imports from the old package name remain.

## Phase 2 - Retrieval-Only Baseline

Goal: get a strong, cheap RAG baseline before KG or TRM complexity.

Build:

- Question-only BM25 retrieval over selected corpora.
- Corpus loader for PubMed/StatPearls and any public textbook corpus that is legally usable.
- Context builder for multiple-choice QA.
- Answer parser that maps model output to A/B/C/D or yes/no/maybe.
- Retrieval diagnostics in evaluation reports.

Acceptance criteria:

- MIRAGE slice runs with BM25-only retrieval.
- Retrieval-only beats no-retrieval baseline on at least one component dataset.
- Reports separate retrieval failures from generation failures.

## Phase 3 - Biomedical Dense Retrieval

Goal: improve recall and robustness beyond lexical matching.

Build:

- Vector index pipeline for chunks and abstracts.
- Open biomedical embedding model integration.
- Hybrid retriever: BM25 + dense.
- Optional reranker interface.
- Ablation configs: BM25 only, dense only, hybrid, hybrid+rerank.

Acceptance criteria:

- Hybrid retrieval has measurable Recall@k or accuracy gain over BM25 on validation slices.
- Index artifacts are versioned by corpus and embedding model.
- Evaluation can reproduce all ablations from config.

## Phase 4 - Knowledge Graph MVP

Goal: add structured biomedical knowledge only where it helps benchmark reasoning.

Build:

- UMLS/PrimeKG ingestion slice.
- Entity normalization and synonym mapping.
- Entity linker for benchmark questions.
- Assertion schema with provenance and confidence.
- Neo4j loader for a small but queryable graph.

Acceptance criteria:

- Questions can be linked to candidate biomedical entities.
- KG lookup returns relevant neighboring entities/assertions.
- KG-augmented retrieval can be compared against retrieval-only.

## Phase 5 - KG-Augmented RAG

Goal: use the graph to improve context selection and answer grounding.

Build:

- Query planner that decides text-only, graph-only, or hybrid path.
- Graph retriever for one-hop and two-hop evidence.
- Context builder that merges graph paths and text snippets.
- Conflict reporting when retrieved evidence disagrees.

Acceptance criteria:

- KG-RAG has an ablation report against hybrid text RAG.
- At least one benchmark component shows a gain or clear diagnostic reason for no gain.
- Retrieved graph paths are serializable and inspectable.

## Phase 6 - Reasoning Path Generation

Goal: generate high-quality paths for multi-hop questions and future TRM training.

Build:

- Task classifier for exam, biomedical research, and multi-hop tasks.
- Metapath planner from entity types and predicates.
- Path finder for one-hop, two-hop, and three-hop templates.
- Path scorer and pruner.
- Negative sampler for invalid or weak paths.

Acceptance criteria:

- Path generation works on a validation set.
- Path coverage and path precision are reported.
- MedHopQA-style examples can produce inspectable multi-step paths.

## Phase 7 - TRM Training Prototype

Goal: test whether a trained reasoning module improves answer selection or path ranking.

Build:

- Reasoning dataset builder from benchmark questions and generated paths.
- Dataset splits with leakage checks.
- Training loop, checkpointing, and experiment tracking.
- Objectives: path validity, next-hop prediction, answer selection.
- Evaluation against retrieval-only and KG-RAG baselines.

Acceptance criteria:

- A small TRM model trains locally.
- Checkpoints and metrics are saved.
- TRM improves at least one target metric or produces actionable failure analysis.

## Phase 8 - Benchmark Optimization Loop

Goal: systematically improve benchmark scores with disciplined ablations.

Experiment tracks:

- Corpus selection: PubMed vs StatPearls vs UMLS/PrimeKG vs mixed corpora.
- Chunking: paragraph, section-aware, sliding window, citation-aware.
- Retrieval: BM25, dense, hybrid, reranking, graph expansion.
- Prompting: direct answer, evidence-first, option-elimination, self-consistency if affordable.
- Model: small local, medium local, reasoning local.
- KG: no graph, entity hints, one-hop paths, multi-hop paths.

Acceptance criteria:

- Every improvement has an ablation table.
- Best config is frozen in `configs/experiments/`.
- Final benchmark run is reproducible from a single command.

## Phase 9 - Research Release

Goal: produce a credible research artifact.

Build:

- Reproducible pipeline instructions.
- Benchmark report.
- Error analysis notebooks.
- Model card or system card.
- Cost/runtime table.
- Limitations and leakage analysis.

Acceptance criteria:

- Another developer can reproduce a small run locally.
- Full run instructions are documented.
- The project clearly states what is research-only and not medical advice.

## First Three Sprints

Sprint 1: benchmark harness and repo stabilization.

- Add `minimed evaluate benchmark`.
- Add MIRAGE component dataset interfaces.
- Add local model provider stub and answer parser.
- Add smoke tests for CLI/API.
- Produce first no-retrieval baseline report.

Sprint 2: BM25 RAG baseline.

- Load one text corpus.
- Build BM25 index.
- Run question-only retrieval.
- Add context builder and evaluation report.
- Compare no-retrieval vs BM25-RAG.

Sprint 3: hybrid retrieval ablation.

- Add vector embeddings and vector index.
- Add hybrid retriever.
- Add configs for BM25, dense, hybrid.
- Run validation ablations.
- Decide whether reranking is worth the cost.

## Key Risks

- Benchmark leakage: prevent by separating training, validation, and final test usage.
- Dataset licensing: record source, license, and permitted usage before indexing.
- Local model variance: pin model versions, quantization, decoding params, and prompts.
- Medical hallucination: even research outputs should prefer evidence-grounded answers.
- Complexity creep: KG and TRM should be added only after retrieval baselines are measurable.

## References

- Medical MIRAGE / MEDRAG: https://aclanthology.org/2024.findings-acl.372/
- MEDRAG toolkit: https://github.com/Teddy-XiongGZ/MedRAG
- General MIRAGE RAG evaluation benchmark: https://arxiv.org/abs/2504.17137
- MMLU: https://arxiv.org/abs/2009.03300
- MedQA: https://arxiv.org/abs/2009.13081
- MedMCQA: https://proceedings.mlr.press/v174/pal22a.html
- PubMedQA: https://arxiv.org/abs/1909.06146
- BioASQ: https://bioasq.org/home
- MedHopQA: https://www.ncbi.nlm.nih.gov/research/bionlp/medhopqa
- Qwen3: https://qwenlm.github.io/blog/qwen3/
- Gemma 3: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-3/
