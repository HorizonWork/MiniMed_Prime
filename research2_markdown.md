# Medical Reasoning System v3 — Comprehensive Technical Specification

## Executive summary and top-line recommendation

**Medical Reasoning System v3** là một agentic recursive KG-RAG pipeline năm-tầng được thiết kế để giảm hallucination trong medical QA bằng cách kết hợp (1) retrieval đa nguồn từ PrimeKG + PubMed + UMLS, (2) tensor-based recursive reasoning với Samsung TRM 7M, (3) bốn agentic judges độc lập, và (4) synthesis trên local quantized medical LLM. **Khuyến nghị chính**: deploy trên Kaggle 2×T4 với `Intelligent-Internet/II-Medical-8B` (Q4_K_M GGUF) làm synthesizer, `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` làm node/entity encoder, `ncbi/MedCPT-*` cho PubMed retrieval, TRM fine-tuned on MedReason-derived GraphTensor pairs, và Gemini 2.5 Flash free tier làm external judge để decorrelate từ synthesizer. Four hard confusion points đã được resolved: **Samsung TRM không phải là "geometric reasoner"** (đó là mischaracterization — paper gọi là "recursive reasoning model"); paper là arXiv 2510.04871 của Alexia Jolicoeur-Martineau (Samsung SAIL Montréal, Oct 2025), repo `github.com/SamsungSAILMontreal/TinyRecursiveModels`. **MedReason là both dataset and model** (arXiv 2504.00993, UCSC-VLAA) với 32,682 KG-grounded CoT pairs và public MedReason-8B weights. **Chỉ Nemotron-Nano-8B-v1 fit T4 cleanly** trong family Nemotron (các Mamba-hybrid biến thể fail trên sm_75). **PrimeKG statistics vẫn holds** (129,375 nodes / ~4M undirected edges / 30 relation types; không có v2 chính thức, chỉ có OMIM extension Dec 2023).

The system achieves hallucination reduction through four independent mechanisms: (i) **provenance enforcement** — mọi claim trong final answer phải link tới PrimeKG edge ID hoặc PubMed PMID; (ii) **multiplicative confidence propagation** (AMG-RAG style) truyền uncertainty qua reasoning paths; (iii) **abstention** khi aggregate confidence < τ = 0.7; (iv) **hard constraints** cho life-critical fields (dosage ranges, contraindication checks via PrimeKG `contraindication` relation). Full evaluation trên 6 benchmarks (MedQA, MedMCQA, PubMedQA, BioASQ-13b, MMLU-med 1089Q, MedBullets 308Q) + 3 hallucination benchmarks (Med-HALT, MedHallu 10K, MEDEC-MS) với McNemar test + bootstrap CI cho significance.

---

# Part 1 — Hallucination Detection Framework (PRIORITY 1)

## 1.1 Taxonomy for medical domain

Chúng ta adopt một **7-category taxonomy** tổng hợp từ Med-HALT (Pal 2023, arXiv 2307.15343), MedHallu (Pandit 2025, arXiv 2502.14302), MedHalu (Agarwal 2024, arXiv 2409.19492), và medical hallucination survey (Kim 2025, arXiv 2503.05777). Each category maps to a concrete detector trong pipeline.

| # | Category | Definition | Concrete medical example | Primary detector |
|---|---|---|---|---|
| H1 | **Factual hallucination** | Claim contradicts established medical fact | *"Metformin là first-line cho Type 1 Diabetes"* (sai — đó là Type 2) | KG triple lookup in PrimeKG `indication` edges |
| H2 | **Logical hallucination** | Premises đúng nhưng inference invalid | *"Patient có X → X gây Y → thus Y hiện diện"* khi chain X→Y không có trong KG | TRM path validity score |
| H3 | **Contextual hallucination** | Info không có trong provided evidence | CoT cites a PubMed finding không có trong retrieved passages | NLI entailment vs context |
| H4 | **Source attribution hallucination** | Wrong PMIDs / wrong KG edge IDs cited | *"Theo PMID 12345678, drug X causes Y"* nhưng PMID đó không tồn tại hoặc không chứa claim | PubMed E-utilities verification + edge ID lookup |
| H5 | **Life-critical hallucination** | Dosage, contraindication, DDI errors | *"Warfarin 50 mg PO daily"* (fatal — therapeutic là 2–10 mg) | Hard numeric constraint + PrimeKG `contraindication`/`drug_drug` |
| H6 | **Temporal hallucination** | Outdated guideline stated as current | Citing pre-2020 HFrEF guidelines without SGLT2i | Publication-date filter on PubMed + guideline versioning |
| H7 | **Population hallucination** | Generalization across demographics where evidence disagrees | Pediatric dosing từ adult trial không có pediatric data | UMLS semantic-type filter (Age Group / Population Group) |

H5 được escalate thành **blocking error** — pipeline phải abstain nếu detector flags H5, không bao giờ "best-effort".

## 1.2 Hallucination metrics — implementation details

### 1.2.1 RAGAS Faithfulness, adapted cho medical domain

Core formula (verified từ `docs.ragas.io`):

```
Faithfulness = |{c ∈ C(r) : c is entailed by R}| / |C(r)|
```

where `r` = generated response, `C(r)` = atomic claim decomposition của `r`, `R` = retrieved context. Ragas implementation uses **two LLM calls**: (1) `statement_prompt` breaks response into statements pronoun-free, (2) `nli_statements_message` classifies each statement as `{1, 0, -1}` = `{supported, neutral, contradicted}`. Score counts only `1`.

**Four medical-domain adaptations** (bắt buộc):

1. **UMLS synonym expansion trước NLI**. Trước khi gọi NLI model, expand mỗi medical term trong claim thành UMLS synonym set via `scispacy_linker`. A claim *"Patient has MI"* entails context *"Patient suffered myocardial infarction"* chỉ khi ta biết `MI ≡ C0027051`. Without expansion, RAGAS dưới-counts supported claims by ~15–20% trên MedQA CoT data.

2. **Abbreviation dictionary pre-pass**. Build static dict `{"AFib": "atrial fibrillation", "COPD": "chronic obstructive pulmonary disease", ...}` covering top-500 medical abbreviations from UMLS LRABR table. Preprocess both claim và context trước statement extraction.

3. **Numeric tolerance in NLI**. Default NLI coi *"5 mg"* ≠ *"5.0 mg"* ≠ *"5 milligrams"*. Inject rule: two numeric expressions entail iff normalized (unit-converted) values match within ±5% relative hoặc ±0.1 absolute (for dosages). Critical vì **exact dosage mismatches là H5**.

4. **Replace default NLI judge with medical NLI**. Swap `cross-encoder/nli-deberta-v3-base` for a MedNLI-fine-tuned variant (`cross-encoder/nli-deberta-v3-base` + continued training on MedNLI 14k pairs) hoặc use `Vectara/hallucination_evaluation_model` (HHEM-2.1-Open T5) that Ragas natively supports via `FaithfulnesswithHHEM`.

### 1.2.2 F1 hallucination score

Dual metric decomposing faithfulness:

```
Precision_hall = |supported claims in r| / |total claims in r|     # = RAGAS Faithfulness
Recall_hall    = |gold claims covered in r| / |total gold claims|   # needs gold reasoning
F1_hall        = 2·P·R / (P+R)
```

Recall requires **gold claim set** — available on MedReason (CoT ground truth), PubMedQA (long_answer), BioASQ (ideal answer). For MCQ benchmarks không có gold CoT, recall = 0 by construction → báo cáo chỉ precision.

### 1.2.3 Contradiction detection rate

For every pair `(claim_i, claim_j)` trong same reasoning chain và cho mọi pair `(claim_i, evidence_k)`:

```
CDR = |{(i,j) : NLI(i,j) = contradiction}| / |all pairs examined|
```

Implementation: batch NLI với `cross-encoder/nli-deberta-v3-base` (on T4: ~500 pairs/sec). Target threshold: **CDR ≤ 2%** trong final answer. CDR > 5% → route tới Judge 3 (Reasoning Soundness Auditor) cho revision.

### 1.2.4 Benchmark-specific metrics

| Benchmark | arXiv | Size | Primary metric | Pipeline role |
|---|---|---|---|---|
| **Med-HALT** | 2307.15343 | 7 reasoning + memory sub-tasks, ~18K items | Accuracy + Pointwise Score (rewards correct + penalizes wrong confidently, per §3.3 paper) | Zero-shot stress test — v3 phải beat zero-shot baseline by ≥10 pts |
| **MedHallu** (Pandit 2025) | 2502.14302 | 10,000 QA pairs (1K from PQA-L + 9K from PQA-A) stratified easy/med/hard | Binary F1; best existing model = 0.625 on "hard" | Detection capability — v3 Judge should hit ≥0.75 F1 |
| **MedHalu** (Agarwal 2024) | 2409.19492 | Real-world patient queries + hallucinated LLM responses, expert-annotated spans | Span-level F1, type classification accuracy | Used for Judge training signal |
| **HaluEval biomed** | 2305.11747 | 10K QA subset với GPT-3.5 generated hallucinations | Binary detection acc | Cross-domain generalization check |
| **MEDEC** (Abacha 2025) | 2412.19260 | 3,848 clinical notes, 5 error types (Diagnosis/Management/Treatment/Pharmacotherapy/CausalOrganism); MS split public, UW via DUA | Error-flag accuracy, error-sentence accuracy, correction ROUGE | Direct H1 + H5 evaluation; Claude 3.5 Sonnet is current top (70.16% flag acc) |
| **MedHallBench** | 2412.18947 | VQA + IRG, ACHMII/ACHMIS metrics | Component-level hallucination % | Multimodal skip unless adding imaging |
| **K-QA** (Manes 2024) | 2401.14493 | 201 real patient questions with must-have / nice-to-have claims | Claim-level P/R/F1 | Long-form evaluation |

v3 must report **Med-HALT pointwise score, MedHallu F1 (overall + hard tier), MEDEC-MS error-flag accuracy, K-QA must-have recall** as its four hallucination headline numbers.

### 1.2.5 SelfCheckGPT — assessment and alternative

SelfCheckGPT (Manakul 2023, arXiv 2303.08896) measures hallucination via **sampling consistency**: generate N stochastic samples ở temperature > 0, compute pairwise similarity (BERTScore / NLI / MQAG / prompt variants); low consistency → high hallucination probability.

**Không suitable for structured CoT reasoning trong v3** vì ba lý do: (1) CoT steps có serial dependencies — sample 2 có thể divergent chỉ vì branching at step 3, không phải vì hallucination; (2) cost cao — N=5 samples × 8B model × 1K test questions ≈ 40K inference calls; (3) SelfCheckGPT's reference-free nature duplicates với RAGAS Faithfulness mà ta đã compute against KG evidence (stronger signal).

**Thay thế**: dùng **semantic entropy** (Farquhar et al., Nature 2024, doi 10.1038/s41586-024-07421-0) on final answer only (not full CoT) — cluster N samples by bidirectional NLI entailment, compute entropy over cluster probabilities; threshold flag at SE > 1.5 nats. N=3 samples đủ cho T4 budget. Acts as a **secondary abstention trigger** on top of KG-based confidence.

## 1.3 Hallucination detection methods — algorithms

### 1.3.1 Claim extraction + verification pipeline

```python
# Pseudocode — runs as part of Judge 2 (Evidence Grounding Inspector)
def verify_answer(answer_text: str, evidence: EvidenceBundle) -> VerificationReport:
    # Step 1: Atomic claim decomposition
    claims = llm_claim_extractor(answer_text)        # FActScore-style prompt, ~1-2 claims/sentence
    
    # Step 2: Medical entity linking per claim
    for c in claims:
        c.entities = scispacy_link(c.text)           # CUIs + spans
        c.abbrevs  = expand_abbreviations(c.text)
        c.numerics = extract_dosage_units(c.text)    # regex + unit normalization
    
    # Step 3: Dual verification
    for c in claims:
        # 3a. KG triple check against PrimeKG subgraph
        triples = extract_triples(c)                 # OpenIE or LLM
        c.kg_support = [primekg.has_edge(h, r, t, tolerance='sem') for h,r,t in triples]
        
        # 3b. Text entailment against PubMed passages
        c.nli_scores = [nli_model(premise=p.text, hypothesis=c.text) 
                        for p in evidence.pubmed_passages]
        c.text_support = any(s == 'entailment' for s in c.nli_scores)
        
        # 3c. Hard constraint check (H5)
        if c.numerics and is_dosage(c):
            c.constraint_ok = dosage_in_range(c.numerics, primekg.drug_info)
    
    # Step 4: Aggregate per-claim verdict
    for c in claims:
        c.supported = (any(c.kg_support) or c.text_support) and c.constraint_ok
    
    return VerificationReport(
        faithfulness = sum(c.supported for c in claims) / len(claims),
        unsupported  = [c for c in claims if not c.supported],
        h5_flags     = [c for c in claims if c.numerics and not c.constraint_ok],
    )
```

### 1.3.2 NLI-based detection — model choice

Pipeline sử dụng **two NLI models** chạy in parallel:

- **Primary**: `cross-encoder/nli-deberta-v3-base` (86M params, ~500 MB, fits GPU 1 alongside verifier) for general-purpose NLI.
- **Medical**: `Vectara/hallucination_evaluation_model` (HHEM-2.1-Open T5-base, ~450 MB) trained explicitly on hallucination detection; Ragas supports natively via `FaithfulnesswithHHEM(device='cuda')`.

Có a third option — `StanfordAIMI/stanford-deidentifier-base` combined with a MedNLI-tuned checkpoint — nhưng MedNLI (Romanov 2018) license restricts redistribution. Workaround: fine-tune DeBERTa-v3 on MedNLI yourself trên Kaggle (~20 min on T4), push to private HF repo.

### 1.3.3 LLM-as-Judge — complete prompt template cho Hallucination Judge

System prompt, user template, và JSON schema trình bày dưới. Template theo best-practice từ G-Eval (arXiv 2303.16634) + Prometheus-2 (arXiv 2405.01535) — structured evaluation with explicit scoring rubric + chain-of-thought rationale before numeric verdict.

```text
SYSTEM:
You are a meticulous medical evidence auditor with training equivalent to a board-certified
internist. Your task is to identify hallucinations in a candidate medical answer by comparing
it strictly against the provided evidence bundle. You MUST NOT use your own medical knowledge
to verify claims — only the evidence. A claim is "supported" only if entailed by the evidence;
a claim is "hallucinated" if (a) contradicted by evidence, (b) absent from evidence, (c) cites
a source (PMID or KG edge) that does not exist in the evidence, (d) gives a drug dose outside
the therapeutic range stated in the PrimeKG drug record, or (e) generalizes across populations
where evidence disagrees. Output strict JSON only — no prose outside JSON.

USER:
# Question
{question}

# Candidate answer
{answer_text}

# Evidence bundle
## PrimeKG subgraph (triples with edge IDs)
{primekg_triples_serialized}

## PubMed passages (PMID + abstract snippet)
{pubmed_passages_serialized}

## Hard constraints (from PrimeKG drug records)
{drug_constraints_serialized}

# Instructions
For EACH atomic factual claim in the candidate answer:
1. Extract the claim verbatim or near-verbatim.
2. Classify into one of: {factual, logical, contextual, source_attribution, life_critical, temporal, population}.
3. Assign verdict: "supported" | "unsupported" | "contradicted" | "out_of_scope".
4. Cite the specific edge_id or PMID that supports/contradicts, or "none" if no evidence addresses it.
5. Score hallucination severity 0–3: 0=supported, 1=minor/unsupported-but-plausible, 
   2=clearly unsupported, 3=life-critical error (dosage, contraindication, DDI).
6. If any claim has severity=3, set overall_recommendation="ABSTAIN".

# Few-shot example
## Evidence:
  KG: (Warfarin, contraindication, Aspirin, edge_id=E_13891)
  PubMed: PMID 28472901 — "Warfarin therapeutic dose for VTE is typically 2–10 mg/day..."
## Answer: "Prescribe Warfarin 50 mg with Aspirin 81 mg for DVT prophylaxis."
## Expected output:
{
  "claims": [
    {"text": "Prescribe Warfarin 50 mg", "type": "life_critical",
     "verdict": "contradicted", "evidence_id": "PMID:28472901",
     "severity": 3, "rationale": "50 mg is 5–25× therapeutic max 10 mg/day"},
    {"text": "with Aspirin 81 mg", "type": "life_critical",
     "verdict": "contradicted", "evidence_id": "edge_id:E_13891",
     "severity": 3, "rationale": "PrimeKG lists Warfarin-Aspirin as contraindication"}
  ],
  "faithfulness_score": 0.0,
  "h5_present": true,
  "overall_recommendation": "ABSTAIN",
  "abstention_reason": "Two life-critical errors detected; patient safety risk."
}

Now produce your JSON output for the candidate answer above.
```

**JSON output schema** (Pydantic-style):

```python
class ClaimVerdict(BaseModel):
    text: str
    type: Literal["factual","logical","contextual","source_attribution",
                  "life_critical","temporal","population"]
    verdict: Literal["supported","unsupported","contradicted","out_of_scope"]
    evidence_id: str            # "PMID:X" | "edge_id:Y" | "none"
    severity: int = Field(ge=0, le=3)
    rationale: str

class JudgeOutput(BaseModel):
    claims: list[ClaimVerdict]
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    h5_present: bool
    overall_recommendation: Literal["ACCEPT","REVISE","ABSTAIN"]
    abstention_reason: str | None
```

Retry/fallback: on invalid JSON parse, retry với temperature=0 + repair prompt; after 3 failures, fall back to deterministic extractor pipeline (§1.3.1) với `overall_recommendation="REVISE"`.

### 1.3.4 Reference-based vs reference-free

| Dataset type | Available reference | Method |
|---|---|---|
| MedQA / MedMCQA / MMLU-med (MCQ) | Gold answer letter only | Reference-free: claim-level verification against retrieved evidence (§1.3.1) |
| PubMedQA | `long_answer` + `final_decision` (yes/no/maybe) | Reference-based: NLI entailment of generated explanation vs `long_answer` |
| BioASQ factoid/list | Exact answers + gold snippets | Reference-based: UMLS-normalized set overlap (F1 on list) + MRR on factoid |
| BioASQ ideal | Gold paragraph | Hybrid: ROUGE-L + BERTScore + LLM-judge on claim-level F1 against gold |
| MedBullets | Gold + explanation | Reference-based on answer; reference-free on CoT |
| MedReason eval split | Gold CoT (KG path) | Reference-based: RCS + RNS + graph-edit distance (§1.4 below) |
| Med-HALT / MedHallu / MEDEC | Hallucination labels | Reference-based: direct binary/multi-class F1 |

### 1.3.5 Multi-sample consistency cho reasoning chains

Thay vì vanilla SelfCheckGPT, adopt **triple-reasoning-path consistency** tailored cho KG-RAG:

```
1. Run pipeline N=3 times with different PrimeKG subgraph seeds 
   (permute top-k neighbors at each expansion step, retain top-50 by PPR).
2. Each run produces a TRM-ranked path set P_i.
3. Compute pairwise path-Jaccard: J(P_i, P_j) = |P_i ∩ P_j| / |P_i ∪ P_j|.
4. Consistency score = mean J over all pairs.
5. If consistency < 0.5 → flag "unstable reasoning" → abstain or revise.
```

Cost: 3× retrieval latency only (TRM + Judge không cần re-run nếu P is identical). On 2×T4 this adds ~5s per question, acceptable for eval.

## 1.4 Trace-based hallucination reduction

### 1.4.1 Provenance annotation enforcement

Hard requirement: **mọi sentence trong final answer phải có ≥1 provenance tag** `[PMID:X]` hoặc `[edge:Y]`. Enforced at two points:

1. **Synthesizer prompt** (Layer 5): system prompt mandates `[PMID:X]` or `[edge:Y]` tags after each sentence. Examples in few-shot.
2. **Judge 2** (Evidence Grounding Inspector): parses tags, verifies each PMID exists in retrieved passages và each edge_id exists in TRM-ranked paths. Untagged sentences → auto-flag as unsupported.

```python
def enforce_provenance(answer: str, evidence: EvidenceBundle) -> tuple[bool, list[str]]:
    sentences = sent_tokenize(answer)
    issues = []
    for s in sentences:
        tags = re.findall(r'\[(?:PMID:\d+|edge:[A-Z0-9_]+)\]', s)
        if not tags:
            issues.append(f"Missing provenance: {s}")
            continue
        for tag in tags:
            if tag.startswith('[PMID:') and tag[6:-1] not in evidence.pmids:
                issues.append(f"Invalid PMID in: {s}")
            elif tag.startswith('[edge:') and tag[6:-1] not in evidence.edge_ids:
                issues.append(f"Invalid edge_id in: {s}")
    return (len(issues) == 0, issues)
```

### 1.4.2 Confidence thresholding — multiplicative, not additive

Adopt **AMG-RAG multiplicative scheme** (Rezaei 2025, arXiv 2502.13010): edge confidences multiply along reasoning path. Rationale — additive averages hide low-confidence hops (one unreliable edge trong 5-edge path vẫn cho avg ≈ 0.8), while multiplicative aggressively penalizes weak links (a 0.3 edge drops path from 0.9³ = 0.73 to 0.22). In medical domain, một weak link = reasoning chain broken.

```
For path p = e₁ → e₂ → … → eₖ with edge confidences c_i ∈ [0,1]:
    conf(p) = ∏ᵢ cᵢ                                       (multiplicative)
```

Edge confidences `c_i` compose hai sources:
- PrimeKG intrinsic source reliability (SIDER=0.7, DrugBank=0.95, etc.) — static lookup table.
- TRM-assigned dynamic weight from recursive reasoning trace (§2.3).

**Abstention policy**:
- `conf(p_best) ≥ 0.7` → emit answer with full confidence tag.
- `0.4 ≤ conf(p_best) < 0.7` → emit with explicit uncertainty phrasing ("Available evidence suggests...").
- `conf(p_best) < 0.4` → **abstain**, respond "Evidence insufficient; consult clinical resources."

### 1.4.3 Abstention — the critical trade-off

Trong medical domain, **false confidence > no answer**. The MedHallu paper (Pandit 2025) empirically shows adding explicit "not sure" option improves precision by **up to 38%** on hard hallucination tier. v3 exposes three abstention triggers:

1. `conf(p_best) < 0.4` (KG-based).
2. Judge 1–4 consensus `overall_recommendation="ABSTAIN"` (any judge flagging H5 triggers immediate abstain).
3. Triple-reasoning-path consistency < 0.5 (§1.3.5).

Abstention output template: `"Based on available evidence from {N} PubMed sources and {M} PrimeKG relations, I cannot reach a confident answer. Key uncertainty: {specific_gap}. Please consult {recommended_resource}."`

### 1.4.4 Hard constraints cho life-critical domains

Three deterministic checks run **before** LLM synthesis:

- **Dosage range check**: for each drug mentioned with a numeric dose, compare against PrimeKG drug record's therapeutic range; reject if outside [min, max] × 1.5 safety margin.
- **Contraindication check**: for each (drug_A, drug_B) pair in question/answer, query PrimeKG `drug_drug` relation với `display_relation ∈ {contraindication}`; if match → force abstain.
- **Population-specific check**: extract patient attributes (age, pregnancy, renal function) via NER; filter PrimeKG evidence to matching subpopulation via UMLS semantic types.

## 1.5 Full evaluation protocol — per dataset

| Dataset | N (test) | Format | Accuracy metric | Hallucination metric | Special handling |
|---|---|---|---|---|---|
| MedQA-USMLE | 1,273 | 4-option MCQ | exact-match acc | F1_hall on CoT claims vs retrieved evidence | Note 7.4% items flagged clinically unfit per re-annotation |
| MedMCQA | 4,183 (dev, labels public) | 4-option MCQ | exact-match acc | F1_hall + RCS vs MedReason-style gold path (when available) | Official test has hidden labels — use dev |
| PubMedQA (PQA-L) | 500 (official test) | Yes/No/Maybe | macro-F1 | NLI entailment explanation→`long_answer` + numeric consistency | Explanation length ~100 words |
| BioASQ-13b | ~500 test per batch | factoid/yes-no/list/ideal | per-type (F1, MRR, ROUGE-L) | Claim-level F1 + citation precision | Use gold snippets as reference context |
| MMLU-med (6 subsets) | 1,089 | 4-option MCQ | exact-match acc | F1_hall on CoT | MMLU has ~6.5% erroneous items — cap reported ceiling |
| MedBullets | 308 | 4 or 5-opt MCQ | exact-match acc | F1_hall + RCS against gold explanation | Smaller N — pool with MedQA for stats power |
| Med-HALT | 7 tasks × N | varies | pointwise score per paper | — (benchmark is itself a hallucination measure) | Sample 50/task = 350 for fast iteration |
| MedHallu | 10,000 (1K labeled + 9K artificial) | binary detection | F1 (easy/med/hard) | — (pipeline plays detector role) | Report three difficulty tiers separately |
| MEDEC-MS | 597 test | error detection + correction | flag accuracy, sentence accuracy, correction ROUGE | — | Use MS split (public); UW requires DUA |
| K-QA | 201 | open-ended | claim-level must-have recall | precision on claims | Long-form, expert-annotated |

**Sample size consideration**: to detect Δ=3% improvement with 80% power at α=0.05, McNemar-paired needs ~1,600 items. MedQA (1,273) and MMLU-med (1,089) individually underpowered → **pool via weighted bootstrap** across MCQ benchmarks for primary headline comparison. BioASQ (~500) only reliably detects Δ≥6%.

## 1.6 Worked example — full hallucination detection on one MedQA-style question

**Question**: *"A 68-year-old male with atrial fibrillation and newly diagnosed peptic ulcer disease is currently on warfarin. Which of the following antibiotics for H. pylori eradication poses the highest bleeding risk? (A) Amoxicillin (B) Clarithromycin (C) Metronidazole (D) Doxycycline"*

**Candidate answer from synthesizer**: *"The answer is (C) Metronidazole. Metronidazole strongly inhibits CYP2C9, increasing warfarin levels 3-fold and causing severe bleeding at 100 mg doses [PMID:99999999]. Clarithromycin is safe with warfarin [edge:E_FAKE]."*

**Hallucination detection trace**:

1. Claim 1: *"Answer is (C) Metronidazole"* → verify against retrieved PrimeKG (warfarin, drug_drug, metronidazole, edge_id=E_22104) with `display_relation=drug_drug` flagged as major DDI → **supported**.
2. Claim 2: *"Metronidazole strongly inhibits CYP2C9"* → PrimeKG (metronidazole, drug_protein, CYP2C9, edge_id=E_08812) confirms → **supported**.
3. Claim 3: *"…increasing warfarin levels 3-fold"* → PMID 99999999 → E-utilities `efetch` returns 404 → **H4 source attribution hallucination**. Severity 2.
4. Claim 4: *"…severe bleeding at 100 mg doses"* → numeric check: metronidazole therapeutic 250–500 mg; 100 mg is sub-therapeutic, doesn't even cause interaction at that dose → **H5 life-critical** (contradicts dose-dependent effect). Severity 3.
5. Claim 5: *"Clarithromycin is safe with warfarin"* → PrimeKG lookup (clarithromycin, drug_drug, warfarin) returns edge_id=E_11508 `drug_drug` = major DDI (CYP3A4 inhibition) → **H1 factual contradiction**. Severity 3. Also edge `E_FAKE` not in evidence → **H4**. Severity 3.

**Judge output**: `overall_recommendation="ABSTAIN"`, `h5_present=true`, `faithfulness_score=0.4`. Pipeline returns abstention response, logs for retraining.

---

# Part 2 — TRM Integration Specification (PRIORITY 2)

## 2.1 Samsung TRM — verified architecture

**Citation**: Alexia Jolicoeur-Martineau, *"Less is More: Recursive Reasoning with Tiny Networks"*, arXiv:2510.04871, Samsung SAIL Montréal, Oct 6, 2025. **Official repo**: `github.com/SamsungSAILMontreal/TinyRecursiveModels` (MIT License). **Author blog**: `alexiajm.github.io/2025/09/29/tiny_recursive_models.html`.

**Terminology correction**: the paper calls TRM a **"recursive reasoning model"**, never a "geometric reasoner" — "geometric" only appears describing ARC-AGI puzzles themselves. Mọi downstream documentation nên drop "geometric" label.

### 2.1.1 Architecture (verbatim from `config/arch/trm.yaml`)

- Single tiny network, **2 transformer layers**, `hidden_size=512`, `num_heads=8` (head_dim=64), SwiGLU expansion factor 4, RMSNorm post-norm, **no biases**, **non-causal multi-head self-attention** (standard MHA, no GQA).
- Position encoding: RoPE for ARC/Maze; disabled (MLP-Mixer over sequence instead) for Sudoku.
- `puzzle_emb_len=16` — 16 learned puzzle-ID tokens prepended to every sequence (critical cho task conditioning).
- Forward dtype: **bfloat16** (problem on T4 which only supports FP16 — **override to float16** in TRM config cho Kaggle).
- Parameter count: **~7M with self-attention variant** (ARC-AGI); 5M MLP-mixer variant (Sudoku); 19M MLP variant cho Maze.

### 2.1.2 Input/output tensor format

**Dataset files on disk**:
```
all__inputs.npy           # [N, seq_len] int64
all__labels.npy           # [N, seq_len] int64
all__puzzle_identifiers.npy  # [N] int32
dataset.json              # {"vocab_size": V, "seq_len": L}
```

**Batch dict to `model.forward`**:
```python
batch = {
    "inputs":              torch.LongTensor  [B, L],   # token IDs
    "labels":              torch.LongTensor  [B, L],
    "puzzle_identifiers":  torch.LongTensor  [B],
}
```

**Internal effective shape**: `[B, L + 16, 512]` (concat learned puzzle embedding).

**Output**: `lm_head(z_H)[:, puzzle_emb_len:]` → `[B, L, vocab_size]` logits; `q_head(z_H[:,0])` → `[B, 2]` halt/continue Q-logits (float32).

### 2.1.3 Recursive mechanism (verbatim, paper Fig. 3)

```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):
        z = net(x + y + z)       # update latent reasoning n times
    y = net(y + z)               # update answer (note: NO x here)
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)
    y, z = latent_recursion(x, y, z, n)   # final pass with gradients
    return (y.detach(), z.detach()), lm_head(y), q_head(y)

# Outer training loop
for x_input, y_true in loader:
    y, z = y_init, z_init
    for step in range(N_sup=16):          # ACT supervision steps
        x = input_embed(x_input)
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)
        loss = stablemax_ce(y_hat, y_true) + bce(q_hat, (y_hat == y_true))
        loss.backward(); opt.step()
        if q_hat > 0: break                # early halt during training
```

- **y** = current *embedded answer* (HRM's z_H).
- **z** = current *latent reasoning feature* (HRM's z_L), acts like CoT.
- Default `T=3, n=6` (Sudoku), `T=3, n=4` (ARC/Maze for memory).
- `halt_max_steps=16`, `halt_exploration_prob=0.1`.
- At inference, no early halt — all 16 supervision steps run.

### 2.1.4 Key training details

Datasets (1000× data augmentation typical):
- **Sudoku-Extreme**: 1K train / 423K test, 9×9 grid, vocab 11.
- **Maze-Hard**: 30×30, 1K+1K, vocab ~6.
- **ARC-AGI-1/2**: 800/1120 tasks + ConceptARC, vocab 12.

Loss: `stablemax_cross_entropy` + BCE halt. Optimizer: AdamW với `adam-atan2` variant, `lr=1e-4`, `weight_decay=1.0`, cosine schedule, EMA enabled.

Reported headline: **44.6% ARC-AGI-1 / 7.8% ARC-AGI-2 với 7M params**, beating Deepseek R1 (671B), o3-mini-high, Gemini 2.5 Pro. On Sudoku-Extreme: 87.4% (MLP-mixer variant).

Compute: Sudoku ~18h on 1×L40S (48GB); ARC ~3 days on 4×H100.

## 2.2 MedicalGraphEmbedder — adapter design

**Role**: convert `{PrimeKG subgraph, PubMed passages, UMLS CUIs}` → TRM-compatible tensor `{inputs [B,L], labels [B,L], puzzle_identifiers [B]}`. This is the critical bridge — TRM expects discrete tokens, we have a heterogeneous graph + free text.

### 2.2.1 Architectural option comparison

| Option | Description | Pros | Cons | Verdict for v3 |
|---|---|---|---|---|
| **A. Pure lookup table** (SapBERT embeddings → dimensionality-reduce → discretize via VQ-VAE codebook) | Precompute SapBERT 768-d cho mọi PrimeKG node, quantize to 512-d TRM tokens via learned codebook of size V=4096 | Simple, cacheable, fast inference | Loses edge-type info; static per node | Reject — 30 edge types are critical signal |
| **B. GNN encoder** (HGT on PrimeKG subgraph → pool → tokenize) | 2-layer HGTConv with `hidden_dim=256`, `heads=4`, output via mean-pool per node then VQ-discretize | Captures heterogeneous edge types; inductive over subgraphs | Extra params (~3M); needs training | **Primary recommendation** |
| **C. Transformer graph encoder** (serialize triples as text → encode with PubMedBERT → use CLS per triple as token) | No custom GNN; leverages PubMedBERT | Full sentence context; off-the-shelf | Slow, loses explicit graph structure | Fallback only |
| **D. Hybrid** (GNN for structure + SapBERT for text, fused via concat + linear) | HGT output concatenated with SapBERT CLS of PubMed passage, fused to 512-d | Best of both | Most complex; more params | Consider for v3.1 |

**Recommended: Option B (HGT-based) for v3 baseline**, với Option D upgrade path. Rationale: PrimeKG's 30 edge types carry semantic signal (contraindication vs indication vs off-label_use must be distinguished); HGT's per-(src_type, edge_type, dst_type) Q/K/V projections exactly model this. Parameter footprint stays modest (~3M HGT params + 2M VQ codebook = 5M addl), giving total MedicalGraphEmbedder + TRM ≈ 12M — still fits comfortably on a single T4.

### 2.2.2 Tensor shape contract

Output shape must match TRM input: **`[B, L, ]` with `inputs` dtype `int64`, `puzzle_identifiers` dtype `int32`**. We choose **L=256** để accommodate typical 2-hop PrimeKG subgraph (~100 nodes + ~150 edges = ~250 tokens after serialization), `vocab_size=8192` (4K graph codebook + 2K PubMed codebook + 1K reserved + 1K special tokens). Puzzle identifier encodes question type (differential diagnosis / drug interaction / dosage / etiology / other = 5 types, extensible to 32).

### 2.2.3 Node feature encoding

For each PrimeKG node:
```
node_feat(v) = concat(
    SapBERT(v.x_name),                       # 768-d text embedding of name
    one_hot(v.x_type, num_types=10),         # node type: disease/drug/protein/…
    deg_features(v)                          # log-degree, PPR score
) → Linear(784 → 256)                        # HGT hidden_dim
```

UMLS CUI integration: when `v.x_source == "UMLS"`, use UMLS canonical name; otherwise use PrimeKG `x_name` directly. SapBERT handles synonym collapsing.

### 2.2.4 Edge feature encoding

For each PrimeKG edge:
```
edge_feat(e) = concat(
    embedding(e.relation, num_relations=30),  # learned 64-d per type
    source_reliability(e.x_source, e.y_source),  # static {DrugBank:0.95, SIDER:0.7,…}
    amg_confidence(e)                            # LLM-assigned 1–10 normalized (AMG-RAG)
) → Linear → 64-d edge feature
```

HGT uses `edge_type` ID to select per-relation projection; edge features modulate attention weights.

### 2.2.5 Multi-source evidence fusion

PubMed passages get their own processing branch:
```
For each top-k PubMed passage p_i:
    emb_i = MedCPT-Article-Encoder(p_i.title + p_i.abstract)   # 768-d
    kg_bridge_i = soft_attend(emb_i, {node_feat(v) for v in subgraph_nodes})
        # cross-attention: passage → graph nodes, gives weighted node alignment
    fused_i = Linear(concat(emb_i, kg_bridge_i))  → 256-d
```

The fused passage embeddings become additional "virtual nodes" in the subgraph before HGT pooling, connected via learned edges to the top-3 most similar real nodes (by SapBERT cosine).

### 2.2.6 Tokenization (graph → sequence)

After HGT encodes subgraph + passages:

```python
def graph_to_tokens(hgt_output, max_len=256, codebook_size=4096):
    # hgt_output: Dict[node_id, 256-d vector], len ~100-200 nodes+virtual
    # Sort nodes by PPR-relevance to question entities, descending
    sorted_nodes = sorted(hgt_output, key=lambda n: n.ppr_score, reverse=True)[:max_len]
    # Vector-quantize each 256-d vector to nearest codebook entry (pretrained VQ-VAE)
    tokens = [vq_codebook.nearest_id(n.embedding) for n in sorted_nodes]
    # Pad with PAD_ID to max_len
    tokens += [PAD_ID] * (max_len - len(tokens))
    return torch.LongTensor(tokens)
```

VQ codebook pretrained separately on 50K random PrimeKG subgraphs (self-supervised reconstruction, 1 hour on T4).

### 2.2.7 Full embedder pseudocode

```python
class MedicalGraphEmbedder(nn.Module):
    def __init__(self, primekg, sapbert, medcpt_article, n_relations=30, n_types=10,
                 hidden=256, codebook_size=4096, max_len=256):
        super().__init__()
        self.node_proj = nn.Linear(768 + n_types + 2, hidden)
        self.edge_emb  = nn.Embedding(n_relations, 64)
        self.edge_proj = nn.Linear(64 + 2, 64)
        self.hgt = HGTConv(hidden, hidden, metadata=primekg.metadata,
                           heads=4, num_layers=2)
        self.passage_proj = nn.Linear(768, hidden)
        self.bridge_attn  = nn.MultiheadAttention(hidden, 4)
        self.vq = VectorQuantize(dim=hidden, codebook_size=codebook_size)
        self.max_len = max_len
        self.sapbert = sapbert        # frozen
        self.medcpt  = medcpt_article # frozen
    
    def forward(self, subgraph: HeteroData, pubmed_passages: list[str],
                question_entities: list[str]) -> dict:
        # 1. Node features
        node_texts = [subgraph[t].x_name for t in subgraph.node_types]
        with torch.no_grad():
            node_emb = self.sapbert(node_texts)  # [N, 768]
        node_feat = self.node_proj(torch.cat([
            node_emb, 
            one_hot(subgraph.node_type, 10),
            degree_features(subgraph)], dim=-1))
        
        # 2. HGT pass
        hgt_out = self.hgt(node_feat, subgraph.edge_index_dict, subgraph.edge_attr_dict)
        
        # 3. PubMed passage fusion
        with torch.no_grad():
            passage_emb = self.medcpt(pubmed_passages)  # [K, 768]
        passage_feat = self.passage_proj(passage_emb)
        bridged, _ = self.bridge_attn(passage_feat, hgt_out, hgt_out)
        
        # 4. Combine nodes + virtual passages
        all_feats = torch.cat([hgt_out, bridged], dim=0)  # [N+K, hidden]
        
        # 5. Rank by PPR, take top max_len
        ppr_scores = personalized_pagerank(subgraph, seeds=question_entities)
        top_idx = ppr_scores.topk(min(self.max_len, len(all_feats))).indices
        selected = all_feats[top_idx]
        
        # 6. Vector quantize → token IDs
        _, token_ids, _ = self.vq(selected)  # [L]
        
        # 7. Pad to max_len
        token_ids = F.pad(token_ids, (0, self.max_len - len(token_ids)),
                          value=PAD_ID)
        
        # 8. Classify question type → puzzle_id
        puzzle_id = classify_question_type(question_entities)
        
        return {
            "inputs": token_ids.long().unsqueeze(0),          # [1, 256]
            "puzzle_identifiers": torch.tensor([puzzle_id]),  # [1]
        }
```

## 2.3 TRM Recursive Loop for Medical Reasoning

Keep TRM's core recursion intact; only customize initialization, halting, and output interpretation cho medical context.

```python
def medical_trm_reason(embedder_output, trm_model, 
                       max_steps=16, conf_threshold=0.85,
                       contradiction_threshold=0.5):
    """
    Returns:
        ranked_paths: list of (path_token_sequence, confidence)
        validity_score: float ∈ [0,1]
        contradiction_flag: bool
        trace: list of per-step states for provenance
    """
    inputs   = embedder_output["inputs"]
    puzzle   = embedder_output["puzzle_identifiers"]
    x        = trm_model.input_embed(inputs, puzzle)        # [1, 16+L, 512]
    y        = trm_model.y_init.expand_as(x)
    z        = trm_model.z_init.expand_as(x)
    
    trace = []
    prev_y = None
    
    for step in range(max_steps):
        # Per-step deep recursion (T=3, n=4 for medical — ARC-like seq_len 256)
        (y, z), y_hat_logits, q_logits = trm_model.deep_recursion(x, y, z, n=4, T=3)
        halt_prob = torch.sigmoid(q_logits[0, 0])
        
        # Halt condition 1: path stability (y changes little)
        if prev_y is not None:
            delta = (y - prev_y).norm() / y.norm()
            stable = delta < 1e-3
        else:
            stable = False
        prev_y = y.detach()
        
        # Halt condition 2: confidence
        conf = top1_confidence(y_hat_logits)               # max softmax over top path
        
        # Halt condition 3: contradiction detection
        contradiction = detect_contradiction(y_hat_logits, x)
        
        trace.append({"step": step, "halt_prob": halt_prob.item(),
                      "conf": conf.item(), "stable": stable,
                      "contradiction": contradiction})
        
        # Early exit
        if halt_prob > 0.5 or stable or conf > conf_threshold:
            break
        if contradiction > contradiction_threshold:
            break
    
    # Decode top-k paths from final y_hat_logits
    ranked_paths = decode_paths(y_hat_logits, top_k=5,
                                 codebook=embedder.vq.codebook,
                                 primekg=embedder.primekg)
    
    # Multiplicative confidence aggregation per path
    for p in ranked_paths:
        p.confidence = prod(edge.weight for edge in p.edges)  # AMG-RAG style
    
    validity_score = ranked_paths[0].confidence
    contradiction_flag = contradiction > contradiction_threshold
    
    return ranked_paths, validity_score, contradiction_flag, trace
```

### 2.3.1 Halting conditions (medical-specific)

Four triggers (OR-composed):
1. **ACT halt probability > 0.5** (TRM native).
2. **Path stability**: `‖y_t − y_{t-1}‖ / ‖y_t‖ < 1e-3` for 2 consecutive steps.
3. **Confidence threshold**: top-1 path confidence > 0.85.
4. **Contradiction detected** (below).

### 2.3.2 Contradiction detection between PrimeKG and PubMed

```python
def detect_contradiction(y_hat_logits, x) -> float:
    # Extract top-5 tokens from y_hat and map back to KG triples/passages
    top_tokens = y_hat_logits.topk(5, dim=-1).indices  # [L, 5]
    triples   = [codebook_to_triple(t) for t in top_tokens if is_kg_token(t)]
    passages  = [codebook_to_passage(t) for t in top_tokens if is_pubmed_token(t)]
    
    max_contra = 0.0
    for tri in triples:
        for pas in passages:
            # NLI: does passage contradict triple?
            score = nli_model(premise=pas.text, hypothesis=triple_to_nl(tri))
            if score['label'] == 'CONTRADICTION':
                max_contra = max(max_contra, score['score'])
    return max_contra
```

Threshold `contradiction > 0.5` → emit contradiction flag to Judge 3.

### 2.3.3 Confidence propagation: multiplicative > additive

Adopt multiplicative (AMG-RAG, arXiv 2502.13010). Additive (mean) hides weak links — in medical reasoning, **mọi weak link breaks chain**. Test: a 5-edge path với confidences `[0.9, 0.9, 0.3, 0.9, 0.9]`:
- Additive mean: 0.78 (looks acceptable)
- Multiplicative: 0.177 (correctly flags)

The 0.3 edge (e.g., low-reliability source) should dominate uncertainty, và multiplicative correctly does this.

### 2.3.4 Attention mechanism — TRM's native recursion suffices

TRM already has self-attention trong its 2 transformer layers. Adding external attention over subgraph would double parameters without clear gain. The recursive update `z ← net(x + y + z)` repeated n=4 times, T=3 times, 16 supervision steps gives effective depth `4 × 3 × 2 × 16 = 384` layers of reasoning — ample for medical chains typically ≤ 5 hops.

## 2.4 TRM Output Decoder

TRM emits `[1, 256, 8192]` logits. Decode:

```python
def decode_paths(logits, top_k=5, codebook, primekg):
    # 1. Decode each position to top-k tokens
    token_ids = logits.topk(top_k, dim=-1).indices  # [L, K]
    
    # 2. Map token IDs back to graph nodes/edges via codebook
    nodes_per_pos = [[codebook.id_to_node(t) for t in token_ids[i]] 
                     for i in range(len(token_ids))]
    
    # 3. Reconstruct paths as connected chains through PrimeKG
    paths = []
    for k in range(top_k):
        chain = [nodes_per_pos[0][k]]
        for i in range(1, len(nodes_per_pos)):
            # Find which candidate at position i is connected via an edge
            candidates = nodes_per_pos[i]
            edges = [(c, primekg.edge_between(chain[-1], c)) for c in candidates]
            edges = [e for e in edges if e[1] is not None]
            if not edges: break
            next_node, edge = max(edges, key=lambda e: e[1].confidence)
            chain.append((next_node, edge))
        paths.append(Path(chain))
    
    return paths
```

Output data structures:

```python
@dataclass
class Edge:
    edge_id: str                   # PrimeKG unique edge ID
    relation: str                  # one of 30 relation types
    source_reliability: float      # static lookup
    trm_confidence: float          # from this reasoning run
    
@dataclass
class Path:
    nodes: list[Node]
    edges: list[Edge]
    confidence: float              # multiplicative product
    supporting_pmids: list[str]    # virtual-node PubMed PMIDs touching this path
    
@dataclass
class TRMOutput:
    ranked_paths: list[Path]       # top-5 by confidence
    validity_score: float          # = ranked_paths[0].confidence
    contradiction_flag: bool
    trace: list[dict]              # per-step diagnostic info
```

### 2.4.1 Decode back to text for LLM consumption

```python
def path_to_text(path: Path) -> str:
    lines = []
    for i, (node, edge) in enumerate(zip(path.nodes[1:], path.edges)):
        prev = path.nodes[i]
        lines.append(
            f"Step {i+1}: {prev.x_name} --[{edge.relation}, "
            f"edge_id={edge.edge_id}, confidence={edge.trm_confidence:.2f}]--> "
            f"{node.x_name}"
        )
    if path.supporting_pmids:
        lines.append(f"Supporting literature: {', '.join(path.supporting_pmids)}")
    lines.append(f"Overall path confidence: {path.confidence:.2f}")
    return "\n".join(lines)
```

### 2.4.2 Transparent trace generation

Every TRM run emits a provenance log:
```json
{
  "question_id": "MedQA_12345",
  "subgraph_stats": {"nodes": 143, "edges": 287, "max_hop": 2},
  "pubmed_retrieved": ["PMID:28472901", "PMID:31234567", ...],
  "trm_steps": [
    {"step": 0, "halt_prob": 0.12, "conf": 0.41, "contradiction": 0.02},
    ...
    {"step": 7, "halt_prob": 0.61, "conf": 0.87, "contradiction": 0.05, "halted": true}
  ],
  "ranked_paths": [
    {"confidence": 0.82, "edge_ids": ["E_22104", "E_08812"], 
     "pmids": ["PMID:28472901"]},
    ...
  ],
  "final_validity": 0.82,
  "contradiction_flag": false
}
```

This trace becomes provenance payload cho Judge 4 và final answer citation.

## 2.5 Translation Layer 2 — JSON schema + worked example

### 2.5.1 JSON schema cho MedReason-style CoT → TRM input

```python
class QuestionEntity(BaseModel):
    surface: str           # "metronidazole"
    cui: str | None        # "C0025872"
    primekg_node_id: str | None  # "drug:DB00916"

class KGEdge(BaseModel):
    edge_id: str
    head: str              # PrimeKG node ID
    tail: str
    relation: str          # one of 30 types
    display_relation: str
    source_reliability: float
    amg_confidence: float = 1.0  # LLM-assigned, default 1.0

class PubMedPassage(BaseModel):
    pmid: str
    title: str
    abstract: str
    relevance_score: float  # MedCPT cross-encoder

class CoTStep(BaseModel):
    step_id: int
    premise_edge_ids: list[str]
    claim_text: str
    supporting_pmids: list[str]

class TRMInputBundle(BaseModel):
    question_id: str
    question_text: str
    question_type: Literal["diagnosis","drug_interaction","dosage","etiology","other"]
    question_entities: list[QuestionEntity]
    subgraph_edges: list[KGEdge]           # 2-hop BFS pruned via PPR
    pubmed_passages: list[PubMedPassage]   # top-5 via MedCPT cross-enc rerank
    medreason_cot: list[CoTStep] | None    # optional seed CoT
    gold_answer: str | None                # for training
```

### 2.5.2 Full worked example (drug-drug interaction, MedQA-style)

**Question** (from §1.6): *"A 68-year-old male with atrial fibrillation and newly diagnosed peptic ulcer disease on warfarin. Which antibiotic for H. pylori poses highest bleeding risk? (A) Amoxicillin (B) Clarithromycin (C) Metronidazole (D) Doxycycline"*

**Stage 1 — entity extraction (ScispaCy)**:
```json
[
  {"surface": "atrial fibrillation", "cui": "C0004238", "primekg_node_id": "disease:MONDO:0004981"},
  {"surface": "warfarin", "cui": "C0043031", "primekg_node_id": "drug:DB00682"},
  {"surface": "H. pylori", "cui": "C0079488", "primekg_node_id": "anatomy:bacteria..."},
  {"surface": "amoxicillin", "cui": "C0002645", "primekg_node_id": "drug:DB01060"},
  {"surface": "clarithromycin", "cui": "C0055856", "primekg_node_id": "drug:DB01211"},
  {"surface": "metronidazole", "cui": "C0025872", "primekg_node_id": "drug:DB00916"},
  {"surface": "doxycycline", "cui": "C0013090", "primekg_node_id": "drug:DB00254"}
]
```

**Stage 2 — PrimeKG 2-hop subgraph** (pruned via PPR, top-50 edges, relation filter `{drug_drug, drug_protein, contraindication}`):
```json
[
  {"edge_id":"E_22104","head":"drug:DB00682","tail":"drug:DB00916",
   "relation":"drug_drug","display_relation":"drug_drug",
   "source_reliability":0.95,"amg_confidence":1.0},
  {"edge_id":"E_11508","head":"drug:DB00682","tail":"drug:DB01211",
   "relation":"drug_drug","display_relation":"drug_drug",
   "source_reliability":0.95,"amg_confidence":1.0},
  {"edge_id":"E_08812","head":"drug:DB00916","tail":"protein:P11712",
   "relation":"drug_protein","display_relation":"target",
   "source_reliability":0.90,"amg_confidence":1.0},
  {"edge_id":"E_08845","head":"drug:DB01211","tail":"protein:P08684",
   "relation":"drug_protein","display_relation":"target",
   "source_reliability":0.90,"amg_confidence":1.0},
  ...
]
```

**Stage 3 — PubMed top-5 (MedCPT)**:
```json
[
  {"pmid":"28472901","title":"Drug interactions with warfarin...",
   "abstract":"Metronidazole and clarithromycin both strongly potentiate warfarin via CYP inhibition...",
   "relevance_score":0.92},
  {"pmid":"30912234","title":"CYP3A4 inhibitors and anticoagulation",
   "abstract":"Clarithromycin, a potent CYP3A4 inhibitor, can elevate warfarin INR by 2-3 fold...",
   "relevance_score":0.88},
  ...
]
```

**Stage 4 — MedReason-style CoT seed**:
```json
[
  {"step_id":1,"premise_edge_ids":["E_22104"],
   "claim_text":"Warfarin and metronidazole have documented drug-drug interaction",
   "supporting_pmids":["PMID:28472901"]},
  {"step_id":2,"premise_edge_ids":["E_08812"],
   "claim_text":"Metronidazole inhibits CYP2C9 (P11712), the primary warfarin-metabolizing enzyme",
   "supporting_pmids":["PMID:28472901"]},
  {"step_id":3,"premise_edge_ids":["E_11508","E_08845"],
   "claim_text":"Clarithromycin targets CYP3A4 (P08684), a secondary warfarin pathway",
   "supporting_pmids":["PMID:30912234"]},
  {"step_id":4,"premise_edge_ids":["E_22104","E_08812"],
   "claim_text":"CYP2C9 inhibition has stronger effect on warfarin than CYP3A4 → Metronidazole highest risk",
   "supporting_pmids":["PMID:28472901","PMID:30912234"]}
]
```

**Stage 5 — MedicalGraphEmbedder output**:
```python
inputs             = torch.LongTensor([[2, 1847, 3921, ..., 0, 0]])  # shape [1, 256]
puzzle_identifiers = torch.LongTensor([1])                           # 1 = drug_interaction
# tokens 2, 1847, 3921 correspond to quantized HGT embeddings of 
# warfarin-metronidazole-edge, metronidazole-CYP2C9-edge, etc., 
# sorted by PPR relevance.
```

**Stage 6 — TRM runs recursion**, halts at step 9 với `conf=0.87`, emits top-5 paths. Path 1:
```
warfarin --[drug_drug, E_22104, conf=0.93]--> metronidazole 
         --[drug_protein:target, E_08812, conf=0.91]--> CYP2C9
Supporting literature: PMID:28472901
Overall confidence: 0.846 (multiplicative)
```

**Stage 7 — decoder emits text trace to Layer 5 synthesizer** along với `validity_score=0.85`, `contradiction_flag=false`.

**Stage 8 — synthesizer answer**: *"The answer is (C) Metronidazole [edge:E_22104]. Metronidazole interacts with warfarin via CYP2C9 inhibition [edge:E_08812, PMID:28472901], which is warfarin's primary metabolic pathway. Although clarithromycin also interacts with warfarin via CYP3A4 [edge:E_11508, PMID:30912234], the CYP2C9 pathway is more clinically significant for bleeding risk."*

## 2.6 TRM training/fine-tuning

### 2.6.1 Fine-tuning is required

Pretrained TRM weights (Sudoku/Maze/ARC) do **not** transfer zero-shot to medical graph reasoning — vocabulary, task, and sequence semantics all differ. Fine-tuning from the pretrained Sudoku-Extreme checkpoint initialization is recommended (transformer attention priors help) but a from-scratch run is also viable given TRM's small size.

### 2.6.2 Training data format

Generate synthetic `(GraphTensor, VerifiedPath)` pairs from MedReason:

```python
for qa_pair in medreason_dataset:   # 32,682 pairs total
    # Extract gold KG path from MedReason CoT
    gold_path = parse_reasoning_chain(qa_pair.reasoning)  # returns list of edge_ids
    
    # Build input bundle
    bundle = build_trm_input_bundle(
        question=qa_pair.question,
        question_entities=scispacy_link(qa_pair.question),
        primekg_subgraph=extract_2hop_subgraph(entities, primekg),
        pubmed_passages=medcpt_retrieve(qa_pair.question, k=5)
    )
    
    # Tokenize via MedicalGraphEmbedder
    trm_input = embedder(bundle)
    
    # Build TRM labels: sequence of path tokens that gold_path maps to, padded
    trm_label = path_to_token_sequence(gold_path, embedder.vq.codebook)
    
    yield (trm_input, trm_label, question_type_to_puzzle_id(qa_pair))
```

Split: 29K train / 2K val / 1.5K held-out. MedReason already filters for clinical correctness via GPT-4o + physician review, making it high-quality training signal.

### 2.6.3 Loss function

Standard TRM loss:
```
L = stablemax_cross_entropy(y_hat_logits, gold_token_sequence) 
  + α · BCE(q_halt, (y_hat == gold))
  + β · reconstruction_loss_vq                 # train codebook jointly
```
With `α=0.1, β=0.05`.

### 2.6.4 Kaggle 2×T4 feasibility

| Component | Params | Training VRAM at batch=8, seq=256 | Feasible? |
|---|---|---|---|
| TRM core (7M) | 7M | ~3 GB | ✅ |
| HGT encoder (3M) | 3M | ~4 GB (includes subgraph storage) | ✅ |
| VQ codebook (2M) | 2M | ~1 GB | ✅ |
| SapBERT frozen | 110M | 1 GB (inference only) | ✅ |
| MedCPT frozen | 110M | 1 GB (inference only) | ✅ |
| **Total single T4** | ~12M trainable | **~10 GB** | ✅ tight |

**Training recipe**:
- `torch.cuda.amp.autocast(dtype=torch.float16)` (not bf16 — T4 limitation).
- **Gradient checkpointing** enabled on HGT and TRM (reduces activation memory ~40%).
- Batch size 8, gradient accumulation 4 → effective 32.
- `N_sup=8` during training (reduced from 16 for memory), restore to 16 at inference.
- Learning rate 1e-4 cosine, EMA 0.999, weight decay 1.0.
- ~30K steps at 29K training examples ≈ 4 epochs. Wall clock: **~7 hours on single T4** → fits in 9-hour Kaggle session with 2h headroom. Leave GPU 1 free for eval during training.
- Checkpoint every 2500 steps to `/kaggle/working` + push to HF private repo as backup against session timeout.

**Watch-outs**: T4 doesn't support FlashAttention-2, so TRM attention uses xformers fallback — ~30% slower than H100. If session hits 9hr limit, resume from last HF checkpoint trong new session.

---

# Part 3 — Supporting Layer Specifications (lighter depth)

## 3.1 Layer 1 — Agentic Retrieval

**Entity linking recommendation**: **ScispaCy `en_core_sci_lg` + built-in UMLS linker** làm primary (zero setup, no license gate, pip-installable, ~1 GB UMLS KB cached as Kaggle dataset). Hybrid với SapBERT-FAISS fallback cho long-tail terms where ScispaCy returns low-confidence. **Skip MetaMap, QuickUMLS, MedCAT-MIMIC** (all require UMLS license and/or Java — infeasible on Kaggle without sudo).

**PrimeKG subgraph extraction**: **2-hop with PPR pruning to top-200 nodes** is the sweet spot. 1-hop often misses second-order mediators (gene → drug paths); 3-hop explodes to ~5K-50K nodes (PrimeKG diameter is small due to protein_protein hub). Workflow:
1. Seed nodes = question entities.
2. 1-hop BFS with edge-type filter (depend on question type — for drug interactions keep `{drug_drug, contraindication, drug_protein}`).
3. 2-hop BFS from 1-hop neighborhood with tighter edge-type filter.
4. Compute Personalized PageRank với seeds as restart distribution; keep top-200 nodes by PPR score.
5. Load into `torch_geometric.data.HeteroData`.

**PubMed retrieval**: MedCPT dual encoder top-50 → MedCPT cross-encoder rerank to top-5. Use NCBI API key để get 10 req/s instead of 3. Cache XML results in `/kaggle/working/pubmed_cache.jsonl` + publish as Kaggle Dataset for session-to-session persistence.

## 3.2 Layer 4 — Four Agentic Judges

Cùng JSON-output pattern as §1.3.3. Brief template sketches:

**Judge 1 — Entity Validator**. Input: `{question, extracted_entities}`. Check: are entities correctly linked to UMLS CUIs? Any missing critical entities? Output: `{valid: bool, missing_entities: list, mislinked: list, confidence: float}`. Fallback: retry entity extraction với higher-recall model (BERN2).

**Judge 2 — Evidence Grounding Inspector**. Input: `{answer, evidence_bundle}`. Implements §1.3.1 full pipeline. Output: ClaimVerdict list + faithfulness_score + h5_present + recommendation.

**Judge 3 — Reasoning Soundness Auditor**. Input: `{answer, cot_steps, trm_paths}`. Checks: (a) each CoT step follows from previous by some edge in TRM paths; (b) no logical gaps (missing mediator nodes); (c) CDR < 0.05. Output: `{sound: bool, gaps: list, contradictions: list, rcs: float, rns: float}`.

**Judge 4 — Answer Faithfulness Guardian**. Input: `{answer, judge1_out, judge2_out, judge3_out}`. Aggregator: weighted vote across 3 prior judges. Output: `{final_verdict: ACCEPT|REVISE|ABSTAIN, reasoning: str}`.

Scoring rubric shared: 0–3 severity như §1.3.3. **Retry logic**: on REVISE, send answer + judge critiques back to Layer 5 synthesizer với max 2 retries; on third failure → ABSTAIN.

**Inference setup on 2×T4**: all 4 judges share one model instance (`medical_o1_verifier_3B` FP16, ~6 GB on GPU 1) with different system prompts. Batch judges 1-3 in parallel (independent), then judge 4 sequentially. Total latency ~8s per question.

## 3.3 Layer 5 — Answer Synthesis

**Model**: `Intelligent-Internet/II-Medical-8B` (Q4_K_M GGUF, ~5 GB on GPU 0) via llama.cpp. Alternative: `UCSC-VLAA/MedReason-8B` if more citation-grounded output desired.

**System prompt template**:
```text
You are a medical expert synthesizing an answer from verified evidence. 
You MUST:
1. Answer using ONLY the provided evidence paths and PubMed passages.
2. Tag every sentence with its source: [edge:EDGE_ID] for KG facts, [PMID:XXXXX] for literature.
3. Acknowledge uncertainty when confidence < 0.7; abstain when < 0.4.
4. For drug-related claims, include therapeutic dosage ranges when available.
5. Never generate a PMID or edge_id not in the provided evidence.
6. Format: Direct answer first, then reasoning with citations, then confidence level.

Evidence confidence: {validity_score:.2f}
Contradiction detected: {contradiction_flag}
```

User prompt: question + serialized TRM paths + top-5 PubMed passages + TRM trace summary.

## 3.4 Ablation design

Five arms, all evaluated on identical test items (paired), temperature=0, 3 seeds cho non-deterministic:

1. **Pure LLM**: II-Medical-8B zero-shot CoT, no retrieval.
2. **Standard RAG**: dense retrieval over PubMed (MedCPT top-5) → LLM.
3. **GraphRAG**: PrimeKG 2-hop subgraph serialized as triples → LLM.
4. **v2 (TRM-reranker)**: retrieve candidate answer options + use TRM as cross-encoder reranker over paths, then LLM.
5. **v3 (TRM-reasoner)**: full pipeline as specified — TRM produces reasoning trace, judges filter, LLM synthesizes.

Primary hypothesis: v3 reduces F1_hall by ≥5 pts absolute over GraphRAG baseline on MedQA + MedMCQA + MedBullets pooled. Secondary: v3 matches or exceeds GraphRAG on accuracy while strictly dominating on hallucination metrics.

## 3.5 Evaluation metrics beyond hallucination

- **Accuracy**: exact-match for MCQ, macro-F1 for yes/no/maybe, claim-F1 for open-ended.
- **RCS (Reasoning Completeness Score)**: `|matched_cot_edges ∩ gold_edges| / |gold_edges|`.
- **RNS (Reasoning Necessity Score)**: `|matched_cot_edges ∩ gold_edges| / |cot_edges|`.
- **Citation precision/recall**: over PMID set vs gold evidence from BioASQ snippets.
- **Latency** (p50/p95/p99) per question.
- **Token cost** if using API judges.
- **Abstention rate** (should be 5–15% for well-calibrated system).

## 3.6 Statistical significance

- **McNemar's paired test** on MCQ correctness: `statsmodels.stats.contingency_tables.mcnemar(table, exact=False, correction=True)`.
- **Bootstrap 95% CI** on accuracy differences (1000 resamples, BCa variant).
- **Holm-Bonferroni** correction for 4 pairwise comparisons (v3 vs each baseline).
- **Cohen's h** for proportion effect sizes; report alongside p-values.
- Sample-size planning: Δ=3% needs ~1,600 paired items — **pool MedQA + MedMCQA + MedBullets + MMLU-med to get ~6,800 items** for headline comparison.

---

# Part 4 — Code skeleton (Python class interfaces)

```python
# ============ Layer 1: Retrieval ============
class AgenticRetriever:
    def __init__(self, primekg: HeteroData, pubmed_client, scispacy_nlp, sapbert, medcpt):
        ...
    def retrieve(self, question: str) -> EvidenceBundle: ...
    def _extract_entities(self, text: str) -> list[QuestionEntity]: ...
    def _subgraph_2hop_ppr(self, entities: list[QuestionEntity], top_k: int = 200) -> HeteroData: ...
    def _pubmed_topk(self, question: str, k: int = 5) -> list[PubMedPassage]: ...

# ============ Layer 2: Translation ============
class MedicalGraphEmbedder(nn.Module):
    def __init__(self, primekg, sapbert, medcpt, hidden=256, codebook_size=4096, max_len=256):
        ...
    def forward(self, bundle: EvidenceBundle) -> dict[str, torch.Tensor]:
        """Returns {'inputs': LongTensor[1,256], 'puzzle_identifiers': LongTensor[1]}"""
    def pretrain_vq(self, subgraph_samples: Iterable[HeteroData], epochs=1): ...

# ============ Layer 3: TRM Reasoning ============
class TRMReasoner:
    def __init__(self, trm_model_path: str, max_steps=16, conf_threshold=0.85):
        self.model = load_trm(trm_model_path)
    def reason(self, embedder_output: dict) -> TRMOutput: ...
    def decode_paths(self, logits: torch.Tensor, codebook, primekg, top_k=5) -> list[Path]: ...
    def train_step(self, batch) -> dict: ...   # for fine-tuning

# ============ Layer 4: Judges ============
class JudgeBase:
    def __init__(self, model, system_prompt: str, output_schema: BaseModel):
        ...
    def evaluate(self, **inputs) -> BaseModel: ...

class EntityValidator(JudgeBase): ...
class EvidenceGroundingInspector(JudgeBase): ...
class ReasoningSoundnessAuditor(JudgeBase): ...
class AnswerFaithfulnessGuardian(JudgeBase): ...

class HallucinationJudge(JudgeBase):
    """The core §1.3.3 judge."""
    def evaluate(self, question: str, answer: str, 
                 evidence: EvidenceBundle) -> JudgeOutput: ...

# ============ Layer 5: Synthesis ============
class AnswerSynthesizer:
    def __init__(self, model_path: str, system_prompt_template: str):
        self.llm = LlamaCpp(model_path, n_ctx=8192, n_gpu_layers=-1)
    def synthesize(self, question: str, trm_output: TRMOutput,
                   evidence: EvidenceBundle) -> str: ...
    def enforce_provenance(self, answer: str, evidence: EvidenceBundle) -> tuple[bool, list[str]]: ...

# ============ Orchestrator ============
class MedicalReasoningSystemV3:
    def __init__(self, retriever, embedder, trm, judges: dict, synthesizer,
                 conf_threshold=0.4, max_retries=2):
        ...
    def answer(self, question: str) -> AnswerWithTrace: ...
```

---

# Part 5 — Risk assessment and Kaggle 2×T4 feasibility

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| TRM doesn't converge on medical graph data | Medium | High | Start from Sudoku-Extreme checkpoint; extensive VQ codebook pretraining; curriculum from 1-hop to 2-hop subgraphs |
| T4 can't handle II-Medical-8B + 3B judge simultaneously | Low | Medium | Q4_K_M (5GB) + 3B FP16 (6GB) + overhead = ~13GB of 16 GB — tight; fallback to Gemini Flash remote judge |
| PubMed rate limit exceeded during eval | Medium | Low | API key (10 rps) + persistent Kaggle Dataset cache + EPost batching |
| 9-hour session insufficient for training | Medium | Medium | Gradient checkpointing, mixed precision, checkpointed resume, reduce N_sup=8 during training |
| ScispaCy UMLS linker misses Vietnamese translated terms | Low (English benchmarks) | Low | Not an issue for MedQA/MedMCQA/etc (all English); would matter only for bilingual deployment |
| MedHallu F1 < 0.65 target | Medium | Medium | Fall back to Judge 4 weighting tuned on MedHallu val split |
| PrimeKG edge confidence scores are static/uncalibrated | High | Medium | Implement AMG-RAG's LLM-refinement stage to rescore edges per-query |
| T4 FP16-only breaks BF16-pretrained models | Known | Medium | Already designed around this — II-Medical-8B GGUF is FP16-compat; avoid Nemotron-H hybrids |
| Final answer accuracy < GPT-4 baseline on MedQA | High | Low | v3's goal is **hallucination reduction with interpretability**, not SOTA accuracy; position accordingly |

**Go / No-go**: feasible. Total VRAM budget: GPU 0 = Synthesizer (5 GB GGUF) + KV cache (3 GB) = 8/16 GB; GPU 1 = Verifier 3B (6 GB) + Embedder+TRM at inference (3 GB) + NLI (500 MB) + buffer = 10/16 GB. Training phase uses GPU 0 alone (embedder+TRM ~10 GB); GPU 1 free for concurrent eval. Session time: retrieval cache warm-up (~30 min for 6 datasets) + training (7h) fits 9h; eval (~2h for 6,800 pooled items at 1s/question) fits separate session.

---

# Part 6 — References with URLs

**Samsung TRM**
- Jolicoeur-Martineau 2025, arXiv:2510.04871 — https://arxiv.org/abs/2510.04871
- Code — https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Author blog — https://alexiajm.github.io/2025/09/29/tiny_recursive_models.html
- HRM (predecessor), Wang et al. 2025, arXiv:2506.21734

**Medical KG**
- PrimeKG (Chandak et al. 2023), doi:10.1038/s41597-023-01960-3 — https://github.com/mims-harvard/PrimeKG; https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
- AMG-RAG (Rezaei et al. 2025), arXiv:2502.13010 — https://github.com/MrRezaeiUofT/AMG-RAG
- MedReason (Wu et al. 2025), arXiv:2504.00993 — https://github.com/UCSC-VLAA/MedReason; HF dataset `UCSC-VLAA/MedReason`; HF model `UCSC-VLAA/MedReason-8B`

**Medical LLMs**
- II-Medical-8B — https://huggingface.co/Intelligent-Internet/II-Medical-8B
- HuatuoGPT-o1 family — https://huggingface.co/FreedomIntelligence
- medical_o1_verifier_3B — https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B
- OpenBioLLM — https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B
- BioMistral-7B — https://huggingface.co/BioMistral/BioMistral-7B
- Meditron — https://huggingface.co/epfl-llm/meditron-7b
- Baichuan-M1-14B — https://huggingface.co/baichuan-inc/Baichuan-M1-14B-Instruct
- Llama-3.1-Nemotron-Nano-8B — https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1
- MedGemma-27B — https://huggingface.co/google/medgemma-27b-it

**Embedders / Retrieval**
- SapBERT — https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
- PubMedBERT — https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
- MedCPT — https://github.com/ncbi/MedCPT; HF `ncbi/MedCPT-Query-Encoder`, `ncbi/MedCPT-Article-Encoder`, `ncbi/MedCPT-Cross-Encoder`
- ScispaCy — https://github.com/allenai/scispacy

**Hallucination benchmarks**
- Med-HALT (Pal et al. 2023), arXiv:2307.15343 — https://medhalt.github.io; https://github.com/medhalt/medhalt
- MedHallu (Pandit et al. 2025), arXiv:2502.14302 — https://medhallu.github.io; HF `UTAustin-AIHealth/MedHallu`; https://github.com/MedHallu/MedHallu
- MedHalu (Agarwal et al. 2024), arXiv:2409.19492
- HaluEval (Li et al. 2023), arXiv:2305.11747
- MEDEC (Ben Abacha et al. 2024/2025), arXiv:2412.19260 — https://github.com/abachaa/MEDEC
- MedHallBench, arXiv:2412.18947
- K-QA (Manes et al. 2024), arXiv:2401.14493

**Evaluation**
- MedQA (Jin et al. 2020), arXiv:2009.13081 — https://github.com/jind11/MedQA
- MedMCQA (Pal et al. 2022), arXiv:2203.14371 — HF `openlifescienceai/medmcqa`
- PubMedQA (Jin et al. 2019), arXiv:1909.06146 — https://pubmedqa.github.io
- BioASQ — http://bioasq.org; overview 2025 arXiv:2508.20554
- MMLU (Hendrycks et al. 2020), arXiv:2009.03300 — HF `cais/mmlu`
- MMLU-Pro (Wang et al. 2024), arXiv:2406.01574 — HF `TIGER-Lab/MMLU-Pro`
- MedBullets (Chen et al. 2024), arXiv:2402.18060

**Hallucination methods**
- RAGAS — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- HHEM-2.1-Open — https://huggingface.co/vectara/hallucination_evaluation_model
- SelfCheckGPT (Manakul et al. 2023), arXiv:2303.08896
- Semantic entropy (Farquhar et al. 2024), Nature doi:10.1038/s41586-024-07421-0
- FActScore (Min et al. 2023), arXiv:2305.14251
- SAFE (Wei et al. 2024), arXiv:2403.18802
- G-Eval (Liu et al. 2023), arXiv:2303.16634
- Prometheus-2 (Kim et al. 2024), arXiv:2405.01535
- Medical hallucination survey (Kim et al. 2025), arXiv:2503.05777
- Conformal abstention (Yadkori et al. 2024), arXiv:2405.01563

**Graph ML**
- HGT (Hu et al. 2020), arXiv:2003.01332 — https://github.com/UCLA-DM/pyHGT; PyG `HGTConv`, `HGTLoader`
- R-GCN (Schlichtkrull et al. 2018), arXiv:1703.06103

---

## Conclusion — kết luận

v3 thể hiện một triết lý cụ thể: **hallucination reduction không đạt được bằng model scaling mà bằng architectural separation of concerns**. Bằng cách decouple retrieval (Layer 1), structured reasoning (Layer 2–3 via TRM 7M), adversarial verification (Layer 4 multi-judge), và natural-language synthesis (Layer 5), mỗi layer có single responsibility và failure mode có thể audit độc lập. TRM's 7M parameters — a 0.1% fraction of the synthesizer — đóng vai trò disproportionately lớn vì nó xử lý đúng task mà LLMs weak nhất: discrete multi-hop reasoning over typed edges với explicit confidence tracking. The multiplicative confidence scheme (AMG-RAG) cùng provenance enforcement và hard constraints cho H5 tạo thành một defense-in-depth phù hợp với life-critical domain.

Two concrete deliverables from this research resolve user confusion permanently: (1) **Samsung TRM verified as non-geometric** — any reference tới "geometric reasoner" is incorrect; proper citation is arXiv 2510.04871 + `SamsungSAILMontreal/TinyRecursiveModels`. (2) **MedReason verified as dual artifact** — dataset (32K KG-grounded CoT) + model (MedReason-8B) both public, both compatible with v3's training pipeline for TRM fine-tuning.

The Kaggle 2×T4 constraint is **tight but feasible**: II-Medical-8B Q4_K_M (5GB) on GPU 0 + medical_o1_verifier_3B (6GB) on GPU 1 leaves enough headroom, and 7M-parameter TRM trains in ~7 hours within a 9-hour session given gradient checkpointing + FP16. The principal open research question is whether TRM's recursion depth (effective 384 layers at inference) is sufficient for 5+ hop medical chains, or whether deeper supervision (`N_sup=32`) would pay off — an ablation the user should prioritize empirically. For deployment, remote Gemini 2.5 Flash (1500 RPD free tier) as an independent judge provides decorrelation insurance that local-only pipelines cannot offer.