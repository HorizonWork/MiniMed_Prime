# Implementation Audit - MiniMed Prime

Ngay luc ghi note nay, trong repo da co skeleton end-to-end va cac fallback deterministic cho nhieu component. Phan TRM da duoc chuyen sang huong dung Samsung TRM goc, nhung chi duoc xem la "real" khi co checkpoint medical-compatible da fine-tune, khong tinh checkpoint ARC/Sudoku/Maze cu.

## Cap nhat ve synthesizer Med 8B

- Report cu noi `Intelligent-Internet/II-Medical-8B` co Q4_K_M GGUF la sai voi hien trang artifact: repo khong co `.gguf` usable.
- Code hien dang dung `data/checkpoints/medreason-8b` lam default Layer 5 model path.
- Backend that mong doi cho Layer 5 la `transformers_4bit`, khong phai `llama_cpp`, khi thu muc model co `config.json` va khong co `.gguf`.
- Dataset Kaggle da mapping `data/checkpoints/medreason-8b` -> `/kaggle/input/medreason-8b` qua `KaggleEnv`.
- Local resolver cung da mapping repo path rong `data/checkpoints/medreason-8b` sang mirror `D:\CapstoneProjectSP26\kaggle_assets\checkpoints\medreason-8b` neu co.
- Neu khong load duoc `transformers_4bit` do thieu `bitsandbytes`/`accelerate`/VRAM, code van fallback ve heuristic va log backend.

## TRM hien tai

- Da co strict loader cho Samsung TinyRecursiveModels trong `src/models/trm_wrapper.py`.
- Da co guard de khong load nham checkpoint ARC vao bai toan medical: checkpoint ARC bi mismatch vocab/puzzle embedding nen bi danh dau khong compatible.
- Da co `scripts/check_trm_ready.py` de kiem tra trang thai TRM.
- Da co dataset builder trong `src/training/trm_dataset_builder.py` va CLI `scripts/build_trm_dataset.py`.
- Chua co checkpoint `medical_trm.pt`/`medical_trm.ckpt` da train that su.
- Chua chay build dataset vi hien chua co file seed kieu `data/trm_seed.jsonl` gom `EvidenceBundle` + `gold_edge_ids`.
- Raw MedReason dataset da tai local tai `D:\CapstoneProjectSP26\kaggle_assets\medreason-upload` va da upload len Kaggle private dataset `huynhnhuthuyk18hcm/medreason`.
- `data/medreason` da map linh hoat: local -> `kaggle_assets\medreason-upload`, Kaggle -> `/kaggle/input/medreason`.
- Raw MedReason la single JSONL, loader tu chia deterministic `train=90%`, `validation=5%`, `test=5%` neu khong co split file rieng.
- Neu chay `--allow-official-untrained`, TRM chi la official-untrained de kiem tra interface, khong duoc tinh la real backend.

Lenh kiem tra TRM:

```powershell
python scripts/check_trm_ready.py
python scripts/check_trm_ready.py --allow-official-untrained
```

Lenh build dataset khi da co seed:

```powershell
python scripts/build_trm_dataset.py --input-jsonl data/trm_seed.jsonl --output-dir data/trm_medical --split train
```

## Report vs code - trang thai theo layer/component

### Layer 1 - Retrieval

- Da co: entity linker scispaCy/fallback, PrimeKG extractor, PubMed client/cache, BM25/dense fallback, structured logging.
- Con thieu/partial: SapBERT disambiguation chua duoc evaluate MedQA >80% recall; MedCPT cross-encoder rerank phu thuoc checkpoint; PubMed triple extraction moi o muc don gian; chua co vong agentic retrieval theo ToG/Adaptive-RAG/CRAG/FLARE trong report; chua co BERN2 high-recall fallback.

### Layer 2 - Graph Embedder

- Da co: EvidenceBundle -> token grid `[1, 256]`, HF encoder/fallback, VQ-like tokenization, TRMInputBundle contract.
- Con thieu/partial: chua phai full PyG HGTConv dung hetero metadata PrimeKG; VQ pretrain 50K subgraph chua chay; codebook hien chua dong nhat hoan toan voi target TRM vocab 8192; chua co edge-position mapping du manh de train TRM tren gold edge sequence; chua co 4K graph + 2K PubMed + 1K reserved + 1K special token partition nhu report.

### Layer 3 - TRM

- Da co: Samsung repo integration, strict medical compatibility check, multiplicative confidence, path-stability halting, train_step, fallback/internal path decoding.
- Con thieu/partial: checkpoint medical fine-tuned; full training script official voi gradient accumulation/EMA/cosine schedule/checkpoint validation; contradiction detection bang NLI real mac dinh; dataset thuc te tu MedReason/MedQA gan voi PrimeKG edge IDs; chua co AMG-RAG edge rescoring per-query.

### Layer 4 - Judges

- Da co: 4 judge class, HallucinationJudge wrapper, heuristic/transformers/Gemini backend, JSON structured parsing fallback.
- Con thieu/partial: checkpoint `medical_o1_verifier_3B`; HHEM/MedNLI/LettuceDetect integration that; hard constraints dosage/range database; UMLS synonym expansion trong judge pipeline; H6 temporal detector va H7 population detector moi o muc heuristic; multi-sample path consistency/semantic entropy chua co.

### Layer 5 - Synthesis

- Da co: GGUF backend, transformers 4-bit backend, heuristic fallback, provenance enforcement, abstention behavior.
- Da cap nhat them: `transformers_4bit` dung NF4, FP16, `trust_remote_code=True`, log VRAM/latency/token count khi generation, retry mot lan voi `max_new_tokens` giam 1/2 neu CUDA OOM.
- Con thieu/partial: MedReason-8B transformers backend chua duoc smoke-test truc tiep tren Kaggle trong audit nay; forced citation chat luong phu thuoc model; dosage range moi lay tu evidence neu co, chua co authoritative dosage DB.

### Evaluation

- Da co: exact match, macro-F1, RAGAS-like faithfulness heuristic, hallucination F1, contradiction rate, RCS/RNS, McNemar, bootstrap CI, local runner.
- Con thieu/partial: loader dataset thuc cho MedQA/MedMCQA/PubMedQA/BioASQ/MMLU-med/MedBullets/Med-HALT/MedHallu/MedHalu/HaluEval/K-QA/MEDEC; 5-arm ablation full; Spearman correlation voi human judgment; benchmark latency/abstention tren dev set that; citation precision/recall vs BioASQ snippets; Holm-Bonferroni va Cohen's h.

### Orchestrator, Kaggle, Ops

- Da co: orchestrator end-to-end, KaggleEnv, structured logger, checkpoint utilities, setup/smoke scripts, notebook templates.
- Con thieu/partial: LangGraph/DSPy workflow nhu report; resume/eval flow chua duoc validate tren Kaggle 2xT4; automatic upload artifact sau khi train TRM chua co; moi component van can backend readiness check truoc benchmark.

## Report gap bo sung khong nam gon trong 5 layer

- H5 da co schema/blocking behavior, nhung hard constraint DB cho dosage/contraindication chua co nguon authoritative rieng.
- H6 temporal hallucination can publication-date/guideline version filter; hien chua co guideline versioning.
- H7 population hallucination can UMLS semantic type Age Group/Population Group; hien chua co semantic-type filter that.
- Semantic entropy alternative cho SelfCheckGPT chua co implementation.
- Triple-reasoning-path consistency `N=3` chua co orchestration/eval mode.
- BioASQ citation precision/recall va K-QA must-have recall chua co runner rieng.
- MedReason raw dataset -> EvidenceBundle -> gold PrimeKG edge IDs da co adapter `scripts/prepare_medreason_seed.py`; ho tro direct edge IDs, heuristic mapping va LLM structured selector. Raw MedReason hien co reasoning text nhung khong co PrimeKG `edge_id` chuan, nen golden labels tot nhat can dung Layer 1 candidate edges + LLM selector; heuristic chi la fallback/smoke-test.
- Notebook NB3 van la training stub, chua phai runbook train TRM hoan chinh tu MedReason.

## Logging/readiness audit

- MedReason preprocessing: da co `scripts/prepare_medreason_seed.py` va `src/training/medreason_adapter.py`; log records seen/saved/skipped, retrieval evidence edge count, PubMed count, selected gold edge count, direct/heuristic/LLM mapping diagnostics.
- PrimeKG loader da ho tro layout Kaggle upload `edges.csv + nodes.csv` va uu tien backend nay de tranh doc nham feature tables/metadata.
- Data processing TRM: da log examples seen/saved/skipped, mean/max edge count, mean/max node count, PubMed passage count, gold path length, labeled token count, output bytes, metadata.
- TRM training: da co `scripts/train_medical_trm.py` va `src/training/trm_trainer.py`; log config, model params, backend, dataset stats, epoch, train loss, token loss, halt loss, token accuracy, sequence accuracy, valid label token count, batch size, seq len, steps taken, learning rate, validation metrics, checkpoint events.
- TRM inference: da log init, recursion start, tung recursion step, halt reason, path lengths, path confidences, contradiction score trong trace, backend status.
- Layer 5 inference: da log backend loaded, synthesis start, transformers generation latency/token count/VRAM, OOM retry, provenance enforcement, answer finalized.
- Eval runner: co per-sample JSONL qua `StructuredLogger`, accuracy, macro-F1, hallucination precision/recall/F1, RAGAS-like faithfulness, contradiction rate, RCS/RNS, abstention, confidence, provenance count, answer length, p50/p95/p99 latency.
- Chua day du: benchmark suite full voi dataset loader that, 5-arm ablation metrics, confusion matrix hallucination H1-H7, calibration curve, GPU peak memory per layer, automatic upload artifact sau train.

Lenh train TRM medical local/Kaggle:

```powershell
python scripts/prepare_medreason_seed.py --source data/medreason --split train --output-jsonl data/trm_seed.jsonl --edge-mapper auto
python scripts/prepare_medreason_seed.py --source data/medreason --split validation --output-jsonl data/trm_seed_val.jsonl --edge-mapper auto
python scripts/train_medical_trm.py --dataset-dir data/trm_medical --train-split train --val-split val --model-path external/TinyRecursiveModels --output-dir data/checkpoints/active --epochs 1 --batch-size 1 --device auto
python scripts/train_medical_trm.py --kaggle --dataset-dir data/trm_medical --train-split train --val-split val --model-path external/TinyRecursiveModels --output-dir data/checkpoints/active --epochs 1 --batch-size 1 --device auto
```

Lenh eval voi output aggregate JSON:

```powershell
python -m src.eval.run_eval data/eval/pubmedqa.jsonl --output-json data/eval_results.json
python -m src.eval.run_eval data/eval/pubmedqa.jsonl --kaggle --output-json data/eval_results.json
```

## Uu tien tiep theo

1. Tao seed dataset nho `data/trm_seed.jsonl` gom `EvidenceBundle` + `gold_edge_ids`.
2. Chay `scripts/build_trm_dataset.py` de sinh `.npy` theo format TRM.
3. Viet/hoan thien `scripts/train_medical_trm.py` dung Samsung TRM goc, FP16, vocab 8192, seq len 256, checkpoint moi 2500 step.
4. Train smoke 10-100 examples, xuat `medical_trm.pt`.
5. Chay `scripts/check_trm_ready.py` de xac nhan `trm_is_real=true`.
6. Upload checkpoint medical len Kaggle dataset roi rerun `scripts/smoke_test.py --kaggle`.
7. Sau TRM, uu tien Layer 2 edge mapping va Layer 4 real verifier vi hai phan nay anh huong truc tiep den provenance va hallucination.
