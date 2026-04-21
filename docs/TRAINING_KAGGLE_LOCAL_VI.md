# Train/Eval TRM Medical - Local va Kaggle

File nay ghi cac lenh moi cho phan train, xu ly data, inference/eval. Tat ca path nen de dang logic path (`data/...`, `external/...`) de `KaggleEnv` tu resolve local/Kaggle.

## Prepare MedReason seed JSONL

Buoc nay lay raw MedReason records, chay Layer 1 retrieval de tao `EvidenceBundle`, roi map reasoning/CoT sang PrimeKG `gold_edge_ids`.

Colab all-in-one:

- Notebook: `notebooks/colab/CB1-Build-Golden-Data.ipynb`
- Flow: materialize source -> install deps -> resolve/download PrimeKG + SapBERT + MedCPT -> build seed -> split `goldish/silver/reject` -> optional build TRM arrays.
- Output quan trong nhat de dua sang Kaggle train la:
  - `quality/train/trm_seed_goldish.jsonl`
  - `quality/validation/trm_seed_goldish.jsonl`

Local:

```powershell
python scripts/prepare_medreason_seed.py --source data/medreason --split train --output-jsonl data/trm_seed.jsonl --edge-mapper auto --limit 100
python scripts/prepare_medreason_seed.py --source data/medreason --split validation --output-jsonl data/trm_seed_val.jsonl --edge-mapper auto --limit 50
```

Wrapper local moi:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/local/New-SeedVenv.ps1
powershell -ExecutionPolicy Bypass -File scripts/local/Invoke-SystemSmoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/local/Invoke-SeedBuild.ps1 -Profile smoke-10
```

File env local:

- `.env.seed`
- `requirements-local-seed.txt`

Profiles co san trong `Invoke-SeedBuild.ps1`:

- `smoke-5`
- `smoke-10`
- `pilot-100`
- `standard-3000`
- `standard-5000`

Luu y practical:

- `smoke-10` phu hop voi uu tien trong audit: chay smoke 10-100 examples truoc khi scale len nhieu nghin mau.
- `standard-3000` la preset hop ly de tao `goldish` va train thu tren local/Kaggle khi smoke da on.

Kaggle:

```bash
!python scripts/prepare_medreason_seed.py --kaggle --source data/medreason --split train --output-jsonl data/trm_seed.jsonl --edge-mapper auto --limit 100
```

Neu muon load truc tiep tu Hugging Face:

```powershell
python scripts/prepare_medreason_seed.py --source UCSC-VLAA/MedReason --split train --output-jsonl data/trm_seed.jsonl --edge-mapper auto
```

Edge mapper co 4 mode:

- `auto`: uu tien LLM neu co `--llm-model-name` that hoac `GEMINI_API_KEY`; neu khong thi moi fallback `heuristic`.
- `direct`: chi dung `edge_id`, `edge:...`, `premise_edge_ids` co san trong record.
- `heuristic`: ket hop direct IDs voi lexical matching tren PrimeKG candidate edges.
- `llm`: dung structured LLM/Judge de chon edge IDs tu candidate edges, fallback ve heuristic neu parse fail.

LLM-assisted mode:

```powershell
python scripts/prepare_medreason_seed.py --source data/medreason --output-jsonl data/trm_seed.jsonl --edge-mapper llm --llm-model-name data/checkpoints/judges/medical_o1_verifier_3B
```

Neu dung OpenAI structured selector:

```powershell
$env:OPENAI_API_KEY="..."
python scripts/prepare_medreason_seed.py --source data/medreason --output-jsonl data/trm_seed.jsonl --edge-mapper auto --llm-model-name gpt-4o-mini
```

Alias `4omini` cung duoc normalize thanh `gpt-4o-mini`.

Luu y: raw MedReason hien co reasoning text nhung khong co PrimeKG `edge_id` chuan. Golden TRM labels duoc tao bang cach: MedReason reasoning + Layer 1 candidate PrimeKG edges + structured LLM edge selector. `heuristic` chi nen dung cho smoke test nho khi chua co LLM.

Raw file Kaggle `huynhnhuthuyk18hcm/medreason` hien la single JSONL (`ours_quality_33000.jsonl`). Loader se chia deterministic neu khong co split file rieng: `train=90%`, `validation=5%`, `test=5%`.

Logs:

- `medreason_adapter.jsonl`
- `layer1_retrieval.jsonl`
- `pubmed_client.jsonl`

## Filter seed quality

Sau khi co `trm_seed.jsonl`, co the tach thanh 3 tier:

- `goldish`: non-empty `gold_edge_ids`, edge mapper `direct` hoac `llm_*`, `entity_linker_backend=scispacy_umls`, PrimeKG khong phai `empty_graph`
- `silver`: van co the dung de inspect/debug nhung thieu it nhat mot quality gate
- `reject`: missing gold edges, missing evidence edges, invalid row, hoac explicit empty-graph retrieval

Local/Kaggle/Colab deu dung chung CLI nay:

```powershell
python scripts/filter_seed_quality.py --input-jsonl data/trm_seed.jsonl --output-dir data/seed_quality/train
```

Output:

- `trm_seed_goldish.jsonl`
- `trm_seed_silver.jsonl`
- `trm_seed_rejects.jsonl`
- `seed_quality_summary.json`

## Build TRM dataset

Input can la JSONL da normalize gom `EvidenceBundle` va `gold_edge_ids` hoac `reasoning` co tag edge.

```powershell
python scripts/build_trm_dataset.py --input-jsonl data/trm_seed.jsonl --output-dir data/trm_medical --split train
python scripts/build_trm_dataset.py --input-jsonl data/trm_seed_val.jsonl --output-dir data/trm_medical --split val
```

Kaggle:

```bash
!python scripts/build_trm_dataset.py --kaggle --input-jsonl data/trm_seed.jsonl --output-dir data/trm_medical --split train
!python scripts/build_trm_dataset.py --kaggle --input-jsonl data/trm_seed_val.jsonl --output-dir data/trm_medical --split val
```

Output split gom:

- `dataset.json`
- `all__inputs.npy`
- `all__labels.npy`
- `all__puzzle_identifiers.npy`
- `all__puzzle_indices.npy`
- `all__group_indices.npy`
- `skipped_examples.json` neu co skipped sample

Logs:

- `trm_dataset_builder.jsonl`
- `build_trm_dataset.jsonl`

## Train medical TRM

Local:

```powershell
python scripts/train_medical_trm.py --dataset-dir data/trm_medical --train-split train --val-split val --model-path external/TinyRecursiveModels --output-dir data/checkpoints/active --epochs 1 --batch-size 1 --device auto
```

Kaggle:

```bash
!python scripts/train_medical_trm.py --kaggle --dataset-dir data/trm_medical --train-split train --val-split val --model-path external/TinyRecursiveModels --output-dir data/checkpoints/active --epochs 1 --batch-size 1 --device auto
```

Neu chi smoke test nho:

```powershell
python scripts/train_medical_trm.py --dataset-dir data/trm_medical --epochs 1 --batch-size 1 --max-train-batches 10 --max-val-batches 5 --eval-every 5 --save-every 10 --device cpu
```

Output:

- Final checkpoint: `data/checkpoints/active/medical_trm.pt`
- Resume checkpoints: `data/checkpoints/active/resume/checkpoint_step_*.pt`
- Step checkpoints: `data/checkpoints/active/medical_trm_step_*.pt`

Logs:

- `medical_trm_trainer.jsonl`
- `layer3_trm.jsonl`
- `checkpoint_manager.jsonl`

Metrics da log:

- Train loss, token loss, halt loss
- Token accuracy, sequence accuracy
- Valid label token count
- Batch size, seq len, recursive steps taken
- Learning rate
- Validation loss/accuracy
- Checkpoint path/step/metrics

## Check TRM readiness

```powershell
python scripts/check_trm_ready.py --model-path data/checkpoints/active/medical_trm.pt
```

Kaggle:

```bash
!python scripts/check_trm_ready.py --kaggle --model-path data/checkpoints/active/medical_trm.pt
```

Chi xem TRM la real neu:

- `trm_is_real=true`
- `checkpoint_loaded=true`
- `medical_compatible=true`
- backend la `samsung_trm_medical`

## Eval/inference logging

Local:

```powershell
python -m src.eval.run_eval data/eval/pubmedqa.jsonl --output-json data/eval_results.json
```

Kaggle:

```bash
!python -m src.eval.run_eval data/eval/pubmedqa.jsonl --kaggle --output-json data/eval_results.json
```

Eval logs per sample vao `evaluator.jsonl`:

- Accuracy / exact match
- RAGAS-like faithfulness
- Hallucination precision/recall/F1
- Contradiction detection rate
- RCS/RNS
- Abstention, confidence
- Provenance count
- Answer char length
- Latency per sample

Aggregate output co:

- `latency_p50_ms`
- `latency_p95_ms`
- `latency_p99_ms`
- `mean_provenance_count`
- `mean_answer_char_length`

## Path mapping

Local:

- `data/medreason` -> `D:\CapstoneProjectSP26\kaggle_assets\medreason-upload` neu repo khong co folder nay.
- Neu `data/checkpoints/medreason-8b` trong repo rong, `KaggleEnv` co the map sang `D:\CapstoneProjectSP26\kaggle_assets\checkpoints\medreason-8b`.
- Neu `data/kg/primekg` trong repo khong co data, `KaggleEnv` co the map sang `D:\CapstoneProjectSP26\kaggle_assets\primekg-upload`.
- Co the override asset root bang env var `MINIMED_KAGGLE_ASSETS`.

Kaggle:

- `data/medreason` -> `/kaggle/input/medreason`
- `data/checkpoints/medreason-8b` -> `/kaggle/input/medreason-8b`
- `data/kg/primekg` -> `/kaggle/input/primekg`
- `external/TinyRecursiveModels` -> `/kaggle/input/trm-real` hoac `/kaggle/working/external/TinyRecursiveModels`
- `data/checkpoints/active` -> `/kaggle/working/checkpoints/active`
- `data/logs` -> `/kaggle/working/logs`
