# Hướng Dẫn Chạy MiniMed Prime Trên Kaggle

Tài liệu này hướng dẫn cách dùng codebase hiện tại với các Kaggle datasets đã upload sẵn, giảm tối đa việc sửa path thủ công trong notebook.

## 1. Trạng thái hiện tại

Các dataset đã được upload lên Kaggle:

- `huynhnhuthuyk18hcm/primekg`
- `huynhnhuthuyk18hcm/sapbert`
- `huynhnhuthuyk18hcm/medcpt-query`
- `huynhnhuthuyk18hcm/medcpt-cross`
- `huynhnhuthuyk18hcm/medcpt-article`
- `huynhnhuthuyk18hcm/medreason`
- `huynhnhuthuyk18hcm/medreason-8b`
- `huynhnhuthuyk18hcm/minimed-prime-source`
- `huynhnhuthuyk18hcm/trm-real`

Trạng thái từng layer:

- Layer 1 có thể chạy thật nếu attach `primekg`, `sapbert`, `medcpt-article` và môi trường có đủ dependency.
- Layer 5 có thể chạy backend thật qua `transformers_4bit` nếu attach `medreason-8b` và có `bitsandbytes`.
- Layer 3 đã có dataset `trm-real` chứa repo TinyRecursiveModels và checkpoint ARC public, nhưng checkpoint này không tương thích trực tiếp với medical token space hiện tại (`vocab_size=8192`, medical puzzle ids). Vì vậy hệ thống vẫn có thể fallback nếu chưa có checkpoint medical-compatible.
- Layer 4 vẫn có thể rơi về heuristic nếu chưa có judge model thật kiểu `medical_o1_verifier_3B`.

## 2. Code map path Kaggle như thế nào

Code đã được chỉnh để tự map path Kaggle qua `KaggleEnv`.

Các đường dẫn logic trong code:

- `data/kg/primekg` -> `/kaggle/input/primekg`
- `data/checkpoints/sapbert` -> `/kaggle/input/sapbert`
- `data/checkpoints/medcpt-query` -> `/kaggle/input/medcpt-query`
- `data/checkpoints/medcpt-cross` -> `/kaggle/input/medcpt-cross`
- `data/checkpoints/medcpt-article` -> `/kaggle/input/medcpt-article`
- `data/medreason` -> `/kaggle/input/medreason`
- `data/checkpoints/medreason-8b` -> `/kaggle/input/medreason-8b`
- `external/TinyRecursiveModels` -> repo nested trong `/kaggle/input/trm-real`

Alias tương thích ngược:

- `data/checkpoints/llm` -> `medreason-8b`
- `data/checkpoints/medcpt/article_encoder` -> `medcpt-article`
- `data/checkpoints/medcpt/query_encoder` -> `medcpt-query`
- `data/checkpoints/medcpt/cross_encoder` -> `medcpt-cross`

Nếu chạy `scripts/setup_environment.py --kaggle --skip-downloads`, script sẽ cố copy repo TRM đã attach sang vùng writable `/kaggle/working/external/TinyRecursiveModels`.

## 3. Cách attach dataset trong Kaggle Notebook

Trong notebook Kaggle:

1. Bật `GPU T4 x2` nếu muốn test gần với target deploy.
2. Vào `Add input`.
3. Attach các dataset:
   - `huynhnhuthuyk18hcm/minimed-prime-source`
   - `huynhnhuthuyk18hcm/primekg`
   - `huynhnhuthuyk18hcm/sapbert`
   - `huynhnhuthuyk18hcm/medcpt-query`
   - `huynhnhuthuyk18hcm/medcpt-cross`
   - `huynhnhuthuyk18hcm/medcpt-article`
   - `huynhnhuthuyk18hcm/medreason`
   - `huynhnhuthuyk18hcm/medreason-8b`
   - `huynhnhuthuyk18hcm/trm-real`

Mount point mong đợi:

```text
/kaggle/input/minimed-prime-source
/kaggle/input/datasets/<owner>/<slug>
/kaggle/input/datasets/<owner>/<slug>/versions/<n>
/kaggle/input/primekg
/kaggle/input/sapbert
/kaggle/input/medcpt-query
/kaggle/input/medcpt-cross
/kaggle/input/medcpt-article
/kaggle/input/medreason
/kaggle/input/medreason-8b
/kaggle/input/trm-real
```

Lưu ý quan trọng:

- Notebook không tự có source code repo chỉ vì bạn attach model/data.
- `minimed-prime-source` là dataset chứa `src.zip`, `scripts.zip`, `notebooks.zip`, `tests.zip` và các file root như `requirements-integration.txt`, `KAGGLE_SOURCE_VERSION.txt`.
- Các notebook Kaggle đã được sửa để tự extract source code từ dataset này vào `/kaggle/working/MiniMed_Prime` trước khi chạy các cell tiếp theo.
- Cell bootstrap sẽ in `Source candidates`, `Source dataset root`, và `Source version marker` để debug mount path và version source đang chạy.

## 4. Cài môi trường trên Kaggle

Nếu model và data đã attach sẵn:

```bash
!python scripts/setup_environment.py --kaggle --skip-downloads
```

`--skip-downloads` quan trọng vì model/data đã nằm trong Kaggle datasets. Không cần tải lại từ Hugging Face hoặc Dataverse.

Tren Kaggle, lenh nay hien chay o `safe pip mode` theo mac dinh:

- Khong nang cap/ha cap cac package da co san trong image.
- Khong co gang cai cac package native rui ro cao nhu `spacy`, `scispacy`, model `en_core_sci_lg`, `llama-cpp-python`, `torch-geometric`, `torch-scatter`, `torch-sparse`.
- Muc tieu la tranh lam vo ABI `numpy/pandas` cua image Kaggle roi keo theo loi import hang loat.

He qua truc tiep cua che do nay:

- Layer 1 co the fallback sang `rule_based` thay vi `scispacy_umls`.
- Smoke test van co the chay, nhung `components_real.layer1` se la `false`.
- Neu log co `reason = "scispacy_package_missing"` hoac `reason = "scispacy_model_missing"` thi do la fallback du kien, khong phai loi path model.

Neu ban chu dong muon force cai day du dependency vao notebook image, set bien moi truong truoc khi chay:

```python
import os
os.environ["MINIMED_KAGGLE_PIP_MODE"] = "force"
```

Chi nen dung `force` khi ban chap nhan rui ro pip co the mutate image va gay loi binary incompatibility.

Nếu chỉ muốn cài dependency:

```bash
!pip install -r requirements-integration.txt
```

`requirements-integration.txt` hiện đã bao gồm các dependency chính:

- `transformers`
- `sentence-transformers`
- `datasets`
- `bitsandbytes`
- `accelerate`
- `torch-geometric`
- `llama-cpp-python`
- `einops`

Neu muon uu tien OpenAI cho judge pipeline va MedReason edge selector, dat Kaggle Secret `OPENAI_API_KEY` roi them:

```python
import os
os.environ["MINIMED_JUDGE_MODEL"] = "gpt-4o-mini"
os.environ["MEDREASON_EDGE_LLM"] = "gpt-4o-mini"
os.environ["MINIMED_SYNTHESIS_MODEL"] = "gpt-4o-mini"
```

## 5. Smoke test nhanh

Chạy:

```bash
!python scripts/smoke_test.py --kaggle
```

Kết quả sẽ:

- In ra JSON report ngắn trên console.
- Ghi full `AnswerWithTrace` vào `/kaggle/working/smoke_test_output.json`.
- Ghi structured logs vào `/kaggle/working/logs/`.

Các trường cần kiểm tra trong report:

- `components_real`
- `components_fallback`
- `backend_details`
- `provenance_tags_found`
- `trm_paths_found`

Ý nghĩa:

- `layer1=true` nghĩa là retrieval đã dùng backend thật nhiều hơn fallback.
- `layer5=true` nghĩa là synthesis đang dùng `llama_cpp` hoặc `transformers_4bit`.
- `layer3=false` hiện vẫn có thể xảy ra dù attach `trm-real`, vì checkpoint ARC public không khớp medical vocabulary/config của pipeline.

## 6. Mặc định hệ thống hiện tại

Khi dùng `SystemConfig()` mặc định, code ưu tiên:

- `sapbert` từ `data/checkpoints/sapbert`
- `medcpt-article` từ `data/checkpoints/medcpt-article`
- `medreason-8b` từ `data/checkpoints/medreason-8b`
- `TinyRecursiveModels` từ `external/TinyRecursiveModels`

Nếu attach đúng dataset trên Kaggle thì thường không cần override các path này.

## 7. Backend Layer 5

Layer 5 `AnswerSynthesizer` hỗ trợ:

- `gguf`
- `transformers_4bit`
- `auto`

Chế độ `auto`:

1. Ưu tiên `.gguf` nếu có.
2. Nếu không có `.gguf`, dùng `transformers_4bit`.

Với setup Kaggle hiện tại, backend thật mong đợi là:

- backend: `transformers_4bit`
- model path: `/kaggle/input/medreason-8b`

## 8. Trạng thái TRM thật

Dataset `trm-real` hiện chứa:

- Repo `TinyRecursiveModels`.
- Checkpoint public ARC: `trm_arc_v1_public_step_518071.pt`.
- Sidecar config: `all_config.yaml`.

Kiểm tra local hiện tại cho thấy checkpoint ARC có shape không khớp với medical TRM wrapper:

- checkpoint token embedding: `12 x 512`
- checkpoint puzzle embedding: `876406 x 512`
- medical wrapper đang cần token vocabulary `8192` và medical puzzle ids.

Kết luận thực tế:

- Có thể kiểm tra được code path import/load TRM thật.
- Không nên claim đây là medical TRM thật cho pipeline Layer 3.
- Để `trm_is_real=True` một cách hợp lệ, cần checkpoint được train/fine-tune với config medical-compatible của pipeline hiện tại, đặc biệt là `vocab_size=8192`.

## 9. Chạy pipeline thủ công trong notebook

Ví dụ:

```python
from src.orchestrator import MedicalReasoningSystemV3, SystemConfig

config = SystemConfig()
system = MedicalReasoningSystemV3(config)

question = (
    "A 68-year-old male with atrial fibrillation and newly diagnosed peptic ulcer disease "
    "is currently on warfarin. Which of the following antibiotics for H. pylori eradication "
    "poses the highest bleeding risk? (A) Amoxicillin (B) Clarithromycin "
    "(C) Metronidazole (D) Doxycycline"
)

result = system.answer(question)
print(result.answer_text)
print(result.provenance)
```

## 10. Structured logs nằm ở đâu

Log JSONL được ghi ở:

- local: `data/logs/`
- Kaggle: `/kaggle/working/logs/`

Các file quan trọng:

- `layer1_retrieval.jsonl`
- `layer2_embedder.jsonl`
- `layer3_trm.jsonl`
- `layer4_judges.jsonl`
- `layer5_synthesis.jsonl`
- `orchestrator.jsonl`

Nên mở `layer3_trm.jsonl` để xem:

- `trm_init`
- `recursion_start`
- `recursion_step`
- `recursion_halted`
- `paths_decoded`

## 11. Lỗi thường gặp

### 11.1 Layer 5 vẫn báo `heuristic`

Nguyên nhân thường là:

- Chưa attach `medreason-8b`.
- Thiếu `bitsandbytes`.
- Thiếu `accelerate`.
- Lỗi load model qua `transformers`.

Kiểm tra:

- `backend_details.layer5`
- `data/logs/layer5_synthesis.jsonl`

### 11.2 Layer 1 không dùng SapBERT/MedCPT thật

Nguyên nhân thường là:

- Chưa attach `sapbert`.
- Chưa attach `medcpt-article`.
- Thiếu `transformers`.
- Thiếu `scispacy`.

### 11.3 TRM vẫn là fallback

Đây là trạng thái có thể chấp nhận ở thời điểm hiện tại nếu chưa có checkpoint medical-compatible.

Checklist:

- Đã attach `huynhnhuthuyk18hcm/trm-real`.
- `scripts/setup_environment.py --kaggle --skip-downloads` đã chạy xong.
- `layer3_trm.jsonl` có event `trm_init`.
- Nếu log báo shape mismatch embedding/head thì checkpoint ARC không dùng được trực tiếp cho medical TRM.

Muốn Layer 3 thật cần một trong hai hướng:

- Train/fine-tune TRM với `vocab_size=8192` và medical puzzle ids của pipeline.
- Viết adapter có kiểm soát để chuyển token space, nhưng cách này cần benchmark lại vì không còn là checkpoint nguyên bản.

### 11.4 OOM trên T4

Việc này dễ xảy ra nhất ở:

- Layer 5 với `medreason-8b`.
- Layer 3 nếu dùng checkpoint thật lớn.

Khuyến nghị:

- Dùng `transformers_4bit`.
- Không bật BF16.
- Giữ batch size nhỏ.
- Chạy smoke test trước khi benchmark.

## 12. Trình tự khuyến nghị để chạy thật trên Kaggle

1. Attach 7 datasets đã upload.
2. Cài dependency bằng `requirements-integration.txt`.
3. Chạy `!python scripts/setup_environment.py --kaggle --skip-downloads`.
4. Chạy `!python scripts/smoke_test.py --kaggle`.
5. Kiểm tra `components_real`, `components_fallback`, `backend_details`, và log JSONL.
6. Nếu smoke test ổn mới chạy notebook eval hoặc benchmark.

## 13. Ghi chú thực tế

- PrimeKG dataset đã upload theo dạng file CSV trực tiếp, không phải zip.
- `primekg-upload.zip` không cần dùng trong notebook Kaggle.
- `trm-real` là asset phục vụ kiểm tra repo/checkpoint TRM, nhưng checkpoint trong đó là ARC public checkpoint, không phải medical TRM checkpoint.
- Code đã được sửa để khớp với slug dataset hiện tại, nên ưu tiên attach đúng tên dataset như phần trên.
