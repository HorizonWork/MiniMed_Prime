# Evidence Fusion Phase Plan

Tai lieu nay ghi lai huong nang MiniMed Prime tu pipeline `KG-first` sang `evidence-fusion`, trong do PrimeKG chi la mot lane bang chung thay vi la dieu kien song con duy nhat.

## 1. Vai tro LLM hien tai trong repo

LLM hien tai khong chi dung de judge. Trong code, LLM dang co 3 vai tro rieng:

1. MedReason edge selector trong buoc tao seed TRM:
   - `scripts/prepare_medreason_seed.py`
   - `src/training/medreason_adapter.py`
   - Khi `--edge-mapper llm` hoac `--edge-mapper auto` gap model/API key hop le, LLM duoc dung de chon `gold_edge_ids` tu tap candidate PrimeKG edges da retrieve.
   - Neu khong co LLM/backend that, pipeline roi ve `heuristic`.

2. Layer 4 judges:
   - `src/layers/layer4_judges.py`
   - Dung de cham hallucination, grounding, logical issues, abstention recommendation.

3. Layer 5 synthesizer:
   - `src/layers/layer5_synthesis.py`
   - Dung de tong hop final answer tu evidence + TRM output + judge feedback.

Ket luan ngan:
- LLM hien tai da tham gia vao buoc map raw MedReason -> pseudo-gold edge path.
- LLM khong chi la judge.
- Tuy nhien LLM chua duoc dung nhu mot lane retrieval/text-evidence co cau truc cho textbook-heavy hoac graph-missing cases.

## 2. Van de hien tai

Pipeline seed/train cua TRM dang gap 4 van de goc:

1. Raw MedReason khong dam bao da co PrimeKG `edge_id` chuan.
   - `parse_reasoning_chain()` co the rut ra edge id neu raw record co tag, nhung day khong phai tinh huong on dinh.
   - Trong da so truong hop, `gold_edge_ids` phai duoc suy ra tu candidate graph va reasoning text.

2. PrimeKG khong cover het loai cau hoi.
   - Anatomy
   - Physiology
   - Definitional / foundational facts
   - Mo ta cau truc
   - Claim moi, guideline moi, evidence moi

3. Mapping hien tai dang qua graph-centric.
   - Neu claim dung nhung khong co path dep trong PrimeKG thi ket qua de roi vao `gold_mapping=[]`, `silver`, `reject`, hoac skip toan bo sample.
   - Truong hop smoke test loai het 1000/1000 candidates la dau hieu logic relation/path mapping hien tai chua hop ly cho du lieu raw.

4. Auditor va provenance contract hien tai dang thien ve `edge_id` va `PMID`.
   - Neu sau nay co textbook retrieval ma schema van chi chap nhan PrimeKG/PubMed thi textbook evidence van bi phat unsupported.

## 3. Dinh huong kien truc moi

Khong nen de textbook/PubMed chi la fallback muon sau khi graph fail hoan toan. Nen doi sang retrieval 3 lane chay song song:

1. Graph lane
   - PrimeKG/UMLS
   - Dung cho drug interaction, contraindication, dosage, mechanism, disease association, relation-heavy reasoning

2. Literature lane
   - PubMed
   - Dung cho claim-level entailment, temporal update, citation, evidence moi

3. Textbook lane
   - textbook chunks co metadata
   - Dung cho anatomy, physiology, foundational fact, definitional knowledge, structure/location questions

Sau retrieval, can co mot `router + merger` de quyet dinh lane nao la nguon bang chung chinh cho tung cau hoi va cho tung claim.

## 4. Muc tieu kien truc sau khi nang cap

Sau khi nang cap, he thong can dat 5 tinh chat:

1. Khong fail cung vi graph khong cover.
2. Cho phep claim duoc support boi `edge`, `PMID`, hoac `TEXTBOOK:*`.
3. Khong de textbook override safety-critical graph/PubMed evidence.
4. Van giu provenance ro rang cho final answer.
5. Trien khai duoc tang dan tren Kaggle ma khong rewrite toan bo pipeline.

## 5. Evidence routing can co

Ban dau co the dung rule-based router nhe, chua can train classifier rieng.

### Route labels toi thieu

- `graph_pubmed`
- `textbook_pubmed`
- `graph_pubmed_textbook`
- `pubmed_first`

### Rule ban dau

1. Neu cau hoi co pattern anatomy / structure / located in / made up of / except / layer / fascia / nerve / vessel:
   - route = `textbook_pubmed`

2. Neu cau hoi co drug names, dosage, contraindication, interaction, coadministration:
   - route = `graph_pubmed`

3. Neu cau hoi la mechanism / etiology / disease association:
   - route = `graph_pubmed_textbook` hoac `graph_pubmed`

4. Neu cau hoi la guideline, temporal update, trial result, recommendation moi:
   - route = `pubmed_first`

5. Neu graph coverage sau retrieval duoi threshold:
   - nang trong so textbook/PubMed thay vi reject som.

## 6. EvidenceBundle v2 de xuat

Schema muc tieu:

```json
{
  "question_text": "...",
  "question_type": "...",
  "question_entities": [],
  "graph_edges": [],
  "pubmed_passages": [],
  "textbook_passages": [],
  "retrieval_metadata": {
    "route": "graph_pubmed_textbook",
    "graph_conf": 0.0,
    "pubmed_conf": 0.0,
    "textbook_conf": 0.0,
    "graph_coverage": 0.0,
    "coverage_gap_reason": null
  }
}
```

### Textbook passage metadata toi thieu

- `source_id`
- `book_title`
- `edition`
- `chapter`
- `section`
- `page_start`
- `page_end`
- `subject`
- `evidence_strength`
- `text`
- `retrieval_score`

## 7. Provenance format moi

Final answer va auditor khong nen ep moi claim phai co PrimeKG edge.

Can mo rong provenance thanh:

- `edge:E_12345`
- `PMID:12345678`
- `TEXTBOOK:book_slug/chapter_slug/section_slug#p120-p122`

### Nguyen tac

1. Drug interaction / contraindication / dosage:
   - textbook mot minh khong du
   - can graph hoac PubMed/guideline support

2. Anatomy / physiology / definition:
   - textbook support manh la hop le
   - graph la optional

3. Neu textbook va PubMed conflict:
   - therapeutic / safety claim: uu tien PubMed/guideline moi hon
   - foundational anatomy/fact: uu tien textbook chuan, tru khi co contradictory consensus source ro rang

## 8. Phases thuc thi

## Phase 0 - Audit va instrumentation truoc khi sua logic

Muc tieu:
- Xac dinh tai sao 1000/1000 candidate bi loai.
- Tach failure do graph coverage, failure do relation filter, failure do edge mapping, failure do quality gate.

Can lam:
- Log them trong `prepare_medreason_seed.py` va `seed_quality.py`
- Tong hop cac ly do:
  - `no_gold_edges_mapped`
  - `missing_edge_mapping`
  - `missing_due_to_ppr_pruning`
  - `entity_linker_backend:*`
  - `edge_mapper_backend:*`
  - `primekg_backend:*`
- Lay mau 50-100 record anatomy/definition de review tay.

Tieu chi xong:
- Co bang thong ke failure reasons.
- Co nhom cau hoi graph-missing duoc tach rieng.

## Phase 1 - Textbook fallback toi thieu, it pha pipeline nhat

Muc tieu:
- Giu nguyen graph-first pipeline.
- Them textbook retrieval fallback khi graph coverage thap hoac route la anatomy/definition.

Can lam:
- Them `question_router`
- Them `textbook_retriever`
- Khong can sua TRM ngay
- Khi `gold_mapping` rong va route la textbook-heavy:
  - khong reject som
  - luu them textbook passages vao evidence
  - danh dau sample la `graph_gap_text_supported`

Tac dong code du kien:
- `src/layers/layer1_retrieval.py`
- `src/schemas.py`
- co the them `src/retrieval/textbook_retriever.py`

Tieu chi xong:
- Cac cau anatomy/definition co evidence thay vi empty evidence.
- Khong con chet oan chi vi PrimeKG khong cover.

## Phase 2 - EvidenceBundle v2 + provenance v2

Muc tieu:
- Bien textbook thanh citizen chinh thuc trong pipeline.

Can lam:
- Mo rong `EvidenceBundle`
- Them `TextbookPassage` schema
- Mo rong provenance schema va cac validator
- Cho phep synthesis va audit hieu `TEXTBOOK:*`

Tac dong code du kien:
- `src/schemas.py`
- `src/layers/layer5_synthesis.py`
- `schemas/posthoc_auditor_schema.py`
- `judges/posthoc_auditor.py`

Tieu chi xong:
- Final answer co the cite textbook section/page.
- Claim supported boi textbook khong bi danh dau unsupported chi vi thieu edge id.

## Phase 3 - Sua auditor theo multi-source evidence

Muc tieu:
- Auditor khong over-penalize textbook-heavy claims.

Can lam:
- Them textbook passages vao context cua judges
- Phan loai claim-level support theo source type:
  - graph-supported
  - pubmed-supported
  - textbook-supported
  - unsupported
- Chi bat H5 khi claim la safety-critical that su
- Khong dung unsupported anatomy statement de trigger abstention qua muc

Tac dong code du kien:
- `src/layers/layer4_judges.py`
- `judges/posthoc_auditor.py`
- `prompts/posthoc_auditor_system.txt`

Tieu chi xong:
- Anatomy/textbook facts co the nhan verdict supported.
- Safety-critical claims van bi gate chat.

## Phase 4 - Sua logic seed building va supervision cho TRM

Muc tieu:
- Khong ep moi sample MedReason phai map thanh PrimeKG path dep.
- Phan biet ro graph-supervised va text-supported samples.

Can lam:
- Bo sung metadata cho seed:
  - `support_mode = graph_only | text_only | hybrid`
  - `graph_gap_reason`
  - `text_support_sources`
- Với sample graph-missing:
  - khong nhat thiet dua vao TRM path supervision ngay
  - co the giu cho eval/judge/synthesis
- Với sample hybrid:
  - giu graph path neu co
  - kem text evidence de judge/synthesis dung

Quyet dinh quan trong:
- TRM nen tiep tuc hoc relation/path-heavy cases.
- Khong nen ep TRM hoc textbook-only facts theo cung label contract nhu graph path.

Tac dong code du kien:
- `src/training/medreason_adapter.py`
- `src/training/seed_quality.py`
- `src/training/trm_dataset_builder.py`

Tieu chi xong:
- Seed builder khong loai sach nhung sample co gia tri chi vi graph khong cover.
- Dataset TRM sach hon, dung muc tieu hon.

## Phase 5 - Scoring hop nhat va conflict resolution

Muc tieu:
- Co quy tac hop nhat bang chung theo loai cau hoi va loai claim.

Can lam:
- Tinh lane score rieng:
  - `graph_support_score`
  - `pubmed_support_score`
  - `textbook_support_score`
- Them policy:
  - dosage/DDI/contraindication: graph hoac PubMed/guideline phai support
  - anatomy/definition: textbook support la du
  - temporal claim: PubMed/guideline moi duoc uu tien

Tac dong code du kien:
- `src/layers/layer4_judges.py`
- `src/layers/layer5_synthesis.py`
- `src/orchestrator.py`

Tieu chi xong:
- He thong abstain thong minh hon.
- Khong danh dong textbook fact va safety claim vao cung mot rule.

## Phase 6 - Dataset, eval, va rollout Kaggle

Muc tieu:
- Dua thay doi vao smoke path va benchmark path ma khong vo pipeline.

Can lam:
- Tao textbook mini-index local/Kaggle
- Them smoke set cho 3 nhom:
  - graph-heavy
  - textbook-heavy
  - graph-missing but pubmed-supported
- Bao cao metrics theo route
- Track:
  - graph coverage
  - textbook rescue rate
  - unsupported rate sau fusion
  - abstention rate sau fusion

Tieu chi xong:
- Co smoke test cho evidence-fusion.
- Co so lieu de quyet tiep phase nao can uu tien.

## 9. Thu tu uu tien de thuc thi

Thu tu toi uu de it vo nhat:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 6
7. Phase 5

Ly do:
- Can do va phan loai failure truoc.
- Can textbook fallback som de tranh reject oan.
- Can schema/provenance truoc khi auditor va synthesis dung duoc textbook.
- TRM supervision nen sua sau khi da ro graph/text split, khong nen va ngay.

## 10. Ket luan thuc thi

Huong dung la nang MiniMed Prime tu `KG-first` thanh `evidence-fusion`.

Y nghia thuc te:
- PrimeKG van la lane rat quan trong cho reasoning relation-heavy va safety-critical.
- PubMed van la lane cho citation va temporal evidence.
- Textbook phai tro thanh lane chinh thuc cho foundational knowledge.
- LLM khong nen chi dung de judge; can duoc dat dung vi tri trong router, mapping, auditor, va synthesis, nhung khong duoc la dieu kien duy nhat de sample song sot.

Khuyen nghi thuc thi ngay:
- Lam Phase 0 + Phase 1 truoc.
- Chua dong vao TRM training cho den khi co du lieu cho thay graph-gap chiem bao nhieu va nhom nao can textbook rescue.
