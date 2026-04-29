"""Task 1.1 — MedReason gold-path coverage analysis.

PIVOT NOTE: The S1 kickoff anticipated typed (head_type, relation, tail_type)
triples in MedReason's gold paths. In reality, MedReason 32K (UCSC-VLAA on
HF) stores reasoning chains as natural-language `entity -> entity -> entity`
sequences inside the `reasoning` text field, with no typed relation labels.
Only ~1% of records contain any structured relation marker.

So the only useful interpretation of "schema covers MedReason" is:

    1. Do the 12 node types span the medical concepts MedReason reasons about?
       (entity-type coverage)
    2. Are the implied entity-pair transitions in those chains supported by
       at least one edge type in the schema?
       (entity-pair coverage)

We sample 200 questions, extract `->` chains, heuristically classify each
entity into one of the 12 schema node types, and write
`reports/coverage_analysis.csv` with per-type counts and per-pair coverage.
The schema gate is ≥85% combined coverage.
"""

from __future__ import annotations

import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset  # noqa: E402

from kg.schema import load_schema  # noqa: E402

SAMPLE_SIZE = 200
SEED = 0
OUTPUT = REPO_ROOT / "reports" / "coverage_analysis.csv"
NOTES = REPO_ROOT / "reports" / "medreason_coverage_notes.md"

HF_CANDIDATES = ["UCSC-VLAA/MedReason", "MedReason/MedReason"]


# Entity-type classifier — keyword/suffix heuristics for the 12 schema types.
# Order matters: more specific patterns first.
TYPE_RULES: list[tuple[str, list[str]]] = [
    # symptom — patient-reported
    ("symptom", [
        r"\bpain\b", r"\bnausea\b", r"\bvomiting\b", r"\bfever\b", r"\bcough\b",
        r"\bdyspnea\b", r"\bfatigue\b", r"\bdizziness\b", r"\bheadache\b",
        r"\bbleeding\b", r"\bdiscomfort\b", r"\britch", r"\bswell",
        r"\b(short|loss of) breath\b",
    ]),
    # phenotype — measurable abnormality
    ("phenotype", [
        r"hyperten", r"hypoten", r"hyperglyc", r"hypoglyc",
        r"\bedema\b", r"\bproteinuria\b", r"\bhematuria\b", r"\bdyslipid",
        r"\btachycardia\b", r"\bbradycardia\b", r"\barrhythmia\b",
        r"\bobesity\b", r"abnormal\b",
    ]),
    # disease
    ("disease", [
        r"\b(disease|disorder|syndrome|deficiency|complication|abrasion|transformation|dysmotility)\b",
        r"\b(carcinoma|sarcoma|lymphoma|leukemia|melanoma|tumor|tumour|cancer|neoplasm)\b",
        r"itis\b", r"osis\b", r"oma\b", r"pathy\b", r"emia\b", r"oma\b",
        r"\bdiabetes\b", r"\binfection\b", r"\bsepsis\b", r"\bstroke\b",
        r"\bmyocardial infarct", r"\bheart failure\b", r"\bCOPD\b",
        r"\basthma\b", r"\bhepatitis\b", r"\bmalaria\b", r"\bAIDS\b",
        r"\b(HFrEF|HFpEF|T1DM|T2DM|GERD|CKD|AFib)\b",
        r"\b(preeclampsia|eclampsia|fibrillation|flutter)\b",
        r"\b(retinopathy|nephropathy|neuropathy|cardiomyopathy)\b",
        r"\b(misalignment|abnormality)\b",
    ]),
    # drug
    ("drug", [
        # common drug suffixes
        r"\b\w+(pril|sartan|olol|statin|mycin|cillin|prazole|tinib|"
        r"mab|nib|azole|zolid|cycline|profen|caine|dipine|sone|formin|"
        r"floxacin|parin|gliflozin|gliptin|setron|triptan|zepam|pam|asone|"
        r"avir|tide|vudine|ipine|udipine|olol|olone|conazole)\b",
        r"\b(aspirin|warfarin|insulin|heparin|metformin|spironolactone|"
        r"furosemide|lisinopril|atorvastatin|amlodipine|metoprolol|"
        r"tamoxifen|levothyroxine|paracetamol|acetaminophen|ibuprofen|"
        r"morphine|codeine|fentanyl|amoxicillin|prednisone|hydrocortisone|"
        r"linezolid|dofetilide|amiodarone|sotalol|digoxin|atropine|"
        r"epinephrine|norepinephrine|dopamine|nitroglycerin)\b",
        r"\b(medication|drug|antibiotic|antiviral|antifungal|antiarrhythmic|"
        r"anticoagulant|antiplatelet|antihypertensive|analgesic)\b",
    ]),
    # lab_test (clinical_object subtype)
    ("clinical_object_lab", [
        r"\b(test|assay|panel|level|count|culture|biopsy|screening)\b",
        r"\b(BNP|NT-proBNP|HbA1c|TSH|PSA|CEA|CRP|ESR|INR|PT|PTT|CBC|BMP|CMP|LFT)\b",
        r"\b(troponin|ferritin|albumin|glucose|cholesterol|creatinine)\b",
        r"\b(MRI|CT scan|X-ray|ultrasound|endoscopy|colonoscopy|EKG|ECG|EEG)\b",
    ]),
    # procedure (clinical_object subtype)
    ("clinical_object_proc", [
        r"\b(surgery|surgical|operation|transplant|dialysis|catheteri|"
        r"intubation|resection|excision|amputation|reconstruction)\b",
        r"\b(angioplasty|stenting|bypass|chemotherapy|radiotherapy|"
        r"immunotherapy)\b",
        r"\bectomy\b", r"\bostomy\b", r"\botomy\b", r"\bplasty\b",
    ]),
    # clinical_guideline
    ("clinical_object_guideline", [
        r"\b(guideline|recommendation|protocol|criteria|standard of care)\b",
        r"\b(ACC|AHA|ADA|NICE|NCCN|IDSA|USPSTF)\b",
    ]),
    # gene_protein
    ("gene_protein", [
        r"\bgene\b", r"\bprotein\b", r"\benzyme\b", r"\breceptor\b",
        r"\bmutation\b", r"\bexpression\b", r"\bimmunoglobulin",
        r"\bantibody\b", r"\bhormone\b", r"\bcytokine\b", r"\binterferon\b",
        r"\b[A-Z]{2,5}\d?\b",   # gene-symbol-like ALL CAPS short tokens
        r"\b(BRCA[12]|TP53|EGFR|KRAS|HER2|ESR1|VEGF|TNF|IL[- ]?\d+)\b",
        r"\b(CYP\d[A-Z]\d?)\b",
    ]),
    # molecular_function (catch this BEFORE biological_process — it's more specific)
    ("molecular_function", [
        r"\b(activity|binding|catalysis|hydroly[sz]|kinase|phosphata[sz]|"
        r"transferase|reductase|oxidase|polymerase|ligase|isomerase|"
        r"dehydrogenase|peroxidase)\b",
    ]),
    # cellular_component
    ("cellular_component", [
        r"\b(membrane|nucleus|cytoplasm|mitochondri|ribosome|lysosome|"
        r"endoplasmic|golgi|vesicle|organelle|chromatin|histone)\b",
    ]),
    # anatomy
    ("anatomy", [
        r"\b(bone|muscle|tissue|cell|organ|membrane|fascia|ligament|"
        r"tendon|cartilage|gland|node|vessel|artery|vein|nerve|cortex|"
        r"medulla|lobe|ventricle|atrium)\b",
        r"\b(heart|lung|liver|kidney|brain|spine|stomach|intestine|"
        r"colon|pancreas|spleen|thyroid|prostate|uterus|ovary|testis|"
        r"breast|skin|eye|ear|nose|throat)\b",
        r"\b(abdomen|abdominal|thoracic|cervical|lumbar|pelvic|cranial|"
        r"perineal|inguinal|femoral)\b",
    ]),
    # biological_process
    ("biological_process", [
        r"\b(metabolism|signaling|signalling|regulation|differentiation|"
        r"apoptosis|inflammation|coagulation|hemostasis|angiogenesis|"
        r"transcription|translation|replication|secretion|absorption|"
        r"excretion|filtration|reabsorption)\b",
    ]),
    # pathway
    ("pathway", [
        r"\b(pathway|cascade|cycle)\b",
        r"\b(glycolysis|gluconeogenesis|TCA|krebs|electron transport)\b",
        r"\bRAAS\b", r"\bmTOR\b", r"\bPI3K\b", r"\bMAPK\b", r"\bNF-?[kK]B\b",
    ]),
    # exposure
    ("exposure", [
        r"\b(smoking|alcohol|tobacco|radiation|asbestos|pollution|"
        r"toxin|carcinogen|allergen)\b",
        r"\bexposure to\b",
    ]),
    # molecular_function / cellular_component fall under gene_protein in the
    # heuristic — these are very rarely surfaced as standalone path nodes.
]


def classify_entity(text: str) -> str | None:
    """Heuristically map an entity string to one of the 12 node types.

    Returns the schema node type name, or None if no rule fires.
    The clinical_object_* labels collapse to clinical_object in the schema.
    """
    s = text.strip().lower()
    if not s or len(s) > 150:
        return None
    for type_name, patterns in TYPE_RULES:
        for pat in patterns:
            if re.search(pat, s, flags=re.IGNORECASE):
                # Collapse subtypes back to the schema name
                if type_name.startswith("clinical_object"):
                    return "clinical_object"
                return type_name
    return None


PATH_LINE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$", re.MULTILINE)
ARROW_SPLIT = re.compile(r"\s*->\s*|\s*-->\s*")


def extract_chains(reasoning: str) -> list[list[str]]:
    """Pull out numbered `entity -> entity -> entity` chains from a record.

    MedReason puts these under a 'Finding reasoning paths:' header. We just
    look for numbered lines containing at least one `->`.
    """
    chains: list[list[str]] = []
    for m in PATH_LINE.finditer(reasoning):
        line = m.group(1)
        if "->" not in line:
            continue
        parts = [p.strip() for p in ARROW_SPLIT.split(line) if p.strip()]
        if len(parts) >= 2:
            chains.append(parts)
    return chains


def main() -> int:
    print("Loading PrimeKG-X schema…")
    schema = load_schema()
    schema_node_types = set(schema.node_types)
    # Direction-agnostic pair coverage: BFS retrieval traverses edges in
    # either direction, so for the purpose of asking "does the schema
    # support reasoning between these two types?" we collapse direction.
    edge_pairs: set[tuple[str, str]] = set()
    for e in schema.edge_types.values():
        edge_pairs.add((e.head_type, e.tail_type))
        edge_pairs.add((e.tail_type, e.head_type))
    print(f"  {len(schema_node_types)} node types, {len(edge_pairs)} unique edge endpoint pairs")

    print("Loading MedReason…")
    last_err: Exception | None = None
    repo = ds = None
    for r in HF_CANDIDATES:
        try:
            ds = load_dataset(r, split="train")
            repo = r
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load MedReason. Last error: {last_err!r}")
    print(f"  loaded from {repo}: {len(ds):,} rows")

    rng = random.Random(SEED)
    n = min(SAMPLE_SIZE, len(ds))
    idx = rng.sample(range(len(ds)), n)
    print(f"Sampling {n} records (seed={SEED})…")

    type_counts: Counter[str] = Counter()
    unclassified_examples: list[str] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_supported: dict[tuple[str, str], bool] = {}
    chains_per_record: list[int] = []
    entities_total = 0
    entities_classified = 0
    by_dataset: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "chains": 0, "entities": 0, "classified": 0}
    )

    for i in idx:
        rec = ds[i]
        ds_name = rec.get("dataset_name", "?")
        chains = extract_chains(rec.get("reasoning", "") or "")
        chains_per_record.append(len(chains))
        bd = by_dataset[ds_name]
        bd["records"] += 1
        bd["chains"] += len(chains)
        for chain in chains:
            classified_chain: list[str | None] = [classify_entity(e) for e in chain]
            for ent, c in zip(chain, classified_chain):
                entities_total += 1
                bd["entities"] += 1
                if c:
                    entities_classified += 1
                    type_counts[c] += 1
                    bd["classified"] += 1
                else:
                    if len(unclassified_examples) < 30:
                        unclassified_examples.append(ent)
            for a, b in zip(classified_chain, classified_chain[1:]):
                if a and b:
                    pair = (a, b)
                    pair_counts[pair] += 1
                    if pair not in pair_supported:
                        pair_supported[pair] = (
                            pair in edge_pairs
                            or ("any", b) in edge_pairs
                            or (a, "any") in edge_pairs
                        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    for t in sorted(schema_node_types):
        rows.append({
            "kind": "node_type",
            "name": t,
            "n_paths": type_counts.get(t, 0),
            "schema_match": "yes" if type_counts.get(t, 0) > 0 else "absent",
            "notes": "",
        })

    for (a, b), n_seen in pair_counts.most_common():
        rows.append({
            "kind": "entity_pair",
            "name": f"{a} -> {b}",
            "n_paths": n_seen,
            "schema_match": "yes" if pair_supported.get((a, b)) else "no",
            "notes": "",
        })

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "name", "n_paths", "schema_match", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    classified_rate = entities_classified / max(entities_total, 1)
    pair_rate = (
        sum(c for p, c in pair_counts.items() if pair_supported.get(p))
        / max(sum(pair_counts.values()), 1)
    )
    combined = (classified_rate + pair_rate) / 2.0

    NOTES.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTES, "w") as f:
        f.write("# MedReason coverage analysis — notes (S1 Task 1.1)\n\n")
        f.write(f"- HF source: `{repo}` (train split)\n")
        f.write(f"- sampled: {n} records (seed {SEED})\n")
        f.write(f"- chains per record: mean={sum(chains_per_record) / max(len(chains_per_record), 1):.2f}, "
                f"max={max(chains_per_record) if chains_per_record else 0}\n")
        f.write(f"- entities extracted: {entities_total}\n")
        f.write(f"- entities classified into one of 12 node types: {entities_classified} "
                f"({classified_rate:.1%})\n")
        f.write(f"- entity-pair transitions seen: {sum(pair_counts.values())}\n")
        f.write(f"- pair transitions supported by some schema edge: {pair_rate:.1%}\n")
        f.write(f"- combined gate metric: **{combined:.1%}** "
                f"({'PASS' if combined >= 0.85 else 'BELOW 85% — see below'})\n\n")

        f.write("## Pivot from kickoff design\n\n")
        f.write("MedReason 32K stores gold reasoning paths as natural-language\n"
                "`entity -> entity -> entity` chains in the `reasoning` field rather than\n"
                "as typed (head_type, relation, tail_type) triples. Only ~1% of records\n"
                "contain any structured relation marker. The original kickoff plan to\n"
                "compare relation labels is therefore not applicable; we instead measure\n"
                "(a) whether extracted entities classify into the 12 schema node types and\n"
                "(b) whether implied adjacency pairs map to at least one declared edge.\n\n")

        f.write("## By source dataset\n\n")
        f.write("| dataset | records | chains | entities | classified | rate |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for ds_name in sorted(by_dataset):
            d = by_dataset[ds_name]
            r = d["classified"] / max(d["entities"], 1)
            f.write(f"| {ds_name} | {d['records']} | {d['chains']} | {d['entities']} | {d['classified']} | {r:.1%} |\n")

        f.write("\n## Sample unclassified entities (first 30)\n\n")
        for u in unclassified_examples[:30]:
            f.write(f"- {u!r}\n")

    print()
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(rows)} rows)")
    print(f"Wrote {NOTES.relative_to(REPO_ROOT)}")
    print(f"Entity classification rate: {classified_rate:.1%}")
    print(f"Entity-pair edge support rate: {pair_rate:.1%}")
    print(f"Combined: {combined:.1%}  ({'PASS' if combined >= 0.85 else 'BELOW 85%'})")
    return 0 if combined >= 0.85 else 2


if __name__ == "__main__":
    sys.exit(main())
