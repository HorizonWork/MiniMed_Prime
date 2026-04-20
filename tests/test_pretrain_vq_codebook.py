from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pretrain_vq_codebook_cli_dry_run_smoke(tmp_path: Path) -> None:
    primekg_dir = tmp_path / "primekg"
    primekg_dir.mkdir(parents=True, exist_ok=True)
    (primekg_dir / "nodes.csv").write_text(
        "\n".join(
            [
                "node_index,node_id,node_name,node_type,node_source",
                "1,drug:aspirin,Aspirin,drug,primekg",
                "2,disease:af,Atrial Fibrillation,disease,primekg",
                "3,protein:ptgs1,PTGS1,protein,primekg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (primekg_dir / "edges.csv").write_text(
        "\n".join(
            [
                "x_index,y_index,relation,display_relation,edge_id",
                "1,2,indication,is indicated for,E1",
                "1,3,drug_protein,targets,E2",
                "3,2,disease_protein,associated with,E3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "vq_artifact"
    command = [
        sys.executable,
        str(Path("tools") / "pretrain_vq_codebook.py"),
        "--output-dir",
        str(output_dir),
        "--dry-run",
        "4",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--codebook-size",
        "32",
        "--hidden-dim",
        "32",
        "--save-every",
        "2",
        "--primekg-path",
        str(primekg_dir),
        "--sapbert-path",
        str(tmp_path / "missing-sapbert"),
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"

    required_files = {
        "codebook.pt",
        "codebook_meta.json",
        "codebook_stats.json",
        "train_progress.json",
        "train_log.jsonl",
    }
    present_files = {path.name for path in output_dir.iterdir()}
    assert required_files.issubset(present_files)

    metadata = json.loads((output_dir / "codebook_meta.json").read_text(encoding="utf-8"))
    progress = json.loads((output_dir / "train_progress.json").read_text(encoding="utf-8"))

    assert metadata["codebook_size"] == 32
    assert metadata["hidden_dim"] == 32
    assert metadata["num_subgraphs_seen"] == 4
    assert progress["subgraphs_seen"] == 4
