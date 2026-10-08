"""Build the committed offline seed dataset under `data/seed/`.

    medmemory seed            # fetch public sources (cached) + generate patients
    medmemory seed --offline  # regenerate from cached HTTP responses only

Idempotent: identical inputs produce byte-identical files (sorted keys, fixed seeds), and the
manifest records a sha256 per file so CI can detect drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medmemory.ingest.catalog import CONDITIONS, DRUGS
from medmemory.ingest.patients import REF_DATE, generate
from medmemory.ingest.sources import icd10cm, medlineplus, openfda, rxnorm
from medmemory.ingest.sources.http import CachedFetcher

log = logging.getLogger(__name__)

LICENSES = {
    "openfda_labels": {
        "source": "openFDA drug label API (https://api.fda.gov/drug/label.json)",
        "license": "Public domain (US Government work); openFDA terms of service",
        "notes": "Only source of drug facts and doses. Sections clipped to 3,500 chars, marked [truncated].",
    },
    "medlineplus_topics": {
        "source": "MedlinePlus health topics web service (https://wsearch.nlm.nih.gov/ws/query)",
        "license": "Public domain (NLM-authored content only). A.D.A.M. and ASHP content excluded.",
        "attribution": medlineplus.ATTRIBUTION,
    },
    "icd10cm": {
        "source": "ICD-10-CM via NLM Clinical Tables API (https://clinicaltables.nlm.nih.gov)",
        "license": "Public domain (US CDC/CMS ICD-10-CM). Not WHO ICD-10.",
    },
    "rxnorm_ingredients": {
        "source": "NLM RxNav REST API (https://rxnav.nlm.nih.gov)",
        "license": "Ingredient-level RxCUIs; no UMLS licence required for this use.",
    },
    "synthetic_patients": {
        "source": "medmemory.ingest.patients (deterministic generator, seed 42)",
        "license": "MIT (this project). Entirely fictional; no real patient data.",
    },
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_seed(
    data_dir: Path, n_patients: int = 50, seed: int = 42, offline: bool = False
) -> dict[str, Any]:
    seed_dir = data_dir / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    fetcher = CachedFetcher(data_dir / "raw" / "http", offline=offline)
    try:
        # ---- drug labels + RxNorm ingredient IDs
        labels, rxcuis, missing_labels = [], {}, []
        for key, spec in sorted(DRUGS.items()):
            label = openfda.fetch_label(fetcher, spec.openfda_name, spec.route)
            rx = rxnorm.ingredient_rxcui(fetcher, key)
            if rx:
                rxcuis[key] = rx
            if label is None:
                missing_labels.append(key)
                log.warning("no openFDA label for %s", key)
                continue
            label["drug"] = key
            label["ingredient_rxcui"] = rx or ""
            labels.append(label)
        log.info("drug labels: %d (missing: %s)", len(labels), missing_labels)

        # ---- ICD-10-CM: every code under the categories we use, plus the exact codes
        icd_rows: dict[str, dict[str, str]] = {}
        for cspec in CONDITIONS.values():
            category = cspec.icd10.split(".")[0]
            for row in icd10cm.fetch_category(fetcher, category):
                icd_rows[row["code"]] = row
            exact = icd10cm.fetch_code(fetcher, cspec.icd10)
            if exact is None:
                raise RuntimeError(f"ICD-10-CM code not found upstream: {cspec.icd10}")
            icd_rows[exact["code"]] = exact
        icd_list = [icd_rows[c] for c in sorted(icd_rows)]

        # ---- MedlinePlus topics
        topics = []
        for term, slug in medlineplus.TOPICS:
            t = medlineplus.fetch_topic(fetcher, term, slug)
            if t is None:
                log.warning("no MedlinePlus topic for %s", slug)
                continue
            topics.append(t)
    finally:
        fetcher.close()

    # ---- synthetic patients + notes
    patients, notes, facts = generate(n_patients, seed, rxcuis)
    icd_names = {r["code"]: r["name"] for r in icd_list}
    for p in patients:
        for c in p["conditions"]:
            c["name"] = icd_names[c["icd10"]]

    files = {
        "drug_labels.jsonl": _write_jsonl(seed_dir / "drug_labels.jsonl", labels),
        "icd10cm.jsonl": _write_jsonl(seed_dir / "icd10cm.jsonl", icd_list),
        "guidelines.jsonl": _write_jsonl(seed_dir / "guidelines.jsonl", topics),
        "patients.jsonl": _write_jsonl(seed_dir / "patients.jsonl", patients),
        "notes.jsonl": _write_jsonl(seed_dir / "notes.jsonl", notes),
        "scenarios.jsonl": _write_jsonl(seed_dir / "scenarios.jsonl", facts),
    }
    manifest = {
        "dataset": "medmemory-seed",
        "version": f"{REF_DATE.isoformat()}-n{n_patients}-s{seed}",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "reference_date": REF_DATE.isoformat(),
        "sources": LICENSES,
        "counts": {
            "patients": len(patients),
            "notes": len(notes),
            "drug_labels": len(labels),
            "icd10cm_codes": len(icd_list),
            "guideline_topics": len(topics),
            "lab_observations": sum(len(p["labs"]) for p in patients),
            "medications": sum(len(p["medications"]) for p in patients),
            "scenario_facts": len(facts),
        },
        "missing_labels": missing_labels,
        "files": {name: {"sha256": digest} for name, digest in files.items()},
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def verify_manifest(data_dir: Path) -> list[str]:
    """Return a list of problems (empty == OK). Used by CI."""
    problems = []
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, meta in manifest["files"].items():
        path = data_dir / "seed" / name
        if not path.exists():
            problems.append(f"missing {name}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            problems.append(f"sha256 mismatch for {name}")
    for key, src in manifest["sources"].items():
        if not src.get("license"):
            problems.append(f"source {key} has no license field")
    return problems
