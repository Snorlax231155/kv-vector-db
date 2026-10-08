"""Export patient records into isolated per-patient directories containing readable .txt files.

Healthcare architecture rationale:
- HIPAA Security Rule / Least Privilege: Monolithic files (e.g. patients.jsonl) expose
  the entire clinical population to anyone with file-read access, creating a large PHI blast radius.
- Directory Isolation: Storing each patient in a dedicated directory (data/patients/P0001/)
  enables granular POSIX permission controls (chmod 700 / ACLs), isolated audit logging,
  and export of individual patient charts without exposing other patients.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any


def _calc_age(birth_date_str: str, ref_date_str: str | None = None) -> int:
    born = date.fromisoformat(birth_date_str)
    ref = date.fromisoformat(ref_date_str) if ref_date_str else date.today()
    return ref.year - born.year - ((ref.month, ref.day) < (born.month, born.day))


def _slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_")


def export_patient(
    p: dict[str, Any], notes: list[dict[str, Any]], out_dir: Path
) -> None:
    pid = p["patient_id"]
    name = p["name"]
    p_dir = out_dir / pid
    p_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = p_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    enc_dates = [e["date"] for e in p.get("encounters", []) if "date" in e]
    last_enc = max(enc_dates) if enc_dates else "None"
    age = _calc_age(p["birth_date"], last_enc if last_enc != "None" else None)

    # 1. demographics.txt
    demog_txt = (
        f"PATIENT IDENTIFICATION & DEMOGRAPHICS\n"
        f"====================================\n"
        f"Patient ID:      {pid}\n"
        f"Full Name:       {name}\n"
        f"Date of Birth:   {p['birth_date']}\n"
        f"Age:             {age} years old\n"
        f"Biological Sex:  {p['sex'].capitalize()}\n"
        f"Last Encounter:  {last_enc}\n"
    )
    (p_dir / "demographics.txt").write_text(demog_txt, encoding="utf-8")

    # 2. conditions.txt
    cond_lines = [
        "DOCUMENTED CONDITIONS & PROBLEM LIST",
        "====================================",
    ]
    if p.get("conditions"):
        for c in p["conditions"]:
            cond_lines.append(
                f"- Diagnosis:   {c['name']}\n"
                f"  ICD-10-CM:   {c['icd10']}\n"
                f"  Status:      {c['status']}\n"
                f"  Onset Date:  {c['onset']}\n"
            )
    else:
        cond_lines.append("No chronic medical conditions documented.\n")
    (p_dir / "conditions.txt").write_text("\n".join(cond_lines), encoding="utf-8")

    # 3. medications.txt
    med_lines = [
        "PHARMACOTHERAPY & MEDICATION REGIMEN",
        "====================================",
    ]
    active_meds = [m for m in p.get("medications", []) if m.get("status") == "active"]
    past_meds = [m for m in p.get("medications", []) if m.get("status") != "active"]

    med_lines.append("ACTIVE MEDICATIONS:")
    if active_meds:
        for m in active_meds:
            med_lines.append(
                f"- {m['drug'].capitalize()} {m.get('dose', '')} ({m.get('frequency', '')})\n"
                f"  Indication:  {m.get('indication', 'N/A')}\n"
                f"  RxCUI:       {m.get('rxcui', 'N/A')}\n"
                f"  Start Date:  {m.get('start', 'N/A')}\n"
                f"  Status:      Active\n"
            )
    else:
        med_lines.append("  None active.\n")

    med_lines.append("DISCONTINUED / PAST MEDICATIONS:")
    if past_meds:
        for m in past_meds:
            med_lines.append(
                f"- {m['drug'].capitalize()} {m.get('dose', '')} ({m.get('frequency', '')})\n"
                f"  Indication:  {m.get('indication', 'N/A')}\n"
                f"  Start Date:  {m.get('start', 'N/A')}\n"
                f"  Stop Date:   {m.get('stop', 'N/A')}\n"
                f"  Stop Reason: {m.get('stop_reason', 'N/A')}\n"
                f"  Status:      Discontinued\n"
            )
    else:
        med_lines.append("  None documented.\n")
    (p_dir / "medications.txt").write_text("\n".join(med_lines), encoding="utf-8")

    # 4. allergies.txt
    allergies = p.get("allergies", [])
    allergy_lines = [
        "ALLERGIES & ADVERSE REACTIONS",
        "=============================",
    ]
    if allergies:
        for a in allergies:
            allergy_lines.append(
                f"- Substance: {a.get('substance', 'Unknown')}\n"
                f"  Reaction:  {a.get('reaction', 'Unknown')}\n"
                f"  Severity:  {a.get('severity', 'Unknown')}\n"
            )
    else:
        allergy_lines.append("No known drug allergies (NKDA) documented.\n")
    (p_dir / "allergies.txt").write_text("\n".join(allergy_lines), encoding="utf-8")

    # 5. labs.txt
    labs = p.get("labs", [])
    lab_lines = [
        "LONGITUDINAL LABORATORY TEST RESULTS",
        "====================================",
        f"{'Date':<12} | {'Test Name':<28} | {'Value':<10} | {'Unit':<15} | {'Flag'}",
        "-" * 78,
    ]
    if labs:
        for l in sorted(labs, key=lambda x: x.get("date", ""), reverse=True):
            val_str = str(l.get("value", ""))
            unit_str = l.get("unit", "")
            lab_lines.append(
                f"{l.get('date', 'N/A'):<12} | {l.get('lab', 'N/A'):<28} | {val_str:<10} | {unit_str:<15} | {l.get('flag', 'normal')}"
            )
    else:
        lab_lines.append("No laboratory records documented.")
    (p_dir / "labs.txt").write_text("\n".join(lab_lines) + "\n", encoding="utf-8")

    # 6. encounters.txt
    encounters = p.get("encounters", [])
    enc_lines = [
        "CLINICAL ENCOUNTERS & CONSULTATIONS",
        "===================================",
    ]
    if encounters:
        for e in sorted(encounters, key=lambda x: x.get("date", ""), reverse=True):
            enc_lines.append(
                f"- Date:      {e.get('date', 'N/A')} [{e.get('encounter_id', 'N/A')}]\n"
                f"  Type:      {e.get('type', 'N/A')}\n"
                f"  Reason:    {e.get('reason', 'N/A')}\n"
            )
    else:
        enc_lines.append("No encounter records documented.\n")
    (p_dir / "encounters.txt").write_text("\n".join(enc_lines), encoding="utf-8")

    # 7. notes/ (individual text files)
    patient_notes = [n for n in notes if n.get("patient_id") == pid]
    patient_notes.sort(key=lambda x: x.get("date", ""))
    for idx, n in enumerate(patient_notes, 1):
        note_date = n.get("date", "undated")
        enc_id = n.get("encounter_id", f"E{idx}")
        n_type = _slug(n.get("type", "clinical_note"))
        filename = f"{note_date}_{enc_id}_{n_type}.txt"
        (notes_dir / filename).write_text(n.get("text", "").strip() + "\n", encoding="utf-8")

    # 8. chart.txt (Comprehensive clinical record in a single structured text document)
    chart_lines = [
        "=" * 80,
        f"PATIENT MEDICAL CHART: {name.upper()} ({pid})",
        "=" * 80,
        "",
        demog_txt.strip(),
        "",
        "-" * 80,
        "\n".join(allergy_lines).strip(),
        "",
        "-" * 80,
        "\n".join(cond_lines).strip(),
        "",
        "-" * 80,
        "\n".join(med_lines).strip(),
        "",
        "-" * 80,
        "\n".join(lab_lines[:25]).strip(),
    ]
    if len(lab_lines) > 25:
        chart_lines.append(f"... [{len(lab_lines) - 25} earlier laboratory readings archived in labs.txt] ...")

    chart_lines.extend([
        "",
        "-" * 80,
        "\n".join(enc_lines).strip(),
        "",
        "-" * 80,
        "CLINICAL NARRATIVE PROGRESS NOTES",
        "=================================",
    ])
    if patient_notes:
        for idx, n in enumerate(patient_notes, 1):
            chart_lines.append(f"\n--- Note {idx}: {n.get('title', 'Clinical Note')} ---")
            chart_lines.append(n.get("text", "").strip())
    else:
        chart_lines.append("No narrative progress notes documented.\n")

    chart_lines.append("\n" + "=" * 80 + "\n[END OF CHART]\n")
    (p_dir / "chart.txt").write_text("\n".join(chart_lines), encoding="utf-8")


def export_all_patients(data_dir: Path | str = "data") -> dict[str, int]:
    base = Path(data_dir)
    seed = base / "seed"
    out = base / "patients"
    out.mkdir(parents=True, exist_ok=True)

    patients_file = seed / "patients.jsonl"
    notes_file = seed / "notes.jsonl"

    if not patients_file.exists():
        raise FileNotFoundError(f"Missing {patients_file}")

    patients = [
        json.loads(line)
        for line in patients_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    notes = (
        [
            json.loads(line)
            for line in notes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if notes_file.exists()
        else []
    )

    manifest_lines = [
        "PATIENT DIRECTORY MANIFEST & ACCESS REGISTER",
        "============================================",
        f"Total Patients: {len(patients)}",
        f"Total Notes:    {len(notes)}",
        f"Storage Mode:   Tenant-Isolated Per-Patient Directory Structure",
        "",
        f"{'ID':<8} | {'Name':<24} | {'Age/Sex':<12} | {'Conditions':<6} | {'Meds':<6} | {'Labs':<6} | {'Notes':<6} | {'Directory'}",
        "-" * 105,
    ]

    for p in patients:
        export_patient(p, notes, out)
        p_notes = [n for n in notes if n.get("patient_id") == p["patient_id"]]
        enc_dates = [e["date"] for e in p.get("encounters", []) if "date" in e]
        last_enc = max(enc_dates) if enc_dates else None
        age = _calc_age(p["birth_date"], last_enc)
        sex_label = f"{age}yo {p['sex'][:1].upper()}"
        manifest_lines.append(
            f"{p['patient_id']:<8} | {p['name']:<24} | {sex_label:<12} | "
            f"{len(p.get('conditions', [])):<6} | {len(p.get('medications', [])):<6} | "
            f"{len(p.get('labs', [])):<6} | {len(p_notes):<6} | data/patients/{p['patient_id']}/"
        )

    (out / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    readme_txt = (
        "# MedMemory: Per-Patient Clinical Data Directory\n\n"
        "## Architectural & Security Overview\n"
        "In healthcare information systems, storing all patient records in a single monolithic file\n"
        "violates the Principle of Least Privilege and introduces severe privacy risks (PHI blast radius).\n\n"
        "This directory implements a **Tenant-Isolated Per-Patient Directory Model**:\n"
        "- Every patient has an independent isolated folder: `data/patients/{patient_id}/`\n"
        "- Formatted as plain text (`.txt`) for human readability, clinician review, and auditability.\n"
        "- Enables OS-level POSIX access controls (`chmod 700 / chmod 600`), per-patient encryption,\n"
        "  and individual chart export without exposing other patients.\n\n"
        "## Folder Contents per Patient\n"
        "Each patient directory contains:\n"
        "- `chart.txt`: The complete, unified clinical chart (demographics, problem list, medications, labs, encounters, progress notes).\n"
        "- `demographics.txt`: Identity, age, biological sex, date of birth, and reference dates.\n"
        "- `conditions.txt`: Documented ICD-10-CM chronic conditions, onset dates, and status.\n"
        "- `medications.txt`: Active medication regimen and discontinued medications with indications.\n"
        "- `allergies.txt`: Documented drug and environmental allergies.\n"
        "- `labs.txt`: Complete longitudinal laboratory results with values, units, and reference flags.\n"
        "- `encounters.txt`: Clinical visit history and consultation logs.\n"
        "- `notes/`: Directory containing each clinical progress note as an individual `.txt` document.\n"
    )
    (out / "README.md").write_text(readme_txt, encoding="utf-8")

    return {"patients": len(patients), "notes": len(notes), "directories": len(patients)}


if __name__ == "__main__":
    res = export_all_patients()
    print(f"Exported {res['patients']} patients into data/patients/ (total {res['notes']} notes).")
