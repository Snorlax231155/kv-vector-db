# MedMemory: Per-Patient Clinical Data Directory

## Architectural & Security Overview
In healthcare information systems, storing all patient records in a single monolithic file
violates the Principle of Least Privilege and introduces severe privacy risks (PHI blast radius).

This directory implements a **Tenant-Isolated Per-Patient Directory Model**:
- Every patient has an independent isolated folder: `data/patients/{patient_id}/`
- Formatted as plain text (`.txt`) for human readability, clinician review, and auditability.
- Enables OS-level POSIX access controls (`chmod 700 / chmod 600`), per-patient encryption,
  and individual chart export without exposing other patients.

## Folder Contents per Patient
Each patient directory contains:
- `chart.txt`: The complete, unified clinical chart (demographics, problem list, medications, labs, encounters, progress notes).
- `demographics.txt`: Identity, age, biological sex, date of birth, and reference dates.
- `conditions.txt`: Documented ICD-10-CM chronic conditions, onset dates, and status.
- `medications.txt`: Active medication regimen and discontinued medications with indications.
- `allergies.txt`: Documented drug and environmental allergies.
- `labs.txt`: Complete longitudinal laboratory results with values, units, and reference flags.
- `encounters.txt`: Clinical visit history and consultation logs.
- `notes/`: Directory containing each clinical progress note as an individual `.txt` document.
