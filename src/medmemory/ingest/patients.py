"""Deterministic synthetic patients and section-structured clinical notes.

Why our own generator (and a Synthea importer alongside it): Synthea codes conditions in
SNOMED CT, which can't be mapped to ICD-10-CM without a UMLS licence, and its free-text
notes are short and templated. This generator produces ICD-10-CM-coded records plus
SOAP-style notes with enough lexical variety for semantic retrieval to be meaningful.
Real Synthea output can still be imported with `python -m medmemory synthea --fhir-dir ...`.

Everything is fictional. Each patient has an independent RNG stream keyed by
(seed, index), so adding patients never changes existing ones.

Deliberate test fixtures (recorded in scenarios.json for the eval gold set):
* medication switch: lisinopril stopped for dry cough, losartan started
* KV-vs-note dose conflicts: a note lists a different dose than the structured record
* negated findings in review of systems ("denies chest pain")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from medmemory.ingest.catalog import (
    ALLERGIES,
    BASE_LABS,
    CONDITIONS,
    DRUGS,
    FIRST_NAMES,
    LABS,
    LAST_NAMES,
    ConditionSpec,
)

REF_DATE = date(2026, 9, 30)

# Scripted demo personas: (sex, age, condition keys, forced drugs, storyline flags)
PERSONAS: dict[int, tuple[str, int, tuple[str, ...], dict[str, str], tuple[str, ...]]] = {
    1: (
        "female",
        64,
        ("type 2 diabetes", "hypertension", "hyperlipidemia", "chronic kidney disease stage 3"),
        {
            "type 2 diabetes": "metformin",
            "hypertension": "lisinopril",
            "hyperlipidemia": "atorvastatin",
            "chronic kidney disease stage 3": "losartan",
        },
        ("ace_cough_switch", "metformin_gi"),
    ),
    2: (
        "male",
        72,
        ("atrial fibrillation", "heart failure", "hypertension"),
        {
            "atrial fibrillation": "apixaban",
            "heart failure": "furosemide",
            "hypertension": "amlodipine",
        },
        ("amlodipine_edema",),
    ),
    3: (
        "female",
        29,
        ("asthma", "gerd", "migraine"),
        {"asthma": "albuterol", "gerd": "omeprazole", "migraine": "sumatriptan"},
        (),
    ),
    4: (
        "female",
        51,
        ("hypothyroidism", "major depressive disorder", "obesity"),
        {"hypothyroidism": "levothyroxine", "major depressive disorder": "sertraline"},
        ("dose_conflict",),
    ),
    5: (
        "male",
        68,
        ("copd", "coronary artery disease", "old myocardial infarction", "hyperlipidemia"),
        {
            "copd": "tiotropium",
            "coronary artery disease": "aspirin",
            "old myocardial infarction": "clopidogrel",
            "hyperlipidemia": "rosuvastatin",
        },
        ("statin_myalgia",),
    ),
    6: (
        "male",
        77,
        (
            "type 2 diabetes with diabetic chronic kidney disease",
            "chronic kidney disease stage 4",
            "heart failure",
        ),
        {
            "type 2 diabetes with diabetic chronic kidney disease": "insulin glargine",
            "chronic kidney disease stage 4": "furosemide",
            "heart failure": "carvedilol",
        },
        (),
    ),
    7: (
        "female",
        58,
        ("atrial fibrillation", "osteoporosis", "hypothyroidism"),
        {
            "atrial fibrillation": "warfarin",
            "osteoporosis": "alendronate",
            "hypothyroidism": "levothyroxine",
        },
        ("warfarin_bruising",),
    ),
}

NOTE_TYPES = {
    "diabetes": "Endocrinology follow-up note",
    "htn": "Primary care progress note",
    "lipids": "Primary care progress note",
    "ckd": "Nephrology consult note",
    "airway": "Pulmonology clinic note",
    "af": "Cardiology clinic note",
    "hf": "Cardiology clinic note",
    "cad": "Cardiology clinic note",
    "mi": "Cardiology clinic note",
    "thyroid": "Primary care progress note",
    "mood": "Behavioral health follow-up note",
    "anxiety": "Behavioral health follow-up note",
    "gi": "Primary care progress note",
    "oa": "Orthopedics clinic note",
    "weight": "Primary care progress note",
    "osa": "Sleep medicine note",
    "stroke": "Neurology follow-up note",
    "bone": "Primary care progress note",
    "heme": "Primary care progress note",
    "headache": "Neurology follow-up note",
}


@dataclass
class Ctx:
    rng: random.Random
    pid: str
    name: str
    sex: str
    age: int
    conditions: list[dict[str, Any]]
    meds: list[dict[str, Any]]
    allergies: list[dict[str, Any]]
    labs: list[dict[str, Any]] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)

    @property
    def he(self) -> str:
        return "she" if self.sex == "female" else "he"

    @property
    def his(self) -> str:
        return "her" if self.sex == "female" else "his"

    def has(self, key: str) -> bool:
        return any(c["key"] == key for c in self.conditions)

    def has_group(self, group: str) -> bool:
        return any(CONDITIONS[c["key"]].group == group for c in self.conditions)

    def med(self, drug: str) -> dict[str, Any] | None:
        for m in self.meds:
            if m["drug"] == drug and m["status"] == "active":
                return m
        return None

    def labs_on(self, d: str) -> dict[str, dict[str, Any]]:
        return {row["lab"]: row for row in self.labs if row["date"] == d}


def _rand_date(rng: random.Random, start: date, end: date) -> date:
    if end <= start:
        return start
    return start + timedelta(days=rng.randrange((end - start).days))


def _fmt(value: float, decimals: int) -> float | int:
    return round(value, decimals) if decimals else round(value)


def _pick_conditions(rng: random.Random, age: int) -> list[str]:
    n = rng.choices([1, 2, 3, 4], weights=[2, 4, 4, 2] if age > 50 else [5, 4, 2, 1])[0]
    chosen: list[str] = []
    groups: set[str] = set()
    pool = [c for c in CONDITIONS.values() if age >= c.min_age]
    while len(chosen) < n:
        weights = []
        for c in pool:
            w = c.weight
            if c.group in groups:
                w = 0.0
            if "diabetes" in groups and c.group in ("htn", "lipids", "ckd"):
                w *= 2.0
            if "af" in groups and c.group == "hf":
                w *= 2.0
            weights.append(w)
        if not any(weights):
            break
        spec = rng.choices(pool, weights=weights)[0]
        chosen.append(spec.key)
        groups.add(spec.group)
    return chosen


def _lab_value(
    rng: random.Random, lab: str, specs: list[ConditionSpec], ctx_meds: set[str]
) -> float | int:
    spec = LABS[lab]
    lo, hi = spec.normal
    for c in specs:
        if lab in c.lab_ranges:
            lo, hi = c.lab_ranges[lab]
    if lab == "inr":
        lo, hi = (1.8, 3.4) if "warfarin" in ctx_meds else (0.9, 1.1)
    return _fmt(rng.uniform(lo, hi), spec.decimals)


def _flag(lab: str, value: float) -> str:
    lo, hi = LABS[lab].normal
    if lab == "inr":
        return "normal"
    if value < lo:
        return "low"
    if value > hi:
        return "high"
    return "normal"


def build_patient(
    index: int, seed: int, rxcuis: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (patient record, notes, scenario facts)."""
    rng = random.Random(f"{seed}:{index}")
    pid = f"P{index:04d}"
    persona = PERSONAS.get(index)
    if persona:
        sex, age, cond_keys, forced, flags = persona
    else:
        sex = rng.choice(["female", "male"])
        age = int(min(89, max(19, rng.gauss(61, 15))))
        cond_keys = tuple(_pick_conditions(rng, age))
        forced, flags = {}, ()
    first = rng.choice(FIRST_NAMES[sex])
    last = rng.choice(LAST_NAMES)
    birth = REF_DATE - timedelta(days=age * 365 + rng.randrange(365))

    conditions = []
    for key in cond_keys:
        spec = CONDITIONS[key]
        onset = _rand_date(
            rng,
            max(birth + timedelta(days=spec.min_age * 365), REF_DATE - timedelta(days=365 * 15)),
            REF_DATE - timedelta(days=400),
        )
        conditions.append(
            {"key": key, "icd10": spec.icd10, "onset": onset.isoformat(), "status": "active"}
        )

    meds: list[dict[str, Any]] = []
    used: set[str] = set()
    for c in conditions:
        spec = CONDITIONS[c["key"]]
        if not spec.drugs:
            continue
        options = [d for d in spec.drugs if d not in used]
        if "chronic kidney disease stage 4" in cond_keys:
            options = [d for d in options if d not in ("metformin", "meloxicam", "ibuprofen")]
        if {"apixaban", "warfarin"} & used:
            options = [d for d in options if d not in ("apixaban", "warfarin")]
        if not options:
            continue
        drug = forced.get(c["key"]) or rng.choice(options)
        if drug in used:
            continue
        used.add(drug)
        dose, freq = rng.choice(DRUGS[drug].regimens)
        if drug == "metformin" and "metformin_gi" in flags:
            dose, freq = "500 mg", "twice daily"
        start = _rand_date(rng, date.fromisoformat(c["onset"]), REF_DATE - timedelta(days=200))
        meds.append(
            {
                "drug": drug,
                "rxcui": rxcuis.get(drug, ""),
                "dose": dose,
                "frequency": freq,
                "indication": c["key"],
                "start": start.isoformat(),
                "status": "active",
                "stop": None,
                "stop_reason": None,
            }
        )

    scenario_facts: list[dict[str, Any]] = []
    if "ace_cough_switch" in flags:
        lis = next(m for m in meds if m["drug"] == "lisinopril")
        stop = REF_DATE - timedelta(days=rng.randrange(240, 330))
        lis.update(status="stopped", stop=stop.isoformat(), stop_reason="persistent dry cough")
        if not any(m["drug"] == "losartan" for m in meds):
            meds.append(
                {
                    "drug": "losartan",
                    "rxcui": rxcuis.get("losartan", ""),
                    "dose": "50 mg",
                    "frequency": "once daily",
                    "indication": "hypertension",
                    "start": (stop + timedelta(days=3)).isoformat(),
                    "status": "active",
                    "stop": None,
                    "stop_reason": None,
                }
            )
        else:
            los = next(m for m in meds if m["drug"] == "losartan")
            los["indication"] = "hypertension; chronic kidney disease stage 3"
            los["start"] = (stop + timedelta(days=3)).isoformat()
        scenario_facts.append(
            {
                "type": "medication_switch",
                "patient_id": pid,
                "from": "lisinopril",
                "to": "losartan",
                "reason": "persistent dry cough",
                "date": stop.isoformat(),
            }
        )

    allergies = []
    if rng.random() < 0.45:
        for sub, reaction, sev in rng.sample(ALLERGIES, k=rng.choice([1, 1, 2])):
            if sub == "lisinopril" and "lisinopril" in used:
                continue
            if sub == "aspirin" and "aspirin" in used:
                continue
            allergies.append({"substance": sub, "reaction": reaction, "severity": sev})

    ctx = Ctx(rng, pid, f"{first} {last}", sex, age, conditions, meds, allergies, flags=set(flags))

    # ---- lab visits over the last two years
    visit_count = rng.choice([3, 4, 4, 5])
    visit_dates = sorted(
        {
            _rand_date(rng, REF_DATE - timedelta(days=720), REF_DATE - timedelta(days=7))
            for _ in range(visit_count)
        }
    )
    specs = [CONDITIONS[c["key"]] for c in conditions]
    active_drugs = {m["drug"] for m in meds}
    wanted_labs = list(BASE_LABS)
    for s in specs:
        for lab in (*s.labs, *s.lab_ranges.keys()):
            if lab not in wanted_labs:
                wanted_labs.append(lab)
    if active_drugs & {"warfarin"}:
        wanted_labs.append("inr")
    for vd in visit_dates:
        for lab in wanted_labs:
            if (
                lab
                in ("ldl cholesterol", "total cholesterol", "tsh", "urine albumin-creatinine ratio")
                and rng.random() < 0.35
            ):
                continue
            value = _lab_value(rng, lab, specs, active_drugs)
            ctx.labs.append(
                {
                    "lab": lab,
                    "value": value,
                    "unit": LABS[lab].unit,
                    "date": vd.isoformat(),
                    "flag": _flag(lab, float(value)),
                }
            )

    # consistency: eGFR and creatinine should move in opposite directions; keep simple by
    # deriving creatinine flag only. A1c trend for persona 1 improves over time.
    if index == 1:
        a1cs = [row for row in ctx.labs if row["lab"] == "hemoglobin a1c"]
        for i, row in enumerate(a1cs):
            row["value"] = round(8.9 - 0.4 * i, 1)
            row["flag"] = _flag("hemoglobin a1c", row["value"])

    encounters = []
    for i, vd in enumerate(visit_dates):
        focus = conditions[i % len(conditions)]
        encounters.append(
            {
                "encounter_id": f"E{index:04d}{i + 1:02d}",
                "date": vd.isoformat(),
                "type": NOTE_TYPES[CONDITIONS[focus["key"]].group],
                "reason": f"{focus['key']} follow-up",
            }
        )

    notes, conflict_facts = _build_notes(ctx, encounters)
    scenario_facts.extend(conflict_facts)

    record = {
        "patient_id": pid,
        "name": ctx.name,
        "sex": sex,
        "birth_date": birth.isoformat(),
        "conditions": [
            {**c, "name": None} for c in conditions
        ],  # name filled from ICD table at build
        "medications": meds,
        "allergies": allergies,
        "labs": ctx.labs,
        "encounters": encounters,
    }
    return record, notes, scenario_facts


# --------------------------------------------------------------------------- notes


def _latest(ctx: Ctx, lab: str, on_or_before: str) -> dict[str, Any] | None:
    rows = [row for row in ctx.labs if row["lab"] == lab and row["date"] <= on_or_before]
    return max(rows, key=lambda row: row["date"]) if rows else None


def _lab_phrase(row: dict[str, Any]) -> str:
    unit = f" {row['unit']}" if row["unit"] and row["unit"] != "%" else row["unit"]
    return f"{row['lab']} {row['value']}{unit}"


def _hpi(ctx: Ctx, cond_key: str, visit: str, visit_idx: int) -> list[str]:
    r = ctx.rng
    his = ctx.his
    out: list[str] = []
    if cond_key.startswith("type 2 diabetes"):
        a1c = _latest(ctx, "hemoglobin a1c", visit)
        out.append(
            r.choice(
                [
                    f"Presents for diabetes follow-up. Reports home fasting glucose readings mostly between {r.randrange(118, 140)} and {r.randrange(150, 195)} mg/dL.",
                    f"Here for routine diabetes care. Checks fingerstick glucose {r.choice(['daily', 'most mornings', 'a few times a week'])}; readings have been {r.choice(['variable', 'slightly above goal', 'improving'])}.",
                ]
            )
        )
        if a1c:
            out.append(f"Most recent {_lab_phrase(a1c)} on {a1c['date']}.")
        out.append(
            r.choice(
                [
                    "Reports mild tingling in both feet in the evenings, no foot ulcers.",
                    "Notes increased thirst and getting up twice a night to urinate.",
                    "Denies hypoglycemic episodes; admits to dietary lapses over the holidays.",
                    "Walking 20 minutes most days and has cut back on sweetened tea.",
                ]
            )
        )
        if "metformin_gi" in ctx.flags and visit_idx >= 1:
            out.append(
                "Tried increasing metformin to 1000 mg twice daily last spring but developed bloating and loose stools, so went back to 500 mg twice daily; symptoms resolved."
            )
    elif cond_key == "hypertension":
        sbp, dbp = (
            _latest(ctx, "systolic blood pressure", visit),
            _latest(ctx, "diastolic blood pressure", visit),
        )
        if sbp and dbp:
            out.append(f"Blood pressure today {sbp['value']}/{dbp['value']} mmHg.")
        out.append(
            r.choice(
                [
                    "Home blood pressure log shows readings in the 130s to 150s systolic.",
                    f"Reports adherence to {his} antihypertensive regimen and a low-salt diet most days.",
                    "Occasional morning headaches, no vision changes.",
                ]
            )
        )
        switched = next((m for m in ctx.meds if m["drug"] == "lisinopril" and m["stop"]), None)
        if "ace_cough_switch" in ctx.flags and switched and visit >= switched["stop"]:
            out.append(
                "Developed a persistent dry, tickly cough several weeks after starting lisinopril; cough is not productive and there is no fever. Lisinopril was stopped and losartan started, after which the cough resolved."
            )
        if "amlodipine_edema" in ctx.flags:
            out.append(
                "Noticed swelling around both ankles since the amlodipine dose was increased; worse at the end of the day, improves with leg elevation."
            )
    elif cond_key == "hyperlipidemia":
        ldl = _latest(ctx, "ldl cholesterol", visit)
        if ldl:
            out.append(f"Lipid panel reviewed: {_lab_phrase(ldl)} on {ldl['date']}.")
        if "statin_myalgia" in ctx.flags:
            out.append(
                "Reports aching in both thighs and calves after climbing stairs, started a few months after beginning a statin. No dark urine. CK was within normal limits."
            )
        else:
            out.append(
                r.choice(
                    [
                        "Tolerating statin therapy without muscle aches.",
                        "Trying to limit fried food and red meat.",
                    ]
                )
            )
    elif cond_key.startswith("chronic kidney disease"):
        egfr, cr = _latest(ctx, "egfr", visit), _latest(ctx, "creatinine", visit)
        if egfr and cr:
            out.append(
                f"Kidney function: {_lab_phrase(egfr)} and {_lab_phrase(cr)} on {egfr['date']}."
            )
        out.append(
            r.choice(
                [
                    "Counseled to avoid NSAIDs such as ibuprofen and naproxen given reduced kidney function.",
                    "Referred to nephrology for co-management; discussed limiting high-sodium processed foods.",
                    "No edema or change in urine output reported.",
                ]
            )
        )
    elif cond_key == "asthma":
        out.append(
            r.choice(
                [
                    "Reports nighttime cough about twice a week and uses the rescue inhaler three or four times a week.",
                    "Symptoms are triggered by cold air, exercise and exposure to cats.",
                ]
            )
        )
        out.append("No emergency visits or oral steroid courses in the past year.")
    elif cond_key == "copd":
        out.append(
            r.choice(
                [
                    "Becomes short of breath after one flight of stairs; uses rescue inhaler about once a day.",
                    "Chronic morning cough with clear sputum; former smoker with a 35 pack-year history, quit in 2015.",
                ]
            )
        )
    elif cond_key == "atrial fibrillation":
        out.append(
            r.choice(
                [
                    "Reports occasional episodes of fluttering in the chest lasting a few minutes, no fainting.",
                    "Rate controlled at today's visit; discussed stroke prevention and bleeding risk with anticoagulation.",
                ]
            )
        )
        if "warfarin_bruising" in ctx.flags:
            inr = _latest(ctx, "inr", visit)
            out.append(
                "Noticed easy bruising on both forearms after minor bumps; no nosebleeds, no blood in stool."
                + (f" INR {inr['value']} on {inr['date']}." if inr else "")
            )
    elif cond_key == "heart failure":
        bnp = _latest(ctx, "bnp", visit)
        out.append(
            r.choice(
                [
                    "Sleeps on two pillows to breathe comfortably and gained 3 pounds over the past week.",
                    "Mild shortness of breath walking to the mailbox; ankle swelling improves overnight.",
                ]
            )
        )
        if bnp:
            out.append(f"BNP {bnp['value']} pg/mL measured on {bnp['date']}.")
        out.append(
            "Advised daily weights and to call if weight increases more than 2 to 3 pounds in a day."
        )
    elif cond_key in ("coronary artery disease", "old myocardial infarction"):
        out.append(
            r.choice(
                [
                    "Stable exertional chest tightness when walking uphill, relieved within minutes by rest; no symptoms at rest.",
                    "History of myocardial infarction treated with a stent; no angina since cardiac rehabilitation.",
                ]
            )
        )
    elif cond_key == "hypothyroidism":
        tsh = _latest(ctx, "tsh", visit)
        out.append(
            r.choice(
                [
                    "Fatigue and cold intolerance have improved since levothyroxine was adjusted.",
                    "Takes levothyroxine on an empty stomach before breakfast, separate from calcium supplements.",
                ]
            )
        )
        if tsh:
            out.append(f"Most recent {_lab_phrase(tsh)} on {tsh['date']}.")
    elif cond_key == "major depressive disorder":
        out.append(
            r.choice(
                [
                    f"Mood is improving; PHQ-9 score decreased from {r.randrange(14, 19)} to {r.randrange(6, 11)}.",
                    "Sleep has improved, still has low energy in the afternoons; engaged in weekly therapy.",
                ]
            )
        )
        out.append("Denies thoughts of self-harm.")
    elif cond_key == "generalized anxiety disorder":
        out.append(
            r.choice(
                [
                    "Worry is less intrusive; using breathing exercises before work meetings.",
                    "GAD-7 score improved with therapy and medication.",
                ]
            )
        )
    elif cond_key == "gerd":
        out.append(
            r.choice(
                [
                    "Burning heartburn after large meals, worse when lying down at night.",
                    "Reflux symptoms well controlled on a proton pump inhibitor; avoids late-night eating.",
                ]
            )
        )
    elif cond_key == "knee osteoarthritis":
        out.append(
            r.choice(
                [
                    "Right knee pain worse with stairs and after prolonged standing; morning stiffness under 30 minutes.",
                    "Bilateral knee pain limits walking to about 15 minutes.",
                ]
            )
        )
    elif cond_key == "obesity":
        bmi = _latest(ctx, "bmi", visit)
        out.append(
            "Discussed nutrition and a gradual increase in physical activity."
            + (f" BMI {bmi['value']} kg/m2." if bmi else "")
        )
    elif cond_key == "obstructive sleep apnea":
        out.append(
            f"Uses CPAP about {r.randrange(4, 7)} hours per night; daytime sleepiness improved."
        )
    elif cond_key == "history of stroke":
        out.append(
            "Prior ischemic stroke without residual deficits; continues secondary prevention."
        )
    elif cond_key == "osteoporosis":
        out.append(
            f"DEXA scan showed a lumbar spine T-score of -{r.randrange(25, 32) / 10}. Takes calcium and vitamin D."
        )
    elif cond_key == "anemia":
        hgb = _latest(ctx, "hemoglobin", visit)
        out.append(
            "Reports fatigue and reduced exercise tolerance."
            + (f" {_lab_phrase(hgb).capitalize()} on {hgb['date']}." if hgb else "")
        )
    elif cond_key == "migraine":
        out.append(
            f"Has {r.randrange(2, 5)} migraines per month with light sensitivity and nausea; triptan helps within two hours."
        )
    return out


def _ros(ctx: Ctx) -> str:
    r = ctx.rng
    neg = r.sample(
        [
            "chest pain",
            "shortness of breath at rest",
            "fever",
            "palpitations",
            "abdominal pain",
            "headache",
            "leg swelling",
            "dizziness",
        ],
        k=3,
    )
    return f"Denies {neg[0]}, {neg[1]}, or {neg[2]}."


def _med_line(m: dict[str, Any], dose_override: str | None = None) -> str:
    return f"- {m['drug']} {dose_override or m['dose']} {m['frequency']}"


def _build_notes(
    ctx: Ctx, encounters: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    notes = []
    facts: list[dict[str, Any]] = []
    n_notes = min(len(encounters), ctx.rng.choice([3, 3, 4, 5]))
    chosen = encounters[-n_notes:]
    conflict_drug = None
    is_persona = int(ctx.pid[1:]) in PERSONAS
    if "dose_conflict" in ctx.flags or (not is_persona and ctx.rng.random() < 0.08 and ctx.meds):
        candidates = [
            m for m in ctx.meds if m["status"] == "active" and len(DRUGS[m["drug"]].regimens) > 1
        ]
        if candidates:
            conflict_drug = candidates[0]
    for idx, enc in enumerate(chosen):
        visit = enc["date"]
        focus_key = enc["reason"].removesuffix(" follow-up")
        others = [c["key"] for c in ctx.conditions if c["key"] != focus_key]
        sections: list[tuple[str, str]] = []
        sections.append(("CHIEF COMPLAINT", f"{focus_key.capitalize()} follow-up."))
        hpi = [f"{ctx.age}-year-old {ctx.sex} with {', '.join([focus_key, *others])}."]
        hpi += _hpi(ctx, focus_key, visit, idx)
        for other in others[:1]:
            hpi += _hpi(ctx, other, visit, idx)[:1]
        sections.append(("HISTORY OF PRESENT ILLNESS", " ".join(hpi)))
        sections.append(("REVIEW OF SYSTEMS", _ros(ctx)))
        active = [
            m for m in ctx.meds if m["start"] <= visit and (m["stop"] is None or m["stop"] > visit)
        ]
        med_lines = []
        for m in active:
            override = None
            if conflict_drug is not None and m is conflict_drug and idx == len(chosen) - 1:
                alternatives = [d for d, _ in DRUGS[m["drug"]].regimens if d != m["dose"]]
                override = alternatives[0]
                facts.append(
                    {
                        "type": "dose_conflict",
                        "patient_id": ctx.pid,
                        "drug": m["drug"],
                        "kv_dose": m["dose"],
                        "note_dose": override,
                        "doc_id": f"note:{ctx.pid}:{enc['encounter_id']}",
                    }
                )
            med_lines.append(_med_line(m, override))
        sections.append(("CURRENT MEDICATIONS", "\n".join(med_lines) if med_lines else "None."))
        allergy_text = (
            "; ".join(f"{a['substance']} ({a['reaction']})" for a in ctx.allergies)
            or "No known drug allergies."
        )
        sections.append(("ALLERGIES", allergy_text))
        todays = ctx.labs_on(visit)
        if todays:
            shown = [_lab_phrase(v) for k, v in todays.items() if k not in ("sodium",)][:7]
            sections.append(("OBJECTIVE", "Vitals and labs: " + "; ".join(shown) + "."))
        plan = [f"1. {focus_key.capitalize()}: {_plan(ctx, focus_key)}"]
        for i, other in enumerate(others[:2], start=2):
            plan.append(f"{i}. {other.capitalize()}: {_plan(ctx, other)}")
        sections.append(("ASSESSMENT AND PLAN", "\n".join(plan)))
        title = enc["type"]
        body = "\n\n".join(f"{h}:\n{t}" for h, t in sections)
        notes.append(
            {
                "doc_id": f"note:{ctx.pid}:{enc['encounter_id']}",
                "patient_id": ctx.pid,
                "encounter_id": enc["encounter_id"],
                "date": visit,
                "type": title,
                "title": f"{title} ({visit})",
                "text": f"{title.upper()}\nDate: {visit}\nPatient: {ctx.name} ({ctx.pid})\n\n{body}",
            }
        )
    return notes, facts


def _plan(ctx: Ctx, key: str) -> str:
    r = ctx.rng
    plans = {
        "type 2 diabetes": [
            "Continue current regimen; repeat hemoglobin A1c in 3 months; annual eye and foot exams.",
            "Reinforced diet and activity goals; review home glucose log at next visit.",
        ],
        "hypertension": [
            "Continue current antihypertensive; recheck blood pressure in 4 weeks.",
            "Home blood pressure monitoring; reduce sodium intake.",
        ],
        "hyperlipidemia": [
            "Continue statin; repeat lipid panel in 6 months.",
            "Lifestyle modification; recheck fasting lipids.",
        ],
        "asthma": [
            "Continue controller therapy; reviewed inhaler technique and asthma action plan."
        ],
        "copd": ["Continue long-acting bronchodilator; pulmonary rehabilitation referral."],
        "atrial fibrillation": [
            "Continue anticoagulation and rate control; discussed bleeding precautions."
        ],
        "heart failure": ["Daily weights, fluid and salt restriction; continue diuretic."],
        "hypothyroidism": ["Continue levothyroxine; repeat TSH in 6 to 8 weeks."],
        "major depressive disorder": ["Continue SSRI and therapy; safety plan reviewed."],
        "gerd": ["Continue proton pump inhibitor; elevate head of bed."],
        "knee osteoarthritis": ["Physical therapy; topical analgesic; weight management."],
    }
    for prefix, options in plans.items():
        if key.startswith(prefix):
            return r.choice(options)
    if key.startswith("chronic kidney disease"):
        return "Monitor renal function and potassium every 3 months; avoid nephrotoxic medications."
    return "Stable; continue current management and follow up in 3 months."


def generate(
    n: int, seed: int, rxcuis: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    patients, notes, facts = [], [], []
    for i in range(1, n + 1):
        p, ns, fs = build_patient(i, seed, rxcuis)
        patients.append(p)
        notes.extend(ns)
        facts.extend(fs)
    return patients, notes, facts
