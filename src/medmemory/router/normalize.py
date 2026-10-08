"""Synonym and abbreviation normalization.

Clinical text is full of abbreviations that are ambiguous out of context ('MI', 'AF', 'K').
Rules here are conservative: short, ambiguous abbreviations only expand when written in
upper case ('MI' yes, 'mi' as in 'mi casa' no); unambiguous ones ('HbA1c') expand in any case.
Every applied rule is reported so the Inspector can show what the router actually saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    replacement: str
    label: str


def _r(
    pattern: str, replacement: str, case_sensitive: bool = False, label: str | None = None
) -> Rule:
    flags = 0 if case_sensitive else re.IGNORECASE
    return Rule(re.compile(pattern, flags), replacement, label or pattern.strip("\\b()?:"))


RULES: tuple[Rule, ...] = (
    # labs
    _r(r"\bhb\s?a1c\b|\ba1c\b", "hemoglobin a1c", label="HbA1c/A1c"),
    _r(r"\bglyc(?:osyl)?ated\s+ha?emoglobin\b", "hemoglobin a1c", label="glycated hemoglobin"),
    _r(r"\bLDL(?:-C)?\b(?!\s*cholesterol)", "ldl cholesterol", True, "LDL"),
    _r(r"\be?GFR\b", "egfr", True, "eGFR/GFR"),
    _r(r"\bUACR\b", "urine albumin-creatinine ratio", True, "UACR"),
    _r(r"\bSBP\b", "systolic blood pressure", True, "SBP"),
    _r(r"\bDBP\b", "diastolic blood pressure", True, "DBP"),
    _r(r"\bBP\b", "blood pressure", True, "BP"),
    _r(r"\bK\+", "potassium", True, "K+"),
    _r(r"\bCr\b", "creatinine", True, "Cr"),
    _r(r"\bblood sugar\b", "blood glucose", label="blood sugar"),
    _r(r"\bHgb\b|\bHb\b", "hemoglobin", True, "Hgb"),
    # conditions
    _r(r"\bT2DM\b|\bDM2\b|\bT2D\b|\btype\s+(?:2|ii|two)\s+DM\b", "type 2 diabetes", label="T2DM"),
    _r(r"\bDM\b", "diabetes", True, "DM"),
    _r(r"\bHTN\b", "hypertension", label="HTN"),
    _r(r"\bhigh blood pressure\b", "hypertension", label="high blood pressure"),
    _r(
        r"\bhyperlipid(?:a)?emia\b|\bHLD\b|\bdyslipid(?:a)?emia\b",
        "hyperlipidemia",
        label="hyperlipidemia",
    ),
    _r(r"\bhigh cholesterol\b", "hyperlipidemia", label="high cholesterol"),
    _r(r"\bCKD\b", "chronic kidney disease", label="CKD"),
    _r(r"\bCAD\b", "coronary artery disease", True, "CAD"),
    _r(r"\bCHF\b", "heart failure", label="CHF"),
    _r(r"\bHF\b", "heart failure", True, "HF"),
    _r(r"\bA-?fib\b|\bAF\b", "atrial fibrillation", label="AFib/AF"),
    _r(r"\bMI\b", "myocardial infarction", True, "MI"),
    _r(r"\bheart attack\b", "myocardial infarction", label="heart attack"),
    _r(r"\bOA\b", "osteoarthritis", True, "OA"),
    _r(r"\bOSA\b", "obstructive sleep apnea", True, "OSA"),
    _r(r"\bMDD\b", "major depressive disorder", True, "MDD"),
    _r(r"\bGAD\b", "generalized anxiety disorder", True, "GAD"),
    _r(r"\bacid reflux\b|\bGORD\b", "gerd", label="reflux"),
    _r(r"\bCVA\b", "stroke", True, "CVA"),
    _r(r"\bSOB\b", "shortness of breath", True, "SOB"),
    _r(r"\bCP\b", "chest pain", True, "CP"),
    # drug classes
    _r(r"\bNSAIDs?\b", "nsaids", label="NSAID"),
    _r(r"\bACE[- ]?i\b|\bACE inhibitors?\b", "ace inhibitor", label="ACEi"),
    _r(r"\bARBs?\b", "angiotensin receptor blocker", True, "ARB"),
    _r(r"\bSGLT-?2i?\b", "sglt2 inhibitor", label="SGLT2i"),
    _r(r"\bPPIs?\b", "proton pump inhibitor", True, "PPI"),
    _r(r"\bSSRIs?\b", "ssri", True, "SSRI"),
    _r(r"\bblood thinners?\b", "anticoagulant", label="blood thinner"),
    # shorthand
    _r(r"\bHx\b", "history", label="Hx"),
    _r(r"\bDx\b", "diagnosis", label="Dx"),
    _r(r"\bmeds\b", "medications", label="meds"),
    _r(r"\bpt\b", "patient", label="pt"),
)


def normalize(query: str) -> tuple[str, list[str]]:
    """Return (normalized lower-case query, list of 'surface → expansion' notes)."""
    text = " ".join(query.split())
    applied: list[str] = []
    for rule in RULES:
        new, n = rule.pattern.subn(rule.replacement, text)
        if n:
            surface = rule.pattern.search(text)
            applied.append(f"{surface.group(0) if surface else rule.label} → {rule.replacement}")
            text = new
    return text.lower(), applied
