"""Clinical catalogs used by the synthetic patient generator and the router gazetteers.

Everything here is either (a) an identifier/description that ingestion re-validates against
the public source (ICD-10-CM codes, RxNorm ingredients, openFDA labels), or (b) synthetic
patient-record content (what a fictional patient was prescribed). Drug *facts* such as
label dosing are never taken from this file; answers cite openFDA labels for those.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DrugSpec:
    key: str  # normalized generic name used everywhere in the system
    openfda_name: str  # exact openFDA generic_name (with salt) for label lookup
    regimens: tuple[tuple[str, str], ...]  # synthetic (dose, frequency) a patient might be on
    route: str = "oral"  # preferred label route when several formulations exist


DRUGS: dict[str, DrugSpec] = {
    d.key: d
    for d in [
        DrugSpec(
            "metformin",
            "metformin hydrochloride",
            (("500 mg", "twice daily"), ("1000 mg", "twice daily"), ("850 mg", "twice daily")),
        ),
        DrugSpec(
            "empagliflozin", "empagliflozin", (("10 mg", "once daily"), ("25 mg", "once daily"))
        ),
        DrugSpec(
            "glipizide",
            "glipizide",
            (("5 mg", "once daily before breakfast"), ("10 mg", "once daily before breakfast")),
        ),
        DrugSpec("sitagliptin", "sitagliptin", (("100 mg", "once daily"),)),
        DrugSpec(
            "insulin glargine",
            "insulin glargine",
            (("18 units", "at bedtime"), ("24 units", "at bedtime")),
            "subcutaneous",
        ),
        DrugSpec("lisinopril", "lisinopril", (("10 mg", "once daily"), ("20 mg", "once daily"))),
        DrugSpec(
            "losartan", "losartan potassium", (("50 mg", "once daily"), ("100 mg", "once daily"))
        ),
        DrugSpec(
            "amlodipine", "amlodipine besylate", (("5 mg", "once daily"), ("10 mg", "once daily"))
        ),
        DrugSpec(
            "hydrochlorothiazide",
            "hydrochlorothiazide",
            (("12.5 mg", "once daily"), ("25 mg", "once daily")),
        ),
        DrugSpec(
            "metoprolol", "metoprolol succinate", (("25 mg", "once daily"), ("50 mg", "once daily"))
        ),
        DrugSpec(
            "carvedilol", "carvedilol", (("6.25 mg", "twice daily"), ("12.5 mg", "twice daily"))
        ),
        DrugSpec(
            "atorvastatin",
            "atorvastatin calcium",
            (("20 mg", "once daily"), ("40 mg", "once daily")),
        ),
        DrugSpec(
            "rosuvastatin",
            "rosuvastatin calcium",
            (("10 mg", "once daily"), ("20 mg", "once daily")),
        ),
        DrugSpec("simvastatin", "simvastatin", (("20 mg", "at bedtime"), ("40 mg", "at bedtime"))),
        DrugSpec("furosemide", "furosemide", (("20 mg", "once daily"), ("40 mg", "once daily"))),
        DrugSpec("spironolactone", "spironolactone", (("25 mg", "once daily"),)),
        DrugSpec("apixaban", "apixaban", (("5 mg", "twice daily"), ("2.5 mg", "twice daily"))),
        DrugSpec(
            "warfarin",
            "warfarin sodium",
            (("5 mg", "once daily, adjusted to INR"), ("3 mg", "once daily, adjusted to INR")),
        ),
        DrugSpec("clopidogrel", "clopidogrel bisulfate", (("75 mg", "once daily"),)),
        DrugSpec("aspirin", "aspirin", (("81 mg", "once daily"),)),
        DrugSpec(
            "levothyroxine",
            "levothyroxine sodium",
            (
                ("50 mcg", "once daily before breakfast"),
                ("75 mcg", "once daily before breakfast"),
                ("100 mcg", "once daily before breakfast"),
            ),
        ),
        DrugSpec(
            "sertraline",
            "sertraline hydrochloride",
            (("50 mg", "once daily"), ("100 mg", "once daily")),
        ),
        DrugSpec("escitalopram", "escitalopram oxalate", (("10 mg", "once daily"),)),
        DrugSpec("buspirone", "buspirone hydrochloride", (("10 mg", "twice daily"),)),
        DrugSpec(
            "omeprazole",
            "omeprazole",
            (("20 mg", "once daily before breakfast"), ("40 mg", "once daily before breakfast")),
        ),
        DrugSpec("pantoprazole", "pantoprazole sodium", (("40 mg", "once daily"),)),
        DrugSpec("montelukast", "montelukast sodium", (("10 mg", "once daily in the evening"),)),
        DrugSpec(
            "albuterol",
            "albuterol sulfate",
            (("90 mcg/actuation, 2 puffs", "every 4 to 6 hours as needed"),),
            "respiratory (inhalation)",
        ),
        DrugSpec(
            "tiotropium",
            "tiotropium bromide",
            (("18 mcg inhaled", "once daily"),),
            "respiratory (inhalation)",
        ),
        DrugSpec("meloxicam", "meloxicam", (("7.5 mg", "once daily"),)),
        DrugSpec("ibuprofen", "ibuprofen", (("400 mg", "as needed"),)),
        DrugSpec("alendronate", "alendronate sodium", (("70 mg", "once weekly"),)),
        DrugSpec(
            "sumatriptan", "sumatriptan succinate", (("50 mg", "as needed at migraine onset"),)
        ),
    ]
}


@dataclass(frozen=True)
class LabSpec:
    key: str
    unit: str
    normal: tuple[float, float]
    decimals: int = 0
    aliases: tuple[str, ...] = ()


LABS: dict[str, LabSpec] = {
    lab.key: lab
    for lab in [
        LabSpec(
            "hemoglobin a1c",
            "%",
            (4.8, 5.6),
            1,
            ("hba1c", "a1c", "glycated hemoglobin", "glycosylated hemoglobin"),
        ),
        LabSpec(
            "fasting glucose",
            "mg/dL",
            (72, 99),
            0,
            ("fasting blood sugar", "fbs", "fasting blood glucose"),
        ),
        LabSpec(
            "ldl cholesterol", "mg/dL", (70, 129), 0, ("ldl", "ldl-c", "low-density lipoprotein")
        ),
        LabSpec("total cholesterol", "mg/dL", (140, 199), 0, ("cholesterol",)),
        LabSpec(
            "egfr",
            "mL/min/1.73m2",
            (75, 110),
            0,
            ("gfr", "estimated glomerular filtration rate", "glomerular filtration rate"),
        ),
        LabSpec("creatinine", "mg/dL", (0.6, 1.2), 2, ("serum creatinine", "cr")),
        LabSpec("potassium", "mmol/L", (3.6, 5.0), 1, ("k", "serum potassium")),
        LabSpec("sodium", "mmol/L", (136, 145), 0, ("na", "serum sodium")),
        LabSpec("tsh", "mIU/L", (0.5, 4.0), 2, ("thyroid stimulating hormone", "thyrotropin")),
        LabSpec("hemoglobin", "g/dL", (12.0, 16.0), 1, ("hgb", "hb")),
        LabSpec("inr", "", (0.9, 1.1), 1, ("international normalized ratio", "pt/inr")),
        LabSpec("systolic blood pressure", "mmHg", (108, 128), 0, ("systolic bp", "sbp")),
        LabSpec("diastolic blood pressure", "mmHg", (66, 82), 0, ("diastolic bp", "dbp")),
        LabSpec("bmi", "kg/m2", (20.5, 26.0), 1, ("body mass index",)),
        LabSpec(
            "bnp", "pg/mL", (10, 95), 0, ("b-type natriuretic peptide", "brain natriuretic peptide")
        ),
        LabSpec(
            "urine albumin-creatinine ratio",
            "mg/g",
            (3, 25),
            0,
            ("uacr", "microalbumin", "albumin creatinine ratio"),
        ),
    ]
}

BASE_LABS = (
    "systolic blood pressure",
    "diastolic blood pressure",
    "bmi",
    "creatinine",
    "egfr",
    "potassium",
    "sodium",
)


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    icd10: str
    weight: float
    group: str  # at most one condition per group per patient
    drugs: tuple[str, ...]
    labs: tuple[str, ...] = ()
    # lab -> (low, high) overriding the normal range for patients with this condition
    lab_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    min_age: int = 25


CONDITIONS: dict[str, ConditionSpec] = {
    c.key: c
    for c in [
        ConditionSpec(
            "type 2 diabetes",
            "E11.9",
            3.0,
            "diabetes",
            ("metformin", "empagliflozin", "glipizide", "sitagliptin"),
            ("hemoglobin a1c", "fasting glucose", "urine albumin-creatinine ratio"),
            {
                "hemoglobin a1c": (6.6, 9.8),
                "fasting glucose": (118, 210),
                "urine albumin-creatinine ratio": (8, 60),
            },
            (
                "type 2 diabetes mellitus",
                "t2dm",
                "dm2",
                "diabetes",
                "non-insulin dependent diabetes",
            ),
        ),
        ConditionSpec(
            "type 2 diabetes with hyperglycemia",
            "E11.65",
            1.0,
            "diabetes",
            ("metformin", "insulin glargine", "empagliflozin"),
            ("hemoglobin a1c", "fasting glucose"),
            {"hemoglobin a1c": (8.8, 11.6), "fasting glucose": (180, 290)},
            ("uncontrolled diabetes", "poorly controlled diabetes"),
        ),
        ConditionSpec(
            "type 2 diabetes with diabetic chronic kidney disease",
            "E11.22",
            0.8,
            "diabetes",
            ("metformin", "empagliflozin", "insulin glargine"),
            ("hemoglobin a1c", "urine albumin-creatinine ratio"),
            {"hemoglobin a1c": (7.0, 9.2), "urine albumin-creatinine ratio": (45, 280)},
            ("diabetic kidney disease", "diabetic nephropathy"),
            40,
        ),
        ConditionSpec(
            "hypertension",
            "I10",
            3.2,
            "htn",
            ("lisinopril", "losartan", "amlodipine", "hydrochlorothiazide"),
            (),
            {"systolic blood pressure": (132, 158), "diastolic blood pressure": (82, 96)},
            ("essential hypertension", "high blood pressure", "htn"),
        ),
        ConditionSpec(
            "hyperlipidemia",
            "E78.5",
            2.6,
            "lipids",
            ("atorvastatin", "rosuvastatin", "simvastatin"),
            ("ldl cholesterol", "total cholesterol"),
            {"ldl cholesterol": (118, 172), "total cholesterol": (195, 262)},
            ("high cholesterol", "dyslipidemia", "hld"),
        ),
        ConditionSpec(
            "chronic kidney disease stage 3",
            "N18.30",
            1.0,
            "ckd",
            ("losartan",),
            ("urine albumin-creatinine ratio",),
            {"egfr": (32, 57), "creatinine": (1.35, 2.1)},
            ("ckd stage 3", "ckd 3", "ckd3", "chronic kidney disease"),
            45,
        ),
        ConditionSpec(
            "chronic kidney disease stage 4",
            "N18.4",
            0.4,
            "ckd",
            ("furosemide",),
            ("urine albumin-creatinine ratio",),
            {"egfr": (17, 28), "creatinine": (2.3, 3.6), "potassium": (4.6, 5.4)},
            ("ckd stage 4", "ckd 4", "ckd4"),
            55,
        ),
        ConditionSpec(
            "asthma",
            "J45.909",
            1.2,
            "airway",
            ("albuterol", "montelukast"),
            (),
            {},
            ("bronchial asthma",),
            18,
        ),
        ConditionSpec(
            "copd",
            "J44.9",
            0.8,
            "airway",
            ("tiotropium", "albuterol"),
            (),
            {},
            ("chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"),
            50,
        ),
        ConditionSpec(
            "atrial fibrillation",
            "I48.91",
            1.0,
            "af",
            ("apixaban", "warfarin", "metoprolol"),
            ("inr",),
            {},
            ("afib", "a-fib", "af"),
            55,
        ),
        ConditionSpec(
            "heart failure",
            "I50.9",
            0.7,
            "hf",
            ("furosemide", "carvedilol", "spironolactone"),
            ("bnp",),
            {"bnp": (180, 820)},
            ("chf", "congestive heart failure", "hf"),
            55,
        ),
        ConditionSpec(
            "coronary artery disease",
            "I25.10",
            1.0,
            "cad",
            ("aspirin", "atorvastatin", "metoprolol", "clopidogrel"),
            ("ldl cholesterol",),
            {"ldl cholesterol": (62, 98)},
            ("cad", "ischemic heart disease", "atherosclerotic heart disease"),
            45,
        ),
        ConditionSpec(
            "old myocardial infarction",
            "I25.2",
            0.5,
            "mi",
            ("aspirin", "clopidogrel"),
            (),
            {},
            ("prior mi", "history of myocardial infarction", "previous heart attack", "old mi"),
            45,
        ),
        ConditionSpec(
            "hypothyroidism",
            "E03.9",
            1.3,
            "thyroid",
            ("levothyroxine",),
            ("tsh",),
            {"tsh": (1.2, 7.8)},
            ("underactive thyroid", "low thyroid"),
        ),
        ConditionSpec(
            "major depressive disorder",
            "F32.9",
            1.1,
            "mood",
            ("sertraline", "escitalopram"),
            (),
            {},
            ("depression", "mdd"),
        ),
        ConditionSpec(
            "generalized anxiety disorder",
            "F41.1",
            0.9,
            "anxiety",
            ("buspirone", "sertraline"),
            (),
            {},
            ("gad", "anxiety"),
        ),
        ConditionSpec(
            "gerd",
            "K21.9",
            1.3,
            "gi",
            ("omeprazole", "pantoprazole"),
            (),
            {},
            ("gastroesophageal reflux disease", "acid reflux", "reflux", "heartburn"),
        ),
        ConditionSpec(
            "knee osteoarthritis",
            "M17.9",
            1.0,
            "oa",
            ("meloxicam", "ibuprofen"),
            (),
            {},
            ("osteoarthritis", "oa", "degenerative joint disease"),
            45,
        ),
        ConditionSpec("obesity", "E66.9", 1.4, "weight", (), (), {"bmi": (31.0, 39.5)}, ("obese",)),
        ConditionSpec(
            "obstructive sleep apnea", "G47.33", 0.7, "osa", (), (), {}, ("osa", "sleep apnea")
        ),
        ConditionSpec(
            "history of stroke",
            "Z86.73",
            0.4,
            "stroke",
            ("clopidogrel", "atorvastatin"),
            (),
            {},
            ("prior stroke", "history of cva", "old stroke"),
            50,
        ),
        ConditionSpec(
            "osteoporosis",
            "M81.0",
            0.6,
            "bone",
            ("alendronate",),
            (),
            {},
            ("age-related osteoporosis", "low bone density"),
            55,
        ),
        ConditionSpec(
            "anemia",
            "D64.9",
            0.6,
            "heme",
            (),
            ("hemoglobin",),
            {"hemoglobin": (8.9, 11.4)},
            ("low hemoglobin",),
        ),
        ConditionSpec(
            "migraine",
            "G43.909",
            0.8,
            "headache",
            ("sumatriptan",),
            (),
            {},
            ("migraine headache", "migraines"),
            18,
        ),
    ]
}

ALLERGIES: tuple[tuple[str, str, str], ...] = (
    ("penicillin", "rash", "moderate"),
    ("sulfonamide antibiotics", "hives", "moderate"),
    ("codeine", "nausea and vomiting", "mild"),
    ("latex", "contact dermatitis", "mild"),
    ("shellfish", "lip swelling", "severe"),
    ("lisinopril", "angioedema", "severe"),
    ("aspirin", "wheezing", "moderate"),
)

FIRST_NAMES = {
    "female": [
        "Asha",
        "Maria",
        "Linda",
        "Priya",
        "Fatima",
        "Grace",
        "Mei",
        "Olivia",
        "Ananya",
        "Sofia",
        "Hannah",
        "Aisha",
        "Rosa",
        "Nisha",
        "Elena",
        "Joy",
        "Kavya",
        "Yuki",
        "Amara",
        "Leah",
        "Meera",
        "Clara",
        "Divya",
        "Irene",
        "Zara",
    ],
    "male": [
        "Rahul",
        "James",
        "Arjun",
        "Carlos",
        "Wei",
        "David",
        "Omar",
        "Rohan",
        "Samuel",
        "Vikram",
        "Daniel",
        "Kenji",
        "Tariq",
        "Michael",
        "Aditya",
        "Luis",
        "Peter",
        "Imran",
        "Thomas",
        "Karan",
        "Noah",
        "Hiro",
        "Sanjay",
        "Ethan",
        "Kwame",
    ],
}
LAST_NAMES = [
    "Verma",
    "Garcia",
    "Okafor",
    "Chen",
    "Sharma",
    "Patel",
    "Nguyen",
    "Johnson",
    "Khan",
    "Rodriguez",
    "Iyer",
    "Kim",
    "Mensah",
    "Novak",
    "Reddy",
    "Silva",
    "Haddad",
    "Tanaka",
    "Brown",
    "Menon",
    "Ali",
    "Kowalski",
    "Das",
    "Fischer",
    "Banerjee",
    "Lopez",
    "Nair",
    "Osei",
    "Singh",
    "Martin",
]
