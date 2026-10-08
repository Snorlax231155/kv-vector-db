"""Validate the frozen MedMemory gold set (eval/gold/queries.jsonl) against the seed data.

Run from the repo root:

    python eval/gold/validate.py            # validates queries.jsonl next to this file
    python eval/gold/validate.py path.jsonl # validates another version file

What it checks
  * schema: ids q001.. in order, required fields, allowed enums, entity vocabulary
    (drugs / labs / conditions must be canonical keys from src/medmemory/ingest/catalog.py);
  * every `kv:` source is a prefix of at least one KV key rebuilt from the seed files using
    the key schema documented in src/medmemory/kv/records.py;
  * every `doc:` source is an existing note / label / topic doc_id;
  * routing/status rules per category, source types per route, patient ownership of sources,
    scope-violation logic, paraphrase links;
  * the required category and route minimums;
  * fact checks: every cited clinical note contains the fact it is cited for (NOTE_PHRASES),
    selected KV facts match the seed (FACT_CHECKS), and every abstention is backed by an
    absence (or a planted dose conflict) in the seed data.

It reads only data/seed/*.jsonl and catalog.py (loaded by file path, so no package import or
third-party dependency is needed). It never imports the router, pipeline or any other system
code: the gold set must stay independent of the system it evaluates.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = ROOT / "data" / "seed"
CATALOG = ROOT / "src" / "medmemory" / "ingest" / "catalog.py"

CATEGORIES = {
    "kv_lookup", "semantic", "hybrid", "red_flag", "red_flag_negative",
    "abstention", "scope_violation", "out_of_scope", "paraphrase",
}
ROUTES = {"KV", "VECTOR", "HYBRID"}
STATUSES = {"answered", "abstained", "red_flag", "out_of_scope", "scope_violation"}
ENTITY_KEYS = ("patient_ids", "icd10_codes", "rxnorm_codes", "drugs", "labs", "conditions")
FIELDS = ("id", "query", "patient_scope", "category", "expected_route", "expected_status",
          "expected_entities", "expected_sources", "paraphrase_of", "notes")

MIN_CATEGORY = {"red_flag": 10, "red_flag_negative": 4, "abstention": 8,
                "scope_violation": 4, "out_of_scope": 3, "paraphrase": 6}
MIN_ROUTE = {"KV": 22, "VECTOR": 20, "HYBRID": 18}
MIN_TOTAL = 84

PID_RE = re.compile(r"^P\d{4,6}$")
ICD_RE = re.compile(r"^[A-Z]\d{2}(\.[0-9A-Z]{1,4})?$")


# --------------------------------------------------------------------------- seed loading


def _jsonl(name: str) -> list[dict]:
    with (SEED / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_catalog():
    sys.dont_write_bytecode = True  # read-only: never leave a __pycache__ under src/
    spec = importlib.util.spec_from_file_location("_gold_catalog", CATALOG)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def slug(text: str) -> str:
    """Same lab slug rule as the KV schema: lowercase, non-alphanumerics -> '_'."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class Seed:
    def __init__(self) -> None:
        self.patients = {p["patient_id"]: p for p in _jsonl("patients.jsonl")}
        self.notes = {n["doc_id"]: n for n in _jsonl("notes.jsonl")}
        self.labels = {lab["drug"]: lab for lab in _jsonl("drug_labels.jsonl")}
        self.topics = {t["topic_id"]: t for t in _jsonl("guidelines.jsonl")}
        self.icd = {c["code"]: c["name"] for c in _jsonl("icd10cm.jsonl")}
        self.scenarios = _jsonl("scenarios.jsonl")
        self.kv_keys = self._build_kv_keys()
        self.doc_ids = (
            set(self.notes)
            | {f"label:{d.replace(' ', '-')}" for d in self.labels}
            | {f"topic:{t}" for t in self.topics}
        )

    def _build_kv_keys(self) -> list[str]:
        """Rebuild KV keys following the schema in src/medmemory/kv/records.py."""
        keys: list[str] = []
        for pid, p in self.patients.items():
            base = f"patient:{pid}"
            keys += [base, f"idx:patient:{pid}"]
            keys += [f"{base}:condition:{c['icd10']}" for c in p["conditions"]]
            keys += [f"{base}:med:{m['drug']}" for m in p["medications"]]
            keys += [f"{base}:allergy:{a['substance']}" for a in p["allergies"]]
            keys += [f"{base}:lab:{slug(lab['lab'])}:{lab['date']}" for lab in p["labs"]]
            keys += [f"{base}:encounter:{e['date']}:{e['encounter_id']}" for e in p["encounters"]]
        keys += [f"icd10:{code}" for code in self.icd]
        for drug, lab in self.labels.items():
            keys += [f"drug:rx:{lab['ingredient_rxcui']}", f"drug:name:{drug}"]
        keys += [f"doc:{d}" for d in self.notes]
        return sorted(keys)

    def kv_prefix_exists(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self.kv_keys)

    # helpers for fact checks
    def labs_of(self, pid: str, name: str) -> list[dict]:
        return sorted((x for x in self.patients[pid]["labs"] if x["lab"] == name), key=lambda x: x["date"])

    def med(self, pid: str, drug: str) -> dict | None:
        return next((m for m in self.patients[pid]["medications"] if m["drug"] == drug), None)

    def note_has(self, doc_id: str, phrase: str) -> bool:
        return doc_id in self.notes and phrase.lower() in self.notes[doc_id]["text"].lower()

    def has_label_for(self, name: str) -> bool:
        name = name.lower()
        return any(
            name == d or name in (lab.get("generic_name") or "").lower()
            or any(name in b.lower() for b in lab.get("brand_names") or [])
            for d, lab in self.labels.items()
        )

    def conflict(self, pid: str, drug: str) -> dict | None:
        return next((s for s in self.scenarios if s.get("type") == "dose_conflict"
                     and s["patient_id"] == pid and s["drug"] == drug), None)


# --------------------------------------------------------------------------- fact checks
# Each entry: item id -> callable(seed) -> bool. They pin the claim in the item's `notes`
# to the seed data, so a regenerated seed that silently changes a fact fails loudly.


def _latest(seed: Seed, pid: str, lab: str):
    rows = seed.labs_of(pid, lab)
    return rows[-1]["value"] if rows else None


FACT_CHECKS = {
    # KV
    "q001": lambda s: _latest(s, "P0001", "hemoglobin a1c") == 7.7,
    "q003": lambda s: [a["substance"] for a in s.patients["P0002"]["allergies"]] == ["sulfonamide antibiotics"],
    "q005": lambda s: s.labels["empagliflozin"]["ingredient_rxcui"] == "1545653"
    and "Jardiance" in s.labels["empagliflozin"]["brand_names"],
    "q006": lambda s: "Eliquis" in s.labels["apixaban"]["brand_names"],
    "q007": lambda s: s.labels["metformin"]["ingredient_rxcui"] == "6809",
    "q010": lambda s: (s.med("P0027", "warfarin") or {}).get("status") == "active",
    "q011": lambda s: (s.med("P0034", "metformin") or {}).get("dose") == "850 mg" and s.conflict("P0034", "metformin") is None,
    "q015": lambda s: max(x["value"] for x in s.labs_of("P0044", "systolic blood pressure")) == 157,
    "q019": lambda s: len(s.labs_of("P0039", "ldl cholesterol")) == 4 and _latest(s, "P0039", "ldl cholesterol") == 156,
    "q021": lambda s: (s.med("P0021", "carvedilol") or {}).get("dose") == "12.5 mg"
    and "Coreg" in s.labels["carvedilol"]["brand_names"],
    "q022": lambda s: (s.med("P0001", "lisinopril") or {}).get("stop_reason") == "persistent dry cough"
    and not any("lisinopril" in n["text"] for n in s.notes.values() if n["patient_id"] == "P0001"),
    "q023": lambda s: _latest(s, "P0011", "potassium") == 5.2,
    "q025": lambda s: {m["drug"] for m in s.patients["P0012"]["medications"]}
    .isdisjoint({"warfarin", "apixaban", "clopidogrel", "aspirin"}),
    "q087": lambda s: any(a["substance"] == "lisinopril" and a["reaction"] == "angioedema"
                          for a in s.patients["P0034"]["allergies"]),
    # hybrid / note items whose label depends on more than "the cited note says X"
    "q044": lambda s: max((n["date"], d) for d, n in s.notes.items()
                          if n["patient_id"] == "P0018" and "DEXA" in n["text"])[1] == "note:P0018:E001804",
    "q058": lambda s: s.med("P0050", "omeprazole") is not None and "omeprazole" in json.dumps(s.labels["clopidogrel"]),
    "q059": lambda s: any(a["substance"] == "lisinopril" and a["severity"] == "severe"
                          for a in s.patients["P0042"]["allergies"]),
    "q065": lambda s: (s.med("P0015", "atorvastatin") or {}).get("status") == "active",
    "q071": lambda s: "warfarin" in json.dumps(s.labels["sertraline"]["sections"]),
    "q089": lambda s: s.med("P0007", "warfarin")["start"] < s.notes["note:P0007:E000704"]["date"],
    # abstentions: the knowledge base must NOT contain the answer (or must hold a conflict)
    "q090": lambda s: not s.has_label_for("semaglutide"),
    "q091": lambda s: not s.has_label_for("ozempic") and not s.has_label_for("semaglutide"),
    "q092": lambda s: not s.has_label_for("lisdexamfetamine"),
    "q093": lambda s: (c := s.conflict("P0004", "levothyroxine")) is not None
    and s.med("P0004", "levothyroxine")["dose"] == c["kv_dose"] and s.note_has(c["doc_id"], c["note_dose"]),
    "q094": lambda s: (c := s.conflict("P0040", "metformin")) is not None
    and s.med("P0040", "metformin")["dose"] == c["kv_dose"] and s.note_has(c["doc_id"], c["note_dose"]),
    "q095": lambda s: not s.labs_of("P0003", "hemoglobin a1c"),
    "q096": lambda s: not s.labs_of("P0013", "inr"),
    "q097": lambda s: not s.has_label_for("dapagliflozin") and s.med("P0024", "dapagliflozin") is None,
    "q098": lambda s: "S72.001A" not in s.icd,
}

# Every item that cites a clinical note must say which fact the note supports: each cited
# `doc:note:` source must contain at least one of these phrases (case-insensitive).
NOTE_PHRASES = {
    "q031": ["since the amlodipine dose was increased"],
    "q032": ["aching in both thighs and calves"],
    "q033": ["easy bruising"],
    "q038": ["PHQ-9 score decreased from 16 to 8"],
    "q043": ["short of breath after one flight of stairs"],
    "q044": ["T-score of -2.8"],
    "q047": ["tingling in both feet"],
    "q050": ["rescue inhaler three or four times a week"],
    "q051": ["CPAP about 4 hours"],
    "q055": ["aching in both thighs and calves"],
    "q065": ["without residual deficits"],
    "q068": ["improved since levothyroxine was adjusted"],
    "q072": ["rescue inhaler three or four times a week"],
    "q073": ["Denies hypoglycemic episodes"],
    "q085": ["chest tightness", "chest pain"],
    "q086": ["Denies thoughts of self-harm"],
    "q088": ["without residual deficits"],
    "q089": ["easy bruising"],
    "q111": ["since the amlodipine dose was increased"],
}


# --------------------------------------------------------------------------- validation


def validate(path: Path) -> tuple[list[str], list[dict]]:
    seed = Seed()
    cat = _load_catalog()
    vocab = {"drugs": set(cat.DRUGS), "labs": set(cat.LABS), "conditions": set(cat.CONDITIONS)}

    errors: list[str] = []
    items: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"line {n}: invalid JSON ({exc})")
    by_id = {it.get("id"): it for it in items}

    for idx, it in enumerate(items, 1):
        iid = it.get("id", f"<line {idx}>")

        def err(msg: str, iid=iid) -> None:
            errors.append(f"{iid}: {msg}")

        # ---- schema
        if list(it.keys()) != list(FIELDS):
            err(f"fields must be exactly {FIELDS} in order, got {tuple(it.keys())}")
        if iid != f"q{idx:03d}":
            err(f"expected id q{idx:03d}")
        if not isinstance(it.get("query"), str) or not it["query"].strip():
            err("empty query")
        if not isinstance(it.get("notes"), str) or not it["notes"].strip():
            err("missing notes rationale")
        category, route, status = it.get("category"), it.get("expected_route"), it.get("expected_status")
        scope = it.get("patient_scope")
        if category not in CATEGORIES:
            err(f"bad category {category!r}")
        if route is not None and route not in ROUTES:
            err(f"bad route {route!r}")
        if status not in STATUSES:
            err(f"bad status {status!r}")
        if scope is not None and (not PID_RE.match(scope) or scope not in seed.patients):
            err(f"patient_scope {scope!r} is not a seed patient")

        ents = it.get("expected_entities") or {}
        if list(ents.keys()) != list(ENTITY_KEYS):
            err(f"expected_entities keys must be {ENTITY_KEYS}")
        for key in ENTITY_KEYS:
            vals = ents.get(key, [])
            if not isinstance(vals, list) or len(vals) != len(set(vals)):
                err(f"entities.{key} must be a list without duplicates")
                continue
            for v in vals:
                if key in vocab and v not in vocab[key]:
                    err(f"entities.{key}: {v!r} is not a canonical catalog key")
                if key == "patient_ids" and (not PID_RE.match(v) or v not in seed.patients):
                    err(f"entities.patient_ids: {v!r} is not a seed patient")
                if key == "patient_ids" and v not in it["query"] and v.lower() not in it["query"].lower():
                    err(f"entities.patient_ids: {v!r} is not written in the query text")
                if key == "icd10_codes" and not ICD_RE.match(v):
                    err(f"entities.icd10_codes: {v!r} is not dotted ICD-10-CM form")
                if key == "rxnorm_codes" and not v.isdigit():
                    err(f"entities.rxnorm_codes: {v!r} is not numeric")
        pids = ents.get("patient_ids", [])

        # ---- category -> route/status rules
        sources = it.get("expected_sources", [])
        if not isinstance(sources, list):
            err("expected_sources must be a list")
            sources = []
        null_route_cats = {"red_flag": "red_flag", "scope_violation": "scope_violation", "out_of_scope": "out_of_scope"}
        if category in null_route_cats:
            if route is not None:
                err(f"{category} items never reach routing; expected_route must be null")
            if status != null_route_cats[category]:
                err(f"{category} items must have status {null_route_cats[category]!r}")
        else:
            if route is None:
                err(f"{category} items must have a route")
        fixed = {"kv_lookup": "KV", "semantic": "VECTOR", "hybrid": "HYBRID"}
        if category in fixed:
            if route != fixed[category]:
                err(f"{category} must route {fixed[category]}")
            if status != "answered":
                err(f"{category} items must be answered (use category 'abstention' otherwise)")
        if category == "abstention" and status != "abstained":
            err("abstention items must have status 'abstained'")
        if category == "red_flag_negative" and status not in {"answered", "abstained"}:
            err("red_flag_negative items must be answered or abstained")
        if category in {"red_flag", "scope_violation", "out_of_scope", "abstention"} and sources:
            err(f"{category} items must have expected_sources []")
        if status == "answered" and not sources:
            err("answered items need at least one expected source")

        # ---- scope logic
        if category == "scope_violation":
            if not (len(pids) >= 2 or any(p != scope for p in pids)):
                err("scope_violation needs two named patients or a named patient different from scope")
        else:
            if len(pids) > 1:
                err("only scope_violation items may name more than one patient")
            if scope is not None and any(p != scope for p in pids):
                err("names a patient other than patient_scope but is not a scope_violation")
        owner = scope or (pids[0] if len(pids) == 1 else None)

        # ---- sources
        kinds = set()
        for src in sources:
            if src.startswith("kv:"):
                kinds.add("kv")
                key = src[3:]
                if not seed.kv_prefix_exists(key):
                    err(f"kv source {src!r} matches no KV key rebuilt from seed")
                m = re.match(r"patient:(P\d+)", key)
                if m and m.group(1) != owner:
                    err(f"kv source {src!r} belongs to {m.group(1)}, item patient is {owner}")
                if re.match(r"patient:P\d+:lab:[a-z0-9_]+$", key):
                    err(f"lab prefix {src!r} must end with ':' so 'hemoglobin' cannot match 'hemoglobin_a1c'")
            elif src.startswith("doc:"):
                kinds.add("doc")
                doc_id = src[4:]
                if doc_id not in seed.doc_ids:
                    err(f"doc source {src!r} is not a seed doc_id")
                if doc_id.startswith("note:"):
                    note_pid = doc_id.split(":")[1]
                    if note_pid != owner:
                        err(f"note {doc_id} belongs to {note_pid}, item patient is {owner}")
                    if scope is None:
                        err(f"note source {doc_id} requires patient_scope (notes are never searched unscoped)")
            else:
                err(f"source {src!r} must start with 'kv:' or 'doc:'")
        if len(sources) != len(set(sources)):
            err("duplicate expected_sources")
        if sources and route in ROUTES:
            want = {"KV": {"kv"}, "VECTOR": {"doc"}, "HYBRID": {"kv", "doc"}}[route]
            if kinds != want:
                err(f"route {route} expects source kinds {sorted(want)}, got {sorted(kinds)}")

        # ---- paraphrases
        para = it.get("paraphrase_of")
        if category == "paraphrase":
            target = by_id.get(para)
            if target is None or int(para[1:]) >= idx:
                err(f"paraphrase_of {para!r} must reference an earlier item")
            else:
                if target["category"] in {"paraphrase", "red_flag", "scope_violation", "out_of_scope", "abstention"} \
                        or target["expected_status"] != "answered":
                    err(f"paraphrase target {para} must be an answerable item")
                for fld in ("patient_scope", "expected_route", "expected_status", "expected_sources"):
                    if target[fld] != it[fld]:
                        err(f"paraphrase must keep {fld} of {para}")
            if not re.match(r"^(close|genuine) rewording of q\d{3}", it.get("notes", "")):
                err("paraphrase notes must start with 'close rewording of qNNN' or 'genuine rewording of qNNN'")
        elif para is not None:
            err("paraphrase_of must be null unless category is 'paraphrase'")

        # ---- cited notes must contain the fact
        note_sources = [s[4:] for s in sources if s.startswith("doc:note:")]
        if note_sources and iid not in NOTE_PHRASES:
            err("cites a clinical note but has no NOTE_PHRASES entry naming the supported fact")
        for doc_id in note_sources:
            if doc_id in seed.notes and not any(seed.note_has(doc_id, ph) for ph in NOTE_PHRASES.get(iid, [])):
                err(f"cited note {doc_id} does not contain the expected fact {NOTE_PHRASES.get(iid)}")

        # ---- fact checks
        check = FACT_CHECKS.get(iid)
        if check is not None:
            try:
                ok = bool(check(seed))
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                ok = False
                err(f"fact check raised {type(exc).__name__}: {exc}")
            if not ok:
                err("fact check failed: the seed data no longer supports this item's label")

    for iid in (*FACT_CHECKS, *NOTE_PHRASES):
        if iid not in by_id:
            errors.append(f"fact check defined for missing item {iid}")
    for it in items:
        if it.get("category") == "abstention" and it.get("id") not in FACT_CHECKS:
            errors.append(f"{it.get('id')}: abstention has no FACT_CHECKS entry proving the answer is absent")

    # ---- mix minimums
    cats = Counter(it.get("category") for it in items)
    routes = Counter(it.get("expected_route") for it in items)
    if len(items) < MIN_TOTAL:
        errors.append(f"need at least {MIN_TOTAL} items, have {len(items)}")
    for c, n in MIN_CATEGORY.items():
        if cats[c] < n:
            errors.append(f"category {c}: need >= {n}, have {cats[c]}")
    for r, n in MIN_ROUTE.items():
        if routes[r] < n:
            errors.append(f"route {r}: need >= {n}, have {routes[r]}")
    kinds = Counter(it["notes"].split(" ", 1)[0] for it in items if it.get("category") == "paraphrase")
    if kinds["close"] < 3 or kinds["genuine"] < 3:
        errors.append(f"paraphrases need >= 3 close and >= 3 genuine, have {dict(kinds)}")
    if len({it["query"].strip().lower() for it in items}) != len(items):
        errors.append("duplicate query text")

    return errors, items


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else HERE / "queries.jsonl"
    errors, items = validate(path)
    cats = Counter(it.get("category") for it in items)
    routes = Counter(str(it.get("expected_route")) for it in items)
    statuses = Counter(it.get("expected_status") for it in items)
    print(f"{path.name}: {len(items)} items")
    print("  categories: " + ", ".join(f"{k}={cats[k]}" for k in sorted(cats)))
    print("  routes:     " + ", ".join(f"{k}={routes[k]}" for k in sorted(routes)))
    print("  statuses:   " + ", ".join(f"{k}={statuses[k]}" for k in sorted(statuses)))
    print(f"  patients referenced: {len({it['patient_scope'] for it in items if it['patient_scope']} | {p for it in items for p in it['expected_entities']['patient_ids']})}")
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors:
            print("  - " + e)
        return 1
    print("OK: all sources resolve and all checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
