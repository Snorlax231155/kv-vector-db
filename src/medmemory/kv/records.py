"""Domain layer over a `KVStore`: key schema, JSON encoding, and entity-driven lookups.

Key schema (all values are UTF-8 JSON):

    patient:{pid}                               demographics
    patient:{pid}:condition:{icd10}             one active/resolved condition
    patient:{pid}:med:{drug}                    one medication (active or stopped)
    patient:{pid}:allergy:{substance}
    patient:{pid}:lab:{lab_slug}:{yyyy-mm-dd}   one observation; prefix scan = time series
    patient:{pid}:encounter:{yyyy-mm-dd}:{eid}
    idx:patient:{pid}                           list-view summary
    icd10:{code}                                ICD-10-CM description
    drug:rx:{rxcui}                             drug identifiers + label pointer (no doses)
    drug:name:{generic}                         -> {"rxcui": ...}
    doc:{doc_id}                                full source text for the citation viewer
    meta:version:{pid} | meta:version:global    data versions for cache invalidation

Scoping rule: every patient read goes through `_patient_prefix(pid)`, and lookups only
accept the request's resolved scope, never an ID parsed from free text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from medmemory.contracts.protocols import KVOp, KVStore
from medmemory.contracts.schemas import Entities, KVRecord, PatientSummary
from medmemory.errors import ScopeViolationError
from medmemory.ingest.catalog import LABS

RECORD_KINDS = ("demographics", "conditions", "medications", "allergies", "labs", "encounters")


def _enc(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _dec(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _patient_prefix(pid: str) -> str:
    if not re.fullmatch(r"P\d{4,6}", pid):
        raise ScopeViolationError(f"invalid patient id {pid!r}")
    return f"patient:{pid}"


# --------------------------------------------------------------------------- rendering


def render(kind: str, data: dict[str, Any], pid: str | None = None) -> tuple[str, str]:
    """(title, sentence) for a record. The sentence is what the generator sees and quotes,
    so it must state the fact plainly with its date and identifiers."""
    who = f"Patient {pid}" if pid else ""
    if kind == "demographics":
        return (
            f"{pid} · demographics",
            f"{who} is {data['name']}; sex: {data['sex']}; date of birth: {data['birth_date']}.",
        )
    if kind == "condition":
        return (
            f"{pid} · condition · {data['icd10']}",
            f"{who} has {data['name']} (ICD-10-CM {data['icd10']}), status {data['status']}, onset {data['onset']}.",
        )
    if kind == "medication":
        if data["status"] == "active":
            s = f"{who} is taking {data['drug']} {data['dose']} {data['frequency']} for {data['indication']}, started {data['start']}."
        else:
            s = (
                f"{who} previously took {data['drug']} {data['dose']} {data['frequency']}; it was stopped on "
                f"{data['stop']} because of {data['stop_reason']}."
            )
        return (f"{pid} · medication · {data['drug']} ({data['status']})", s)
    if kind == "allergy":
        return (
            f"{pid} · allergy · {data['substance']}",
            f"{who} has a documented allergy to {data['substance']} (reaction: {data['reaction']}, severity: {data['severity']}).",
        )
    if kind == "lab":
        unit = data["unit"]
        value = f"{data['value']}{unit}" if unit == "%" else f"{data['value']} {unit}".strip()
        return (
            f"{pid} · lab · {data['lab']} · {data['date']}",
            f"{who} {data['lab']} on {data['date']}: {value} (flag: {data['flag']}).",
        )
    if kind == "encounter":
        return (
            f"{pid} · encounter · {data['date']}",
            f"{who} had a {data['type']} on {data['date']} for {data['reason']}.",
        )
    if kind == "icd10":
        return (
            f"ICD-10-CM {data['code']}",
            f"ICD-10-CM code {data['code']} means: {data['name']}.",
        )
    if kind == "drug":
        brands = ", ".join(data.get("brand_names", [])[:3]) or "none listed"
        return (
            f"RxNorm {data['rxcui']} · {data['name']}",
            f"{data['name'].capitalize()} has RxNorm ingredient identifier (RxCUI) {data['rxcui']}; "
            f"label generic name: {data['generic_name']}; brand names: {brands}; FDA label set ID {data['label_set_id']}.",
        )
    return (kind, json.dumps(data, sort_keys=True))


def _kind_from_key(key: str) -> str:
    parts = key.split(":")
    if parts[0] == "patient":
        return "demographics" if len(parts) == 2 else {"med": "medication"}.get(parts[2], parts[2])
    if parts[0] == "drug":
        return "drug"
    return parts[0]


def to_record(key: str, raw: bytes) -> KVRecord:
    data = _dec(raw)
    kind = _kind_from_key(key)
    pid = key.split(":")[1] if key.startswith("patient:") else None
    title, sentence = render(kind, data, pid)
    return KVRecord(
        key=key, kind=kind, patient_id=pid, title=title, data={**data, "text": sentence}
    )


# --------------------------------------------------------------------------- repository


class RecordRepository:
    def __init__(self, kv: KVStore) -> None:
        self.kv = kv
        self._names: list[tuple[re.Pattern[str], str]] | None = None

    def patients_named(self, text: str) -> list[str]:
        """Patient IDs whose full name appears in `text`. Scope checks use this so 'what is
        Fatima Brown on?' can't bypass the active-patient rule by avoiding the ID."""
        if self._names is None:
            self._names = [
                (re.compile(rf"\b{re.escape(s['name'])}\b", re.IGNORECASE), s["patient_id"])
                for s in (_dec(raw) for _, raw in self.kv.scan("idx:patient:"))
            ]
        return sorted({pid for pattern, pid in self._names if pattern.search(text)})

    # ---- versions (exact-cache keys include these; bumping one invalidates by construction)

    def data_version(self, pid: str | None) -> int:
        g = self.kv.get("meta:version:global")
        base = int(g) if g else 0
        if pid is None:
            return base
        p = self.kv.get(f"meta:version:{pid}")
        return base * 1_000_000 + (int(p) if p else 0)

    def bump_version(self, pid: str | None) -> int:
        key = "meta:version:global" if pid is None else f"meta:version:{pid}"
        current = int(self.kv.get(key) or b"0") + 1
        self.kv.put(key, str(current).encode())
        return self.data_version(pid)

    # ---- writes (ingestion + API)

    def load_patient(self, p: dict[str, Any]) -> int:
        pid = p["patient_id"]
        pre = _patient_prefix(pid)
        ops = [
            KVOp("put", pre, _enc({k: p[k] for k in ("patient_id", "name", "sex", "birth_date")}))
        ]
        for c in p["conditions"]:
            ops.append(KVOp("put", f"{pre}:condition:{c['icd10']}", _enc(c)))
        for m in p["medications"]:
            ops.append(KVOp("put", f"{pre}:med:{slug(m['drug'])}", _enc(m)))
        for a in p["allergies"]:
            ops.append(KVOp("put", f"{pre}:allergy:{slug(a['substance'])}", _enc(a)))
        for lab in p["labs"]:
            ops.append(KVOp("put", f"{pre}:lab:{slug(lab['lab'])}:{lab['date']}", _enc(lab)))
        for e in p["encounters"]:
            ops.append(KVOp("put", f"{pre}:encounter:{e['date']}:{e['encounter_id']}", _enc(e)))
        ops.append(KVOp("put", f"idx:patient:{pid}", _enc(self._summary_dict(p))))
        self.kv.batch(ops)
        self._names = None
        return len(ops)

    @staticmethod
    def _summary_dict(p: dict[str, Any]) -> dict[str, Any]:
        enc_dates = [e["date"] for e in p["encounters"]]
        return {
            "patient_id": p["patient_id"],
            "name": p["name"],
            "sex": p["sex"],
            "birth_date": p["birth_date"],
            "conditions": [c["name"] for c in p["conditions"]],
            "medication_count": sum(1 for m in p["medications"] if m["status"] == "active"),
            "lab_count": len(p["labs"]),
            "note_count": p.get("note_count", 0),
            "last_encounter": max(enc_dates) if enc_dates else None,
        }

    def load_icd10(self, rows: Iterable[dict[str, str]]) -> int:
        ops = [KVOp("put", f"icd10:{r['code']}", _enc(r)) for r in rows]
        self.kv.batch(ops)
        return len(ops)

    def load_drug(self, label: dict[str, Any]) -> None:
        rx = label.get("ingredient_rxcui") or (label["rxcuis"][0] if label["rxcuis"] else "")
        drug = {
            "rxcui": rx,
            "name": label["drug"],
            "generic_name": label["generic_name"],
            "brand_names": label["brand_names"],
            "label_set_id": label["set_id"],
            "label_url": label["source_url"],
            "routes": label["route"],
        }
        self.kv.batch(
            [
                KVOp("put", f"drug:rx:{rx}", _enc(drug)),
                KVOp("put", f"drug:name:{slug(label['drug'])}", _enc({"rxcui": rx})),
            ]
        )

    def load_document(self, doc_id: str, payload: dict[str, Any]) -> None:
        self.kv.put(f"doc:{doc_id}", _enc(payload))

    def add_lab(self, pid: str, lab: str, value: float, unit: str, on: date) -> tuple[str, int]:
        pre = _patient_prefix(pid)
        if self.kv.get(pre) is None:
            raise KeyError(pid)
        key = f"{pre}:lab:{slug(lab)}:{on.isoformat()}"
        spec = LABS.get(lab.lower())
        flag = "unknown"
        if spec is not None:
            lo, hi = spec.normal
            flag = "low" if value < lo else "high" if value > hi else "normal"
        self.kv.put(
            key,
            _enc(
                {
                    "lab": lab.lower(),
                    "value": value,
                    "unit": unit,
                    "date": on.isoformat(),
                    "flag": flag,
                }
            ),
        )
        summary = self.kv.get(f"idx:patient:{pid}")
        if summary:
            s = _dec(summary)
            s["lab_count"] = self.kv.count(f"{pre}:lab:")
            self.kv.put(f"idx:patient:{pid}", _enc(s))
        return key, self.bump_version(pid)

    # ---- reads

    def get(self, key: str) -> KVRecord | None:
        raw = self.kv.get(key)
        return to_record(key, raw) if raw is not None else None

    def document(self, doc_id: str) -> dict[str, Any] | None:
        raw = self.kv.get(f"doc:{doc_id}")
        return _dec(raw) if raw else None

    def patient_exists(self, pid: str) -> bool:
        return self.kv.get(_patient_prefix(pid)) is not None

    def list_patients(self, query: str = "", limit: int = 100) -> list[PatientSummary]:
        out = []
        q = query.lower().strip()
        for _, raw in self.kv.scan("idx:patient:"):
            s = _dec(raw)
            hay = f"{s['patient_id']} {s['name']} {' '.join(s['conditions'])}".lower()
            if q and q not in hay:
                continue
            born = date.fromisoformat(s["birth_date"])
            ref = date.fromisoformat(s["last_encounter"]) if s["last_encounter"] else date.today()
            age = ref.year - born.year - ((ref.month, ref.day) < (born.month, born.day))
            out.append(PatientSummary(**{**s, "age": age}))
            if len(out) >= limit:
                break
        return out

    def patient_records(self, pid: str, kinds: Iterable[str] | None = None) -> list[KVRecord]:
        pre = _patient_prefix(pid)
        wanted = set(kinds) if kinds else None
        out = []
        for key, raw in self.kv.scan(pre):
            if key != pre and not key.startswith(pre + ":"):
                continue  # P0001 must not match P00010
            rec = to_record(key, raw)
            if wanted is None or rec.kind in wanted:
                out.append(rec)
        return out

    def lab_series(self, pid: str, lab: str) -> list[KVRecord]:
        pre = _patient_prefix(pid)
        return [to_record(k, v) for k, v in self.kv.scan(f"{pre}:lab:{slug(lab)}:")]

    def icd10(self, code: str) -> KVRecord | None:
        return self.get(f"icd10:{code.upper()}")

    def drug_by_name(self, name: str) -> KVRecord | None:
        raw = self.kv.get(f"drug:name:{slug(name)}")
        if raw is None:
            return None
        return self.get(f"drug:rx:{_dec(raw)['rxcui']}")

    def drug_by_rxcui(self, rxcui: str) -> KVRecord | None:
        return self.get(f"drug:rx:{rxcui}")

    # ---- the KV side of retrieval

    def lookup(
        self, entities: Entities, scope: str | None, max_records: int = 24
    ) -> list[KVRecord]:
        """Translate router entities into exact KV reads, constrained to `scope`.

        Without a scope only global tables (ICD-10, drugs) are read. With a scope, entity
        mentions select the matching patient records; record-type words ('medications',
        'allergies') select whole sections; a bare patient question returns a compact
        overview (demographics, active conditions, active meds, allergies).
        """
        out: list[KVRecord] = []
        seen: set[str] = set()

        def add(rec: KVRecord | None) -> None:
            if rec is not None and rec.key not in seen:
                seen.add(rec.key)
                out.append(rec)

        for code in entities.icd10_codes:
            add(self.icd10(code))
        for rx in entities.rxnorm_codes:
            add(self.drug_by_rxcui(rx))

        if scope is None:
            for drug in entities.drugs:
                add(self.drug_by_name(drug))
            return out[:max_records]

        pre = _patient_prefix(scope)
        tr = entities.time_range
        for lab in entities.labs:
            series = self.lab_series(scope, lab)
            if tr and (tr.start or tr.end):
                series = [
                    r
                    for r in series
                    if (tr.start is None or r.data["date"] >= tr.start.isoformat())
                    and (tr.end is None or r.data["date"] <= tr.end.isoformat())
                ]
            if tr and tr.latest_only:
                series = series[-1:]
            elif not tr:
                series = series[-3:]  # recent trend by default
            for r in series:
                add(r)
        for drug in entities.drugs:
            add(self.get(f"{pre}:med:{slug(drug)}"))
            add(self.drug_by_name(drug))
        if entities.conditions:
            conds = self.patient_records(scope, ["condition"])
            for c in conds:
                hay = f"{c.data.get('key', '')} {c.data.get('name', '')}".lower()
                if any(term in hay for term in entities.conditions):
                    add(c)

        kinds = {
            "medications": "medication",
            "allergies": "allergy",
            "conditions": "condition",
            "encounters": "encounter",
            "demographics": "demographics",
            "labs": "lab",
        }
        for rt in entities.record_types:
            kind = kinds.get(rt)
            if kind is None:
                continue
            recs = self.patient_records(scope, [kind])
            if kind == "lab":
                latest: dict[str, KVRecord] = {}
                for r in recs:
                    latest[r.data["lab"]] = r  # keys sort by date, so last wins
                recs = list(latest.values())
            if kind == "encounter":
                recs = recs[-5:]
            for r in recs:
                add(r)

        nothing_specific = not (
            entities.labs or entities.drugs or entities.conditions or entities.record_types
        )
        if nothing_specific:
            add(self.get(pre))
            for r in self.patient_records(scope, ["condition", "medication", "allergy"]):
                if r.kind != "medication" or r.data.get("status") == "active":
                    add(r)
        return out[:max_records]
