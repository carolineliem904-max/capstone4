from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import pandas as pd

@dataclass
class Box:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

def _center_in_person(item: Box, person: Box) -> bool:
    cx = (item.x1 + item.x2) / 2
    cy = (item.y1 + item.y2) / 2
    return (person.x1 <= cx <= person.x2) and (person.y1 <= cy <= person.y2)

def _parse_ultralytics_result(res) -> Tuple[List[Box], Dict[int, str]]:
    """
    res: ultralytics Results (single image): results[0]
    returns: list of boxes, id->name mapping
    """
    names = res.names  # dict id->name
    boxes = []
    if res.boxes is None:
        return boxes, names

    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls  = res.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        boxes.append(Box(cls=cls[i], conf=float(conf[i]), x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes, names

def evaluate_compliance(res, conf_min: float = 0.25, min_person_area: float = 0.0):
    boxes, names = _parse_ultralytics_result(res)
    boxes = [b for b in boxes if b.conf >= conf_min]

    name_to_id = {v: k for k, v in names.items()}
    PERSON = name_to_id["person"]
    HELM   = name_to_id["helmet"]
    NOH    = name_to_id["no-helmet"]
    VEST   = name_to_id["vest"]
    NOV    = name_to_id["no-vest"]

    persons_all = [b for b in boxes if b.cls == PERSON]

    # (A) filter tiny persons (far crowd)
    persons = []
    for p in persons_all:
        area = max(0.0, (p.x2 - p.x1)) * max(0.0, (p.y2 - p.y1))
        if area >= float(min_person_area):
            persons.append(p)

    items = [b for b in boxes if b.cls in (HELM, NOH, VEST, NOV)]

    def in_region(item: Box, person: Box, y0: float, y1: float) -> bool:
        """item center must be inside person box AND within vertical slice [y0,y1] of person height"""
        cx = (item.x1 + item.x2) / 2
        cy = (item.y1 + item.y2) / 2
        if not (person.x1 <= cx <= person.x2 and person.y1 <= cy <= person.y2):
            return False
        h = (person.y2 - person.y1)
        top = person.y1 + y0 * h
        bot = person.y1 + y1 * h
        return (top <= cy <= bot)

    per_person = []
    for idx, p in enumerate(sorted(persons, key=lambda b: (b.x1, b.y1)), start=1):
        # (B) Helmet only in top 40% of person
        helmet_cands = [it for it in items if it.cls in (HELM, NOH) and in_region(it, p, 0.0, 0.45)]
        # (C) Vest only in middle body region
        vest_cands   = [it for it in items if it.cls in (VEST, NOV) and in_region(it, p, 0.30, 0.95)]

        def pick_best(cands, positive_id, negative_id):
            if not cands:
                return ("unknown", 0.0)
            best_pos = max([c for c in cands if c.cls == positive_id], default=None, key=lambda x: x.conf)
            best_neg = max([c for c in cands if c.cls == negative_id], default=None, key=lambda x: x.conf)
            if best_pos is None and best_neg is None:
                return ("unknown", 0.0)
            if best_neg is None:
                return ("yes", best_pos.conf)
            if best_pos is None:
                return ("no", best_neg.conf)
            return ("yes", best_pos.conf) if best_pos.conf >= best_neg.conf else ("no", best_neg.conf)

        helmet_state, helmet_conf = pick_best(helmet_cands, HELM, NOH)
        vest_state, vest_conf     = pick_best(vest_cands, VEST, NOV)

        passed = (helmet_state == "yes") and (vest_state == "yes")

        per_person.append({
            "person_id": idx,
            "helmet": helmet_state,
            "helmet_conf": round(helmet_conf, 3),
            "vest": vest_state,
            "vest_conf": round(vest_conf, 3),
            "status": "PASS" if passed else "FAIL",
        })

    # IMPORTANT: Summary should be per-person based, not raw detection counts
    
    summary = {
    "total_person": len(persons),
    "pass": sum(1 for r in per_person if r["status"] == "PASS"),
    "fail": sum(1 for r in per_person if r["status"] == "FAIL"),
    "helmet_yes": sum(1 for r in per_person if r["helmet"] == "yes"),
    "helmet_no":  sum(1 for r in per_person if r["helmet"] == "no"),
    "helmet_unknown": sum(1 for r in per_person if r["helmet"] == "unknown"),
    "vest_yes":   sum(1 for r in per_person if r["vest"] == "yes"),
    "vest_no":    sum(1 for r in per_person if r["vest"] == "no"),
    "vest_unknown": sum(1 for r in per_person if r["vest"] == "unknown"),
}

    summary["safety_score"] = round((summary["pass"] / summary["total_person"] * 100), 2) if summary["total_person"] else 0.0

    df = pd.DataFrame(per_person)
    return summary, df
