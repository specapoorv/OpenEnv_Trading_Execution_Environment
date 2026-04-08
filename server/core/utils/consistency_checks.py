from __future__ import annotations


def price_consistent(bloomberg: dict, report: dict) -> bool:
    if bloomberg.get("ok") and report.get("ok") and "mid_price" in bloomberg and "mid_price" in report:
        return abs(float(bloomberg["mid_price"]) - float(report["mid_price"])) <= 0.75
    return True


def position_consistent(oms: dict, report: dict) -> bool:
    if oms.get("ok") and report.get("ok") and "position" in oms and "position" in report:
        return abs(int(oms["position"]) - int(report["position"])) <= 5
    return True
