"""보유 수량 장부(config/holdings.yaml) 관리.

매달 리포트의 매수 계획을 그대로 실행했다고 가정하고 보유 수량을 갱신합니다.
실제로 다르게 샀다면 holdings.yaml을 직접 고치거나 Claude에게 알려 주면 됩니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save(path: Path, holdings: dict) -> None:
    header = (
        "# 계좌별 보유 수량 장부 — 월간 실행 때 '지난달 매수 계획대로 샀다'고 가정해 자동 갱신됩니다.\n"
        "# 실제와 다르면 직접 고치세요. shares는 종목코드: 수량, cash는 계좌에 남은 현금(원)입니다.\n"
    )
    body = yaml.safe_dump(holdings, allow_unicode=True, sort_keys=False)
    path.write_text(header + body, encoding="utf-8")


def apply_previous_plans(holdings_path: Path, reports_dir: Path, current_month: str) -> list[str]:
    """current_month 이전 리포트 중 아직 반영하지 않은 매수 계획을 장부에 더합니다."""
    holdings = load(holdings_path)
    applied = set(holdings.get("applied_reports", []) or [])
    notes = []
    for snap_path in sorted(reports_dir.glob("*.json")):
        month = snap_path.stem
        if month >= current_month or month in applied or len(month) != 7:
            continue
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        for plan in snap.get("plans", []):
            acct = holdings.setdefault(plan["account"], {"cash": 0, "shares": {}})
            shares = acct.setdefault("shares", {}) or {}
            for buy in plan["buys"]:
                shares[buy["code"]] = int(shares.get(buy["code"], 0)) + int(buy["shares"])
            acct["shares"] = shares
            acct["cash"] = round(float(plan.get("leftover", 0)))
        applied.add(month)
        notes.append(f"{month} 리포트의 매수 계획을 실행했다고 가정해 보유 수량을 갱신했습니다.")
    if notes:
        holdings["applied_reports"] = sorted(applied)
        save(holdings_path, holdings)
    return notes
