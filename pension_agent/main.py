"""실행 진입점: python -m pension_agent [--month 2026-12] [--no-news]"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

from . import allocator as al
from . import ledger
from .data import REQUIRED_HOSTS, DataError, fred_many, latest_news, macro_fallback, names_match, naver_history
from .macro import FRED_SERIES, analyze
from .report import ReportInput, render, snapshot

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fetch_prices(universe: list[dict]) -> tuple[dict, list[str], set[str]]:
    prices, warnings, bad = {}, [], set()
    for e in universe:
        try:
            hist = naver_history(e["code"])
        except DataError as exc:
            warnings.append(f"{e['name']}({e['code']}) 시세 수집 실패: {exc}")
            bad.add(e["code"])
            if not prices and len(bad) >= 3:  # 처음부터 연속 실패 → 네트워크 문제로 보고 중단
                break
            continue
        if hist.name and not names_match(e["name"], hist.name):
            warnings.append(
                f"{e['code']}의 실제 종목명은 '{hist.name}'으로 설정('{e['name']}')과 달라 추천에서 제외했습니다. "
                "config/etf_universe.yaml을 확인하세요."
            )
            bad.add(e["code"])
            continue
        prices[e["code"]] = hist.close
    return prices, warnings, bad


def run(month: str | None = None, with_news: bool = True, out_dir: Path | None = None,
        assume_last_plan: bool = False, holdings_file: Path | None = None) -> Path:
    profile = load_yaml(ROOT / "config" / "profile.yaml")
    universe_all = load_yaml(ROOT / "config" / "etf_universe.yaml")["etfs"]
    today = dt.date.today()
    month = month or today.strftime("%Y-%m")
    out_dir = out_dir or ROOT / "reports"
    holdings_file = holdings_file or ROOT / "config" / "holdings.yaml"
    ledger_notes = ledger.apply_previous_plans(holdings_file, out_dir, month) if assume_last_plan else []
    holdings = ledger.load(holdings_file)

    prices, warnings, bad = fetch_prices(universe_all)
    if not prices:
        raise DataError(
            "시세를 하나도 받지 못했습니다. 네트워크에서 다음 도메인 접속이 허용되어 있는지 확인하세요: "
            + ", ".join(REQUIRED_HOSTS)
        )
    universe = [e for e in universe_all if e["code"] not in bad]
    if al.CASH_CODE not in {e["code"] for e in universe}:
        raise DataError("현금성 ETF(357870) 시세가 없어 계산할 수 없습니다.")

    series, fred_err = fred_many(list(FRED_SERIES))
    if fred_err:
        warnings.append(f"FRED에서 {len(fred_err)}개 지표를 받지 못했습니다: " + ", ".join(e.split(":")[0] for e in fred_err))
        extra, notes = macro_fallback(series)
        series.update(extra)
        warnings += [f"대체 데이터: {n}" for n in notes]
    macro = analyze(series) if series else None

    tac_cfg = profile["tactical"]
    core, core_notes = al.core_weights(universe, macro)
    tactical = al.tactical_weights(
        universe, prices, macro, top_n=tac_cfg["top_n"], lookbacks=tac_cfg["lookbacks"], ma=tac_cfg["trend_ma"]
    )

    lw, sw = profile["split"]["long_term"], profile["split"]["short_term"]
    combined = {c: lw * core.get(c, 0) + sw * tactical.weights.get(c, 0) for c in set(core) | set(tactical.weights)}
    risk = profile["risk"]
    cap = risk["irp_risky_limit"] - risk["irp_risky_buffer"]
    rets = al.build_returns(prices, universe, list(combined), profile["risk_model"]["history_days"])
    stats_before = al.portfolio_stats(combined, rets) if len(rets) > 250 else None
    k, risk_notes = al.fit_risk(combined, universe, rets if len(rets) > 250 else None, risk["max_drawdown"], cap)
    if stats_before is None:
        warnings.append("과거 데이터가 부족해 최대 손실 시뮬레이션을 하지 못했습니다.")

    final_core = al.scale_risky(core, universe, k)
    final_tac = al.scale_risky(tactical.weights, universe, k)
    final = {c: lw * final_core.get(c, 0) + sw * final_tac.get(c, 0) for c in set(final_core) | set(final_tac)}
    stats_after = al.portfolio_stats(final, rets) if stats_before else None

    last_prices = {c: float(s.dropna().iloc[-1]) for c, s in prices.items()}
    price_date = max(s.index[-1] for s in prices.values()).date().isoformat()
    plans = []
    for acct in profile["accounts"]:
        h = holdings.get(acct["id"], {}) or {}
        plans.append(
            al.buy_plan(
                acct["id"], final, last_prices, universe, acct["monthly_contribution"],
                holdings={str(c): v for c, v in (h.get("shares") or {}).items()},
                cash=float(h.get("cash", 0) or 0), risky_cap=cap,
                max_lines=acct.get("max_etfs_per_month"),
            )
        )

    news = []
    if with_news:
        news, news_err = latest_news()
        warnings += news_err

    inp = ReportInput(
        as_of=today, universe=universe, profile=profile, macro=macro, core=core, core_notes=core_notes,
        tactical=tactical, final_core=final_core, final_tactical=final_tac, risk_notes=risk_notes,
        stats_before=stats_before, stats_after=stats_after, plans=plans, news=news,
        warnings=warnings, price_date=price_date, ledger_notes=ledger_notes,
        has_holdings=any((holdings.get(a["id"]) or {}).get("shares") for a in profile["accounts"]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{month}.md"
    md_path.write_text(render(inp), encoding="utf-8")
    (out_dir / f"{month}.json").write_text(
        json.dumps(snapshot(inp), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return md_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="개인연금(IRP) 월간 투자 리포트 생성")
    ap.add_argument("--month", help="리포트 파일명에 쓸 연월 (기본: 이번 달, 예: 2026-12)")
    ap.add_argument("--no-news", action="store_true", help="뉴스 수집 생략")
    ap.add_argument("--out", type=Path, help="출력 폴더 (기본: reports/)")
    ap.add_argument("--assume-last-plan", action="store_true",
                    help="지난 리포트의 매수 계획대로 샀다고 가정해 config/holdings.yaml을 갱신한 뒤 계산")
    args = ap.parse_args(argv)
    try:
        path = run(args.month, not args.no_news, args.out, args.assume_last_plan)
    except DataError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    print(f"리포트 생성: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
