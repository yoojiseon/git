"""마크다운 리포트 생성."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from .allocator import BuyPlan, TacticalResult, by_code, tilt_group
from .data import NewsItem
from .macro import MacroView

ROLE_TEXT = {
    "equity": "장기 성장 엔진. 연금 수령까지 25년 넘게 남아 있어 주식이 수익의 대부분을 만들어 줍니다.",
    "gold": "분산·위기 방어. 주식과 다르게 움직여 큰 하락장에서 손실을 줄여 줍니다. 최근 하락처럼 혼자 크게 흔들릴 수 있어 10% 안팎만 담습니다.",
    "long_bond": "경기침체 보험. 침체가 오면 금리가 내려가 가격이 오르므로 주식 손실을 일부 메워 줍니다.",
    "short_bond": "안정적인 이자 수입. 가격 변동이 작아 최대 손실 -10% 목표를 지키는 바탕이 됩니다.",
    "cash": "완충 장치이자 대기 자금. 하락장에서 싸게 살 수 있는 여력이 됩니다.",
}

COMMENTARY_MARKER = "<!-- CLAUDE_COMMENTARY -->"


@dataclass
class ReportInput:
    as_of: dt.date
    universe: list[dict]
    profile: dict
    macro: MacroView | None
    core: dict[str, float]
    core_notes: list[str]
    tactical: TacticalResult
    final_core: dict[str, float]
    final_tactical: dict[str, float]
    risk_notes: list[str]
    stats_before: dict | None
    stats_after: dict | None
    plans: list[BuyPlan]
    news: list[NewsItem]
    warnings: list[str] = field(default_factory=list)
    price_date: str = ""
    ledger_notes: list[str] = field(default_factory=list)
    has_holdings: bool = False


def _pct(x: float, digits: int = 1) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


def _won(x: float) -> str:
    return f"{x:,.0f}원"


def render(inp: ReportInput) -> str:
    u = by_code(inp.universe)
    p = inp.profile
    long_w, short_w = p["split"]["long_term"], p["split"]["short_term"]
    total = {c: long_w * inp.final_core.get(c, 0) + short_w * inp.final_tactical.get(c, 0)
             for c in set(inp.final_core) | set(inp.final_tactical)}
    risky_total = sum(w for c, w in total.items() if u[c]["irp"] == "risky")
    years_left = p["household"]["pension_start_age"] - p["household"]["current_age"]

    L: list[str] = []
    a = L.append
    a(f"# 개인연금(IRP) 월간 투자 리포트 — {inp.as_of:%Y년 %m월}")
    a("")
    a(f"> 생성일 {inp.as_of.isoformat()} · 시세 기준일 {inp.price_date or '-'} · "
      f"연금 수령까지 약 {years_left}년 · 목표 최대 손실 -{p['risk']['max_drawdown']:.0%} · "
      f"장기 {long_w:.0%} : 단기 {short_w:.0%}")
    a(">")
    a("> 이 리포트는 공개 데이터와 정해진 규칙으로 만든 **공부용 참고 자료**이며 투자 권유가 아닙니다. "
      "매수 전에 증권사 앱에서 IRP 매수 가능 여부와 상품 설명서를 확인하세요.")
    a("")

    # 1. 요약
    a("## 1. 이번 달 할 일")
    a("")
    if inp.macro:
        a(f"- **경제 국면: {inp.macro.regime}** — {inp.macro.regime_desc}")
    a(f"- 목표 포트폴리오: 위험자산 {_pct(risky_total, 0)} / 안전자산 {_pct(1 - risky_total, 0)} "
      f"(IRP 한도 위험자산 {p['risk']['irp_risky_limit']:.0%} 이하 ✅)")
    if inp.stats_after:
        a(f"- 과거 시뮬레이션 최대 손실: {_pct(inp.stats_after['mdd'])} "
          f"({inp.stats_after['start']} ~ {inp.stats_after['end']}, 목표 -{p['risk']['max_drawdown']:.0%})")
    a("")
    acct = {x["id"]: x for x in p["accounts"]}
    for plan in inp.plans:
        info = acct[plan.account_id]
        a(f"### {info['label']} — 이번 달 {_won(plan.contribution)}")
        a("")
        if not plan.lines:
            a("매수할 종목이 없습니다(가격 데이터 부족).")
        else:
            a("| 종목 | 코드 | 수량 | 예상 금액 | 바구니 | 분류 |")
            a("|---|---|---:|---:|---|---|")
            for line in plan.lines:
                basket = []
                if inp.final_core.get(line.code, 0) > 0:
                    basket.append("장기")
                if inp.final_tactical.get(line.code, 0) > 0:
                    basket.append("단기")
                cls = "위험" if u[line.code]["irp"] == "risky" else "안전"
                a(f"| {u[line.code]['name']} | {line.code} | {line.shares}주 | {_won(line.amount)} | "
                  f"{'·'.join(basket)} | {cls} |")
            a("")
            a(f"남는 돈 {_won(plan.leftover)}은 계좌에 현금으로 두었다가 다음 달 납입금과 합쳐 매수합니다. "
              f"매수 후 이 계좌의 위험자산 비중은 {_pct(plan.risky_after, 0)}입니다.")
        a("")
    a("> 매수 수량은 '현재 보유 종목 + 이번 달 납입금' 기준으로 목표 비중보다 부족한 종목부터 채웁니다. "
      "한 달에 사는 종목 수를 제한해서, 이번 달에 빠진 종목은 다음 달에 먼저 채워집니다.")
    for n in inp.ledger_notes:
        a(f"> - {n} 실제와 다르면 `config/holdings.yaml`을 고치거나 Claude에게 알려 주세요.")
    if not inp.has_holdings:
        a("> - 아직 보유 수량 장부가 비어 있어 '처음 시작' 기준으로 계산했습니다.")
    a("")

    # 2. 경제 상황
    a("## 2. 경제 상황 점검")
    a("")
    if inp.macro:
        a("| 지표 | 현재 | 판단 | 해석 |")
        a("|---|---|:---:|---|")
        for s in inp.macro.signals:
            mark = "🟢" if s.score > 0 else "🔴" if s.score < 0 else "⚪"
            a(f"| {s.name} | {s.value} | {mark} | {s.comment} |")
        a("")
        a(f"**종합:** 성장 점수 {inp.macro.growth:+.2f}, 시장 스트레스 점수 {inp.macro.stress:+.2f}"
          f"(+는 양호, -는 경계), 물가 부담 {'있음' if inp.macro.inflation_high else '낮음'} → **{inp.macro.regime}**")
        if inp.core_notes:
            a("")
            a("장기 바구니 조정: " + " / ".join(inp.core_notes))
        if inp.macro.missing:
            a("")
            a(f"_수집하지 못한 지표: {', '.join(inp.macro.missing)}_")
    else:
        a("경제 지표를 수집하지 못해 기본 비중을 사용했습니다.")
    a("")

    # 3. 목표 포트폴리오
    a("## 3. 목표 포트폴리오")
    a("")
    a("| 종목 | 자산군 | IRP 분류 | 장기 바구니 | 단기 바구니 | 전체 비중 |")
    a("|---|---|---|---:|---:|---:|")
    for c in sorted(total, key=lambda c: -total[c]):
        if total[c] < 0.0005:
            continue
        a(f"| {u[c]['name']} | {u[c]['asset_class']} | {'위험' if u[c]['irp'] == 'risky' else '안전'} | "
          f"{_pct(inp.final_core.get(c, 0))} | {_pct(inp.final_tactical.get(c, 0))} | **{_pct(total[c])}** |")
    a("")
    a("_장기·단기 바구니 비중은 각 바구니 안에서의 비율이고, 전체 비중 = 장기 × "
      f"{long_w:.0%} + 단기 × {short_w:.0%}입니다._")
    a("")
    a("### 위험 점검")
    a("")
    if inp.stats_before and inp.stats_after:
        a("| 구분 | 최대 손실(MDD) | 연 변동성 | 연평균 수익률 |")
        a("|---|---:|---:|---:|")
        a(f"| 조정 전 | {_pct(inp.stats_before['mdd'])} | {_pct(inp.stats_before['vol'])} | {_pct(inp.stats_before['cagr'])} |")
        a(f"| **조정 후(최종)** | **{_pct(inp.stats_after['mdd'])}** | {_pct(inp.stats_after['vol'])} | {_pct(inp.stats_after['cagr'])} |")
        a("")
        a(f"- 기간: {inp.stats_after['start']} ~ {inp.stats_after['end']}(최대 손실 시점 {inp.stats_after['mdd_date']}). "
          "상장 기간이 짧은 종목은 비슷한 종목의 과거 수익률로 채웠습니다.")
        a("- 과거 최대 손실이 미래를 보장하지는 않습니다. 실제 위기에서는 이보다 크게 떨어질 수 있습니다.")
    for n in inp.risk_notes:
        a(f"- {n}")
    a(f"- 목표 -{p['risk']['max_drawdown']:.0%}는 연금 기간(약 {years_left}년)에 비해 보수적인 편입니다. "
      "손실을 조금 더 견딜 수 있다면 `config/profile.yaml`의 `max_drawdown`을 0.15로 올려 주식 비중을 늘릴 수 있습니다.")
    a("")

    # 4. 장기 바구니
    a(f"## 4. 장기 바구니 ({long_w:.0%}) — 꾸준히 모아 가는 종목")
    a("")
    a("매달 같은 비율로 사서 오래 들고 가는 바구니입니다. 1년에 한두 번 비중이 목표에서 5%p 이상 벗어나면 맞춰 줍니다(리밸런싱).")
    a("")
    for c in sorted(inp.final_core, key=lambda c: -inp.final_core[c]):
        e = u[c]
        a(f"### {e['name']} ({c}) — 바구니 내 {_pct(inp.final_core[c])}")
        a("")
        a(f"- **이게 뭔가요?** {e['desc']}")
        a(f"- **왜 담나요?** {ROLE_TEXT[tilt_group(e)]}")
        if e.get("watch"):
            a(f"- **공부 포인트:** {e['watch']}")
        a("")

    # 5. 단기 바구니
    t = inp.tactical
    a(f"## 5. 단기 바구니 ({short_w:.0%}) — 3개월~1년 추세 추종")
    a("")
    a("매달 후보 ETF를 다시 평가해서 **최근 3·6·12개월 수익률이 좋고(모멘텀) 상승 추세(200일 이동평균 위)인 종목**을 "
      f"최대 {p['tactical']['top_n']}개 고릅니다. 조건을 통과한 종목이 부족하면 그 몫은 채권이나 현금성 ETF로 옮겨 하락장을 피합니다. "
      "이미 고른 종목과 너무 비슷하게 움직이는 종목은 빼서 한쪽으로 몰리지 않게 합니다.")
    a("")
    a("### 이번 달 선정")
    a("")
    a("| 종목 | 단기 바구니 비중(조정 후) | 선정 이유 |")
    a("|---|---:|---|")
    for pick in t.picks:
        a(f"| {u[pick.code]['name']} | {_pct(inp.final_tactical.get(pick.code, 0))} | {pick.reason} |")
    a("")
    for pick in t.picks:
        e = u[pick.code]
        if pick.code in inp.final_core:
            continue
        a(f"- **{e['name']}**: {e['desc']} _공부 포인트: {e.get('watch', '-')}_")
    a("")
    if not t.table.empty:
        a("### 후보 순위표")
        a("")
        a(f"현금성 ETF 기준 수익률(허들): {_pct(t.hurdle)}")
        a("")
        a("| 순위 | 종목 | 1개월 | 3개월 | 6개월 | 12개월 | 평균 점수 | 200일선 | 변동성 | 결과 |")
        a("|---:|---|---:|---:|---:|---:|---:|:---:|---:|---|")
        tbl = t.table.sort_values("score", ascending=False, na_position="last")
        lbs = p["tactical"]["lookbacks"]
        picked = {x.code for x in t.picks}
        for i, (c, row) in enumerate(tbl.iterrows(), 1):
            result = "✅ 선정" if c in picked else t.rejected.get(c, "-")
            a(f"| {i} | {u[c]['name']} | {_pct(row['r1m'])} | {_pct(row[f'r{lbs[0]}'])} | {_pct(row[f'r{lbs[1]}'])} | "
              f"{_pct(row[f'r{lbs[2]}'])} | {_pct(row['score'])} | {'위' if row['above_ma'] else '아래'} | "
              f"{_pct(row['vol'], 0)} | {result} |")
        a("")

    # 6. 공부할 후보 리스트
    a("## 6. 공부할 후보 ETF 전체 목록")
    a("")
    a("| 종목 | 코드 | 자산군 | IRP 분류 | 용도 | 한 줄 설명 |")
    a("|---|---|---|---|---|---|")
    for e in inp.universe:
        if e.get("roles") == ["proxy"]:
            continue
        roles = "·".join({"core": "장기", "tactical": "단기"}[r] for r in e["roles"] if r != "proxy")
        short = e["desc"].split(".")[0].strip() + "."
        a(f"| {e['name']} | {e['code']} | {e['asset_class']} | {'위험' if e['irp'] == 'risky' else '안전'} | {roles} | {short} |")
    a("")
    a("**IRP 규칙:** 위험자산(주식형·금) 합계는 70% 이하. 채권형, 현금성, 주식 40% 미만 채권혼합형은 안전자산입니다. "
      "레버리지·인버스와 선물 기반 파생형 ETF는 IRP에서 살 수 없습니다.")
    a("")

    # 7. 뉴스
    a("## 7. 최근 2주 주요 뉴스")
    a("")
    if inp.news:
        topic = None
        for n in inp.news:
            if n.topic != topic:
                topic = n.topic
                a("")
                a(f"**{topic}**")
                a("")
            src = f" — {n.source}" if n.source else ""
            a(f"- [{n.title}]({n.link}){src}")
    else:
        a("뉴스를 수집하지 못했습니다.")
    a("")

    # 8. 해설
    a("## 8. Claude 해설")
    a("")
    a(COMMENTARY_MARKER)
    a("_월간 자동 실행에서 Claude가 위 데이터와 최신 뉴스를 읽고 해설을 덧붙입니다._")
    a("")

    if inp.warnings:
        a("## 부록: 데이터 경고")
        a("")
        for w in inp.warnings:
            a(f"- {w}")
        a("")
    return "\n".join(L)


def snapshot(inp: ReportInput) -> dict:
    p = inp.profile
    u = by_code(inp.universe)
    return {
        "as_of": inp.as_of.isoformat(),
        "price_date": inp.price_date,
        "regime": inp.macro.regime if inp.macro else None,
        "indicators": inp.macro.indicators if inp.macro else {},
        "core": inp.final_core,
        "tactical": inp.final_tactical,
        "split": p["split"],
        "stats": inp.stats_after,
        "plans": [
            {
                "account": pl.account_id,
                "buys": [{"code": l.code, "name": u[l.code]["name"], "shares": l.shares, "price": l.price} for l in pl.lines],
                "leftover": pl.leftover,
            }
            for pl in inp.plans
        ],
        "tactical_table": (
            inp.tactical.table.replace({np.nan: None}).reset_index().to_dict(orient="records")
            if not inp.tactical.table.empty else []
        ),
        "warnings": inp.warnings,
    }

