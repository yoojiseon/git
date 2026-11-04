"""장기(전략) + 단기(전술) 바구니 비중 계산, 위험 점검, 월 매수 계획."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .macro import MacroView

CASH_CODE = "357870"
SHORT_BOND_CODE = "114260"
FALLBACK = {"risky": "360750", "safe": "114260"}


# ---------------------------------------------------------------- 유니버스 헬퍼
def tilt_group(etf: dict) -> str:
    ac = etf["asset_class"]
    if ac == "금":
        return "gold"
    if ac.startswith("현금성"):
        return "cash"
    if "장기채" in ac:
        return "long_bond"
    if etf["irp"] == "safe":
        return "short_bond"
    return "equity"


def by_code(universe: list[dict]) -> dict[str, dict]:
    return {e["code"]: e for e in universe}


# ---------------------------------------------------------------- 장기 바구니
def core_weights(universe: list[dict], macro: MacroView | None) -> tuple[dict[str, float], list[str]]:
    w = {e["code"]: float(e["core_weight"]) for e in universe if "core" in e.get("roles", []) and e.get("core_weight")}
    notes: list[str] = []
    if macro is None:
        return _normalize(w), ["경제 지표가 없어 기본 비중을 그대로 사용했습니다."]

    u = by_code(universe)
    groups: dict[str, list[str]] = {}
    for code in w:
        groups.setdefault(tilt_group(u[code]), []).append(code)

    labels = {"equity": "주식", "long_bond": "장기채", "short_bond": "단기·종합채", "gold": "금", "cash": "현금성"}
    for g, delta in macro.tilts.items():
        if g == "kr_equity_shift" or not delta or g not in groups:
            continue
        members = groups[g]
        base = sum(w[c] for c in members)
        for c in members:
            share = w[c] / base if base > 0 else 1 / len(members)
            w[c] = max(0.0, w[c] + delta * share)
        notes.append(f"{labels[g]} {delta:+.0f}%p ({macro.regime})")

    shift = macro.tilts.get("kr_equity_shift", 0)
    if shift and "360750" in w and "069500" in w:
        moved = min(shift, w["360750"])
        w["360750"] -= moved
        w["069500"] += moved
        notes.append(f"원화 약세 구간이라 환노출 미국주식 → 국내주식 {moved:.0f}%p 이동")

    return _normalize(w), notes


def _normalize(w: dict[str, float]) -> dict[str, float]:
    total = sum(w.values())
    return {k: v / total for k, v in w.items() if v > 0} if total > 0 else {}


# ---------------------------------------------------------------- 모멘텀
def momentum_table(prices: dict[str, pd.Series], codes: list[str], lookbacks: list[int], ma: int) -> pd.DataFrame:
    rows = []
    for code in codes:
        s = prices.get(code)
        if s is None or len(s) < min(lookbacks) + 1:
            continue
        s = s.dropna()
        rets = {}
        for lb in lookbacks:
            rets[lb] = float(s.iloc[-1] / s.iloc[-lb - 1] - 1) if len(s) > lb else np.nan
        valid = [v for v in rets.values() if not np.isnan(v)]
        daily = s.pct_change().dropna()
        rows.append({
            "code": code,
            "r1m": float(s.iloc[-1] / s.iloc[-22] - 1) if len(s) > 22 else np.nan,
            **{f"r{lb}": rets[lb] for lb in lookbacks},
            "score": float(np.mean(valid)) if len(valid) >= 2 else np.nan,
            "vol": float(daily.iloc[-63:].std() * math.sqrt(252)),
            "above_ma": bool(len(s) >= ma and s.iloc[-1] > s.iloc[-ma:].mean()),
            "price": float(s.iloc[-1]),
            "history_days": len(s),
        })
    return pd.DataFrame(rows).set_index("code") if rows else pd.DataFrame()


@dataclass
class TacticalPick:
    code: str
    weight: float
    reason: str


@dataclass
class TacticalResult:
    weights: dict[str, float]
    picks: list[TacticalPick]
    rejected: dict[str, str]
    table: pd.DataFrame
    hurdle: float


def tactical_weights(
    universe: list[dict],
    prices: dict[str, pd.Series],
    macro: MacroView | None,
    top_n: int = 3,
    lookbacks: list[int] | None = None,
    ma: int = 200,
    slot_target_vol: float = 0.15,
    max_corr: float = 0.85,
) -> TacticalResult:
    lookbacks = lookbacks or [63, 126, 252]
    u = by_code(universe)
    cands = [e["code"] for e in universe if "tactical" in e.get("roles", [])]
    table = momentum_table(prices, cands, lookbacks, ma)
    hurdle = float(table.loc[CASH_CODE, "score"]) if CASH_CODE in table.index and not np.isnan(table.loc[CASH_CODE, "score"]) else 0.0

    slots = top_n
    if macro is not None and macro.regime == "위기·고변동":
        slots = max(1, top_n - 1)

    rejected: dict[str, str] = {}
    picks: list[TacticalPick] = []
    weights: dict[str, float] = {}
    rets = pd.DataFrame({c: prices[c].pct_change() for c in table.index if c in prices}).iloc[-252:]

    risky = [c for c in table.index if u[c]["irp"] == "risky"]
    risky = sorted(risky, key=lambda c: -np.nan_to_num(table.loc[c, "score"], nan=-9))
    for code in risky:
        row = table.loc[code]
        if np.isnan(row["score"]):
            rejected[code] = "상장 기간이 짧아 모멘텀을 계산할 수 없음"
            continue
        if row["score"] <= hurdle:
            rejected[code] = f"평균 수익률 {row['score']:+.1%}로 현금성 ETF({hurdle:+.1%})보다 낮음"
            continue
        if not row["above_ma"]:
            rejected[code] = f"현재가가 {ma}일 이동평균 아래(하락 추세)"
            continue
        if len(picks) >= slots:
            rejected[code] = "조건은 통과했지만 상위 종목에 밀림"
            continue
        corr_hit = next(
            (p.code for p in picks if rets[[code, p.code]].dropna().corr().iloc[0, 1] > max_corr), None
        )
        if corr_hit:
            rejected[code] = f"이미 고른 {u[corr_hit]['name']}와 너무 비슷하게 움직임(상관계수 > {max_corr})"
            continue
        w = (1 / top_n) * min(1.0, slot_target_vol / row["vol"]) if row["vol"] > 0 else 1 / top_n
        weights[code] = w
        reason = (
            f"3·6·12개월 평균 수익률 {row['score']:+.1%}(현금 대비 {row['score'] - hurdle:+.1%}p), "
            f"{ma}일선 위 상승 추세"
        )
        if w < 1 / top_n - 1e-9:
            reason += f", 변동성 {row['vol']:.0%}가 커서 비중 축소"
        picks.append(TacticalPick(code, w, reason))

    remainder = 1.0 - sum(weights.values())
    if remainder > 1e-9:
        inflation = macro.inflation_high if macro else False
        safe = []
        for c in table.index:
            if u[c]["irp"] != "safe" or c == CASH_CODE:
                continue
            row = table.loc[c]
            if inflation and "장기채" in u[c]["asset_class"]:
                rejected[c] = "물가가 높은 구간이라 장기채 제외"
                continue
            if np.isnan(row["score"]) or row["score"] <= hurdle or not row["above_ma"]:
                rejected[c] = "채권 추세가 현금성 ETF보다 약함"
                continue
            safe.append(c)
        safe.sort(key=lambda c: -table.loc[c, "score"])
        if safe:
            best = safe[0]
            weights[best] = weights.get(best, 0) + remainder / 2
            picks.append(TacticalPick(best, remainder / 2, f"방어 자산 중 추세가 가장 좋음(평균 {table.loc[best, 'score']:+.1%})"))
            remainder /= 2
        weights[CASH_CODE] = weights.get(CASH_CODE, 0) + remainder
        picks.append(TacticalPick(CASH_CODE, remainder, "조건을 통과한 위험자산이 부족한 몫은 현금성 ETF에 대기"))

    return TacticalResult(weights, picks, rejected, table, hurdle)


# ---------------------------------------------------------------- 위험 점검
def build_returns(prices: dict[str, pd.Series], universe: list[dict], codes: list[str], days: int) -> pd.DataFrame:
    u = by_code(universe)
    raw = pd.DataFrame({c: s.pct_change() for c, s in prices.items()}).sort_index()
    raw = raw.iloc[-days:]
    out = {}
    for c in codes:
        r = raw[c] if c in raw else pd.Series(np.nan, index=raw.index)
        proxy = u[c].get("proxy")
        if proxy and proxy in raw:
            r = r.fillna(raw[proxy])
        fb = FALLBACK[u[c]["irp"]]
        if c == CASH_CODE:
            r = r.fillna(0.0)
        elif fb in raw:
            r = r.fillna(raw[fb])
        out[c] = r
    return pd.DataFrame(out).dropna()


def portfolio_stats(weights: dict[str, float], rets: pd.DataFrame) -> dict[str, float]:
    w = pd.Series(weights).reindex(rets.columns).fillna(0)
    port = rets @ w
    cum = (1 + port).cumprod()
    dd = cum / cum.cummax() - 1
    years = len(port) / 252
    return {
        "mdd": float(dd.min()),
        "mdd_date": str(dd.idxmin().date()) if len(dd) else "",
        "vol": float(port.std() * math.sqrt(252)),
        "cagr": float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0,
        "start": str(rets.index[0].date()) if len(rets) else "",
        "end": str(rets.index[-1].date()) if len(rets) else "",
    }


def risky_share(weights: dict[str, float], universe: list[dict]) -> float:
    u = by_code(universe)
    return sum(w for c, w in weights.items() if u[c]["irp"] == "risky")


def scale_risky(weights: dict[str, float], universe: list[dict], k: float) -> dict[str, float]:
    """위험자산을 k배로 줄이고, 줄인 몫은 현금성 ETF와 국고채3년에 반씩 나눕니다."""
    u = by_code(universe)
    out, freed = {}, 0.0
    for c, w in weights.items():
        if u[c]["irp"] == "risky":
            out[c] = w * k
            freed += w * (1 - k)
        else:
            out[c] = w
    if freed > 0:
        if SHORT_BOND_CODE in u:
            out[SHORT_BOND_CODE] = out.get(SHORT_BOND_CODE, 0) + freed / 2
            freed /= 2
        out[CASH_CODE] = out.get(CASH_CODE, 0) + freed
    return {c: w for c, w in out.items() if w > 1e-12}


def fit_risk(weights: dict[str, float], universe: list[dict], rets: pd.DataFrame | None,
             max_dd: float, risky_cap: float) -> tuple[float, list[str]]:
    """MDD 목표와 IRP 한도를 지키는 위험자산 축소 배율 k(0~1)를 찾습니다."""
    notes = []
    k = 1.0
    rs = risky_share(weights, universe)
    if rs > risky_cap:
        k = risky_cap / rs
        notes.append(f"IRP 위험자산 한도 때문에 위험자산을 {1 - k:.0%} 줄였습니다.")
    if rets is not None and len(rets) > 250:
        target = max_dd * 0.95  # 5% 여유
        if -portfolio_stats(scale_risky(weights, universe, k), rets)["mdd"] > target:
            lo, hi = 0.0, k
            for _ in range(30):
                mid = (lo + hi) / 2
                if -portfolio_stats(scale_risky(weights, universe, mid), rets)["mdd"] > target:
                    hi = mid
                else:
                    lo = mid
            notes.append(
                f"과거 데이터로 시뮬레이션한 최대 손실이 목표(-{max_dd:.0%})를 넘어서 "
                f"위험자산을 {1 - lo / k:.0%} 줄이고 그만큼 현금성 ETF와 국고채3년으로 옮겼습니다."
            )
            k = lo
    return k, notes


# ---------------------------------------------------------------- 매수 계획
@dataclass
class BuyLine:
    code: str
    shares: int
    price: float

    @property
    def amount(self) -> float:
        return self.shares * self.price


@dataclass
class BuyPlan:
    account_id: str
    contribution: float
    lines: list[BuyLine] = field(default_factory=list)
    leftover: float = 0.0
    risky_after: float = 0.0
    value_after: float = 0.0


def buy_plan(
    account_id: str,
    target: dict[str, float],
    prices: dict[str, float],
    universe: list[dict],
    contribution: float,
    holdings: dict[str, float] | None = None,
    cash: float = 0.0,
    risky_cap: float = 0.67,
    max_lines: int | None = None,
) -> BuyPlan:
    """새 납입금으로 목표 비중보다 부족한 종목을 채웁니다(매도 없음, 1주 단위).

    max_lines를 주면 부족분이 큰 순서대로 그 개수만큼만 삽니다. 나머지 종목은 다음 달로 넘어갑니다.
    """
    u = by_code(universe)
    holdings = holdings or {}
    codes = [c for c in target if c in prices and prices[c] > 0]
    cur = {c: holdings.get(c, 0) * prices[c] for c in codes}
    other = sum(holdings.get(c, 0) * prices[c] for c in holdings if c not in codes and c in prices)
    budget = contribution + cash
    total = sum(cur.values()) + other + budget

    deficit = {c: max(target[c] * total - cur[c], 0.0) for c in codes}
    if max_lines:
        keep = set(sorted(deficit, key=deficit.get, reverse=True)[:max_lines])
        deficit = {c: (d if c in keep else 0.0) for c, d in deficit.items()}
    dsum = sum(deficit.values())
    scale = min(1.0, budget / dsum) if dsum > 0 else 0.0
    shares = {c: int((deficit[c] * scale) // prices[c]) for c in codes}

    def risky_value() -> float:
        return sum((holdings.get(c, 0) + shares.get(c, 0)) * prices[c] for c in codes if u[c]["irp"] == "risky") + sum(
            holdings.get(c, 0) * prices[c] for c in holdings if c not in codes and c in prices and u.get(c, {}).get("irp") == "risky"
        )

    # IRP 한도 초과분 정리
    while risky_value() > risky_cap * total:
        rc = [c for c in codes if u[c]["irp"] == "risky" and shares[c] > 0]
        if not rc:
            break
        worst = max(rc, key=lambda c: (cur[c] + shares[c] * prices[c]) - target[c] * total)
        shares[worst] -= 1

    left = budget - sum(shares[c] * prices[c] for c in codes)
    # 남은 돈으로 부족분이 가장 큰 종목부터 1주씩 추가
    while True:
        gaps = {
            c: target[c] * total - (cur[c] + shares[c] * prices[c])
            for c in codes
            if prices[c] <= left and (not max_lines or c in keep)
        }
        gaps = {c: g for c, g in gaps.items() if g > 0}
        if not gaps:
            break
        pick = max(gaps, key=gaps.get)
        if u[pick]["irp"] == "risky" and risky_value() + prices[pick] > risky_cap * total:
            gaps.pop(pick)
            safe_gaps = {c: g for c, g in gaps.items() if u[c]["irp"] == "safe"}
            if not safe_gaps:
                break
            pick = max(safe_gaps, key=safe_gaps.get)
        shares[pick] += 1
        left -= prices[pick]

    lines = [BuyLine(c, shares[c], prices[c]) for c in codes if shares[c] > 0]
    lines.sort(key=lambda l: -l.amount)
    plan = BuyPlan(account_id, contribution, lines, left)
    plan.value_after = total
    plan.risky_after = risky_value() / total if total else 0.0
    return plan
