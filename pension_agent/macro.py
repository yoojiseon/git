"""거시경제 지표로 현재 경제 국면을 판단하고 포트폴리오 조정 방향(틸트)을 정합니다.

규칙은 단순하고 설명 가능하게 만들었습니다. 지표마다 점수(-1 ~ +1)를 매기고 이유를 문장으로 남깁니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

FRED_SERIES = {
    "DGS10": "미국 10년 국채금리(%)",
    "DGS2": "미국 2년 국채금리(%)",
    "T10Y3M": "장단기 금리차 10년-3개월(%p)",
    "FEDFUNDS": "미국 기준금리(%)",
    "CPIAUCSL": "미국 소비자물가지수",
    "UNRATE": "미국 실업률(%)",
    "VIXCLS": "VIX 공포지수",
    "BAMLH0A0HYM2": "하이일드 신용스프레드(%p)",
    "DEXKOUS": "원·달러 환율(원)",
    "DFII10": "미국 10년 실질금리(%)",
    "IRLTLT01KRM156N": "한국 10년 국채금리(%)",
}


@dataclass
class Signal:
    name: str
    value: str
    score: float  # +는 위험자산에 우호적, -는 방어적
    comment: str


@dataclass
class MacroView:
    signals: list[Signal]
    growth: float
    inflation_high: bool
    stress: float
    easing: bool
    krw_weak: bool | None
    regime: str
    regime_desc: str
    tilts: dict[str, float] = field(default_factory=dict)
    indicators: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def _last(s: pd.Series | None) -> float | None:
    return None if s is None or s.empty else float(s.dropna().iloc[-1])


def _value_ago(s: pd.Series, days: int) -> float | None:
    s = s.dropna()
    target = s.index[-1] - pd.Timedelta(days=days)
    past = s[s.index <= target]
    return None if past.empty else float(past.iloc[-1])


def sahm_gap(unrate: pd.Series) -> float:
    """샴의 법칙: 실업률 3개월 평균이 직전 12개월 저점보다 0.5%p 이상 오르면 경기침체 신호."""
    m = unrate.resample("MS").last().dropna()
    avg3 = m.rolling(3).mean()
    low12 = avg3.shift(1).rolling(12).min()
    return float((avg3 - low12).dropna().iloc[-1])


def cpi_yoy(cpi: pd.Series) -> tuple[float, float]:
    m = cpi.resample("MS").last().dropna()
    yoy = (m.iloc[-1] / m.iloc[-13] - 1) * 100
    ann3m = ((m.iloc[-1] / m.iloc[-4]) ** 4 - 1) * 100
    return float(yoy), float(ann3m)


def analyze(series: dict[str, pd.Series]) -> MacroView:
    sig: list[Signal] = []
    ind: dict[str, float] = {}
    missing = [k for k in FRED_SERIES if k not in series]

    growth_scores, stress_scores = [], []

    # 1) 장단기 금리차
    if "T10Y3M" in series:
        s = series["T10Y3M"].dropna()
        cur = _last(s)
        ind["T10Y3M"] = cur
        inverted_recently = (s[s.index >= s.index[-1] - pd.Timedelta(days=540)] < 0).any()
        if cur < 0:
            sc, c = -0.5, "장단기 금리가 역전되어 있습니다. 과거 경기침체 1~2년 전에 자주 나타난 신호입니다."
        elif inverted_recently and cur < 1.0:
            sc, c = -0.5, "최근 역전이 풀렸습니다. 역사적으로 역전이 풀리는 구간에서 경기 둔화가 시작된 경우가 많아 경계합니다."
        else:
            sc, c = 0.5, "장단기 금리차가 정상(양수)이라 경기 확장과 맞는 모습입니다."
        growth_scores.append(sc)
        sig.append(Signal("장단기 금리차(10Y-3M)", f"{cur:+.2f}%p", sc, c))

    # 2) 고용 — 샴의 법칙
    if "UNRATE" in series:
        gap = sahm_gap(series["UNRATE"])
        cur = _last(series["UNRATE"])
        ind["UNRATE"], ind["SAHM_GAP"] = cur, gap
        if gap >= 0.5:
            sc, c = -1.0, f"샴 지표 {gap:.2f}%p로 경기침체 신호(0.5 이상)가 켜졌습니다."
        elif gap >= 0.3:
            sc, c = -0.5, f"샴 지표 {gap:.2f}%p로 고용이 식어 가고 있습니다(0.5 이상이면 침체 신호)."
        else:
            sc, c = 0.5, f"샴 지표 {gap:.2f}%p로 고용이 아직 탄탄합니다."
        growth_scores.append(sc)
        sig.append(Signal("미국 실업률 / 샴 지표", f"{cur:.1f}% / {gap:+.2f}%p", sc, c))

    # 3) 물가
    inflation_high = False
    if "CPIAUCSL" in series:
        yoy, ann3m = cpi_yoy(series["CPIAUCSL"])
        ind["CPI_YOY"], ind["CPI_3M_ANN"] = yoy, ann3m
        inflation_high = yoy >= 3.5 or (yoy >= 3.0 and ann3m > yoy)
        if inflation_high:
            sc, c = -0.5, f"물가 상승률 {yoy:.1f}%(최근 3개월 연율 {ann3m:.1f}%)로 높습니다. 장기채에 불리하고 금·단기채가 상대적으로 유리합니다."
        elif yoy <= 2.5:
            sc, c = 0.5, f"물가 상승률 {yoy:.1f}%로 목표(2%)에 가까워 금리 인하 여력이 있습니다."
        else:
            sc, c = 0.0, f"물가 상승률 {yoy:.1f}%(최근 3개월 연율 {ann3m:.1f}%)로 목표보다 조금 높은 수준입니다."
        sig.append(Signal("미국 소비자물가(전년 대비)", f"{yoy:.1f}%", sc, c))

    # 4) 통화정책 방향
    easing = False
    if "FEDFUNDS" in series:
        s = series["FEDFUNDS"].dropna()
        cur, ago = _last(s), _value_ago(s, 183)
        ind["FEDFUNDS"] = cur
        if ago is not None:
            chg = cur - ago
            easing = chg <= -0.25
            if easing:
                c = f"최근 6개월 기준금리가 {chg:+.2f}%p 바뀌어 인하 국면입니다. 채권 가격에 우호적입니다."
            elif chg >= 0.25:
                c = f"최근 6개월 기준금리가 {chg:+.2f}%p 바뀌어 인상 국면입니다. 주식·채권 모두 부담입니다."
            else:
                c = "최근 6개월 기준금리 변화가 거의 없습니다(동결 국면)."
            sig.append(Signal("미국 기준금리", f"{cur:.2f}% (6개월 {chg:+.2f}%p)", 0.0, c))

    # 5) 시장 스트레스 — VIX
    if "VIXCLS" in series:
        cur = _last(series["VIXCLS"])
        ind["VIX"] = cur
        if cur >= 30:
            sc, c = -1.0, "공포지수가 30 이상으로 시장이 매우 불안합니다."
        elif cur >= 20:
            sc, c = -0.5, "공포지수가 20 이상으로 변동성이 커진 상태입니다."
        elif cur < 13:
            sc, c = 0.0, "공포지수가 매우 낮습니다. 안정적이지만 시장이 낙관에 치우쳤을 수도 있습니다."
        else:
            sc, c = 0.5, "공포지수가 안정 구간(13~20)입니다."
        stress_scores.append(sc)
        sig.append(Signal("VIX 공포지수", f"{cur:.1f}", sc, c))

    # 6) 신용 스프레드
    if "BAMLH0A0HYM2" in series:
        s = series["BAMLH0A0HYM2"].dropna()
        cur, ago = _last(s), _value_ago(s, 91)
        ind["HY_SPREAD"] = cur
        widen = (cur - ago) if ago is not None else 0.0
        if cur >= 5.0 or widen >= 1.0:
            sc, c = -1.0, f"신용 스프레드가 {cur:.2f}%p(3개월 {widen:+.2f}%p)로 기업 부도 위험을 걱정하는 상황입니다."
        elif cur >= 4.0 or widen >= 0.5:
            sc, c = -0.5, f"신용 스프레드가 {cur:.2f}%p(3개월 {widen:+.2f}%p)로 벌어지는 중입니다."
        else:
            sc, c = 0.5, f"신용 스프레드가 {cur:.2f}%p로 낮아 금융시장이 안정적입니다."
        stress_scores.append(sc)
        sig.append(Signal("하이일드 신용스프레드", f"{cur:.2f}%p", sc, c))

    # 7) 실질금리 (금에 영향)
    if "DFII10" in series:
        s = series["DFII10"].dropna()
        cur, ago = _last(s), _value_ago(s, 91)
        ind["REAL_10Y"] = cur
        if ago is not None:
            chg = cur - ago
            c = (
                f"실질금리가 3개월간 {chg:+.2f}%p 움직였습니다. "
                + ("내려가는 중이라 금에 우호적입니다." if chg < -0.15 else
                   "올라가는 중이라 금에 부담입니다." if chg > 0.15 else "큰 변화는 없습니다.")
            )
            sig.append(Signal("미국 10년 실질금리", f"{cur:.2f}%", 0.0, c))

    # 8) 환율
    krw_weak = None
    if "DEXKOUS" in series:
        s = series["DEXKOUS"].dropna()
        cur = _last(s)
        window = s[s.index >= s.index[-1] - pd.Timedelta(days=365 * 3)]
        pct = float((window < cur).mean() * 100)
        ind["USDKRW"], ind["USDKRW_PCTL_3Y"] = cur, pct
        krw_weak = pct >= 80
        if krw_weak:
            level = "최근 3년 중 가장 높은 수준" if pct >= 99 else f"최근 3년 중 상위 {100 - pct:.0f}% 수준"
            c = f"원·달러 {cur:,.0f}원으로 {level}(원화 약세)입니다. 환노출 달러 자산은 나중에 환율이 내려가면 손실이 생길 수 있어 국내·환헤지 자산을 조금 더 섞습니다."
        elif pct <= 20:
            c = f"원·달러 {cur:,.0f}원으로 최근 3년 중 낮은 편(원화 강세)입니다. 환노출 미국 자산을 사기 좋은 환경입니다."
        else:
            c = f"원·달러 {cur:,.0f}원으로 최근 3년 범위의 중간 수준입니다."
        sig.append(Signal("원·달러 환율", f"{cur:,.0f}원", 0.0, c))

    if "DGS10" in series:
        ind["DGS10"] = _last(series["DGS10"])
    if "IRLTLT01KRM156N" in series:
        ind["KR10Y"] = _last(series["IRLTLT01KRM156N"])

    growth = sum(growth_scores) / len(growth_scores) if growth_scores else 0.0
    stress = sum(stress_scores) / len(stress_scores) if stress_scores else 0.0

    # 국면 분류
    if stress <= -0.75:
        regime, desc = "위기·고변동", "금융시장 스트레스가 큽니다. 주식 비중을 줄이고 현금성·국채로 방어합니다."
    elif growth < 0 and inflation_high:
        regime, desc = "스태그플레이션 경계", "성장은 둔화되는데 물가가 높습니다. 장기채보다 단기채·금 위주로 방어합니다."
    elif growth < 0:
        regime, desc = "경기 둔화", "경기 둔화 신호가 있습니다. 주식은 조금 줄이고, 금리가 내려갈 때 유리한 장기채를 늘립니다."
    elif inflation_high:
        regime, desc = "확장 + 물가 부담", "경기는 괜찮지만 물가가 높습니다. 주식은 유지하고 장기채 대신 단기채·금을 활용합니다."
    else:
        regime, desc = "확장·안정", "경기와 금융시장이 안정적입니다. 기본 비중을 유지합니다."

    tilts = derive_tilts(regime, easing, krw_weak)
    return MacroView(sig, growth, inflation_high, stress, easing, krw_weak, regime, desc, tilts, ind, missing)


def derive_tilts(regime: str, easing: bool, krw_weak: bool | None) -> dict[str, float]:
    """장기 바구니 기본 비중을 조정할 값(%p). allocator가 사용합니다."""
    t = {"equity": 0.0, "long_bond": 0.0, "short_bond": 0.0, "gold": 0.0, "cash": 0.0, "kr_equity_shift": 0.0}
    if regime == "위기·고변동":
        t.update(equity=-10, cash=+7, short_bond=+3)
    elif regime == "스태그플레이션 경계":
        t.update(equity=-6, long_bond=-5, short_bond=+6, gold=+3, cash=+2)
    elif regime == "경기 둔화":
        t.update(equity=-5, long_bond=+4, cash=+1)
    elif regime == "확장 + 물가 부담":
        t.update(long_bond=-5, short_bond=+3, gold=+2)
    if easing and regime not in ("스태그플레이션 경계", "확장 + 물가 부담"):
        t["long_bond"] += 2
        t["cash"] -= 2
    if krw_weak:
        t["kr_equity_shift"] = 3  # 환노출 미국 주식 3%p → 국내 주식
    return t
