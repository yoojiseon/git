"""합성 데이터로 로직을 검증하는 테스트 (네트워크 불필요)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from pension_agent import allocator as al
from pension_agent import data, macro, main
from pension_agent.report import COMMENTARY_MARKER

ROOT = main.ROOT
UNIVERSE = yaml.safe_load((ROOT / "config/etf_universe.yaml").read_text(encoding="utf-8"))["etfs"]
PROFILE = yaml.safe_load((ROOT / "config/profile.yaml").read_text(encoding="utf-8"))

# 자산군별 (연 기대수익, 연 변동성) — 테스트용 가정
PARAMS = {
    "risky": (0.08, 0.20),
    "gold": (0.04, 0.15),
    "safe": (0.03, 0.05),
    "cash": (0.03, 0.002),
}


def synth_prices(seed: int = 0, days: int = 1500) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-11-03", periods=days)
    market = rng.normal(0, 1, days)
    out = {}
    for e in UNIVERSE:
        kind = "cash" if e["code"] == al.CASH_CODE else "gold" if e["asset_class"] == "금" else e["irp"]
        mu, vol = PARAMS[kind]
        beta = 0.8 if kind == "risky" else 0.0
        noise = rng.normal(0, 1, days)
        z = beta * market + np.sqrt(1 - beta**2) * noise
        r = mu / 252 + vol / np.sqrt(252) * z
        out[e["code"]] = pd.Series(10000 * np.cumprod(1 + r), index=idx)
    return out


def synth_fred() -> dict[str, pd.Series]:
    d = pd.date_range("2018-01-01", "2026-11-01", freq="D")
    m = pd.date_range("2018-01-01", "2026-10-01", freq="MS")
    return {
        "T10Y3M": pd.Series(0.6, index=d),
        "UNRATE": pd.Series(4.2, index=m),
        "CPIAUCSL": pd.Series(300 * 1.0022 ** np.arange(len(m)), index=m),
        "FEDFUNDS": pd.Series(np.linspace(5.3, 3.9, len(m)), index=m),
        "VIXCLS": pd.Series(16.0, index=d),
        "BAMLH0A0HYM2": pd.Series(3.1, index=d),
        "DEXKOUS": pd.Series(np.linspace(1200, 1380, len(d)), index=d),
    }


# ---------------------------------------------------------------- 설정 검증
def test_universe_is_consistent():
    codes = [e["code"] for e in UNIVERSE]
    assert len(codes) == len(set(codes)), "종목코드 중복"
    core_sum = sum(e.get("core_weight", 0) for e in UNIVERSE if "core" in e["roles"])
    assert core_sum == 100
    for e in UNIVERSE:
        assert e["irp"] in ("risky", "safe")
        assert "선물" not in e["name"] or e["roles"] == ["proxy"], f"{e['name']}: 선물형은 IRP 매수 불가"
        if e.get("proxy"):
            assert e["proxy"] in codes
    assert PROFILE["split"]["long_term"] + PROFILE["split"]["short_term"] == pytest.approx(1.0)


def test_names_match_handles_rebrands():
    assert data.names_match("KIWOOM 국고채10년", "KOSEF 국고채10년")
    assert data.names_match("KODEX 미국나스닥100", "KODEX 미국나스닥100TR")
    assert not data.names_match("TIGER 미국S&P500", "TIGER 차이나CSI300")


def test_parse_naver_chart():
    xml = (
        '<?xml version="1.0" encoding="EUC-KR" ?><protocol>'
        '<chartdata symbol="360750" name="TIGER 미국S&amp;P500" count="2" timeframe="day">'
        '<item data="20261102|100|110|90|105|1000" />'
        '<item data="20261103|105|112|101|111|900" />'
        "</chartdata></protocol>"
    ).encode("euc-kr")
    h = data.parse_naver_chart(xml, "360750")
    assert h.name == "TIGER 미국S&P500"
    assert list(h.close) == [105.0, 111.0]


# ---------------------------------------------------------------- 거시
def test_macro_expansion_regime():
    view = macro.analyze(synth_fred())
    assert view.regime in ("확장·안정", "확장 + 물가 부담")
    assert view.krw_weak is True
    assert view.tilts["kr_equity_shift"] == 3


def test_macro_crisis_regime():
    f = synth_fred()
    f["VIXCLS"] = f["VIXCLS"] * 0 + 38
    f["BAMLH0A0HYM2"] = f["BAMLH0A0HYM2"] * 0 + 6.5
    view = macro.analyze(f)
    assert view.regime == "위기·고변동"
    assert view.tilts["equity"] < 0


def test_sahm_rule_triggers():
    m = pd.date_range("2024-01-01", periods=24, freq="MS")
    u = pd.Series([3.7] * 18 + [4.0, 4.2, 4.4, 4.5, 4.6, 4.7], index=m)
    assert macro.sahm_gap(u) >= 0.5


# ---------------------------------------------------------------- 비중
def test_core_weights_sum_to_one_and_tilt():
    base, _ = al.core_weights(UNIVERSE, None)
    assert sum(base.values()) == pytest.approx(1.0)
    f = synth_fred()
    f["VIXCLS"] = f["VIXCLS"] * 0 + 38
    f["BAMLH0A0HYM2"] = f["BAMLH0A0HYM2"] * 0 + 6.5
    tilted, notes = al.core_weights(UNIVERSE, macro.analyze(f))
    assert sum(tilted.values()) == pytest.approx(1.0)
    assert al.risky_share(tilted, UNIVERSE) < al.risky_share(base, UNIVERSE)
    assert notes


def test_tactical_goes_defensive_when_everything_falls():
    prices = synth_prices()
    for e in UNIVERSE:
        if e["irp"] == "risky":
            s = prices[e["code"]]
            prices[e["code"]] = pd.Series(10000 * np.exp(np.linspace(0.0, -1.0, len(s))), index=s.index)  # 꾸준한 하락
    res = al.tactical_weights(UNIVERSE, prices, None)
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert al.risky_share(res.weights, UNIVERSE) == pytest.approx(0.0)
    assert res.weights[al.CASH_CODE] > 0


def test_tactical_avoids_highly_correlated_picks():
    prices = synth_prices()
    base = prices["360750"]
    boom = base * np.linspace(1.0, 2.0, len(base))
    for c in ("360750", "379810", "381180"):
        prices[c] = boom * (1 + 0.0001 * np.random.default_rng(1).normal(size=len(base))).cumprod()
    res = al.tactical_weights(UNIVERSE, prices, None)
    picked = [p.code for p in res.picks if p.code in ("360750", "379810", "381180")]
    assert len(picked) == 1


def test_fit_risk_respects_mdd_and_irp_cap():
    prices = synth_prices()
    w = {"360750": 0.5, "379810": 0.3, "411060": 0.1, al.CASH_CODE: 0.1}
    rets = al.build_returns(prices, UNIVERSE, list(w), 1260)
    k, notes = al.fit_risk(w, UNIVERSE, rets, 0.10, 0.67)
    final = al.scale_risky(w, UNIVERSE, k)
    assert sum(final.values()) == pytest.approx(1.0)
    assert al.risky_share(final, UNIVERSE) <= 0.67 + 1e-9
    assert -al.portfolio_stats(final, rets)["mdd"] <= 0.10
    assert notes


# ---------------------------------------------------------------- 매수 계획
def test_buy_plan_stays_within_budget_and_cap():
    target = {"360750": 0.4, "379810": 0.15, "411060": 0.1, "114260": 0.15, al.CASH_CODE: 0.2}
    prices = {"360750": 22000, "379810": 21000, "411060": 25000, "114260": 60000, al.CASH_CODE: 55000}
    plan = al.buy_plan("me", target, prices, UNIVERSE, 400000, risky_cap=0.67)
    spent = sum(l.amount for l in plan.lines)
    assert spent + plan.leftover == pytest.approx(400000)
    assert plan.leftover >= 0
    assert plan.risky_after <= 0.67
    assert min(prices.values()) > plan.leftover or all(
        target[c] * plan.value_after <= sum(l.amount for l in plan.lines if l.code == c) for c in target
        if prices[c] <= plan.leftover
    )


def test_buy_plan_fills_underweight_first():
    target = {"360750": 0.5, al.CASH_CODE: 0.5}
    prices = {"360750": 20000, al.CASH_CODE: 50000}
    plan = al.buy_plan("spouse", target, prices, UNIVERSE, 200000, holdings={al.CASH_CODE: 4})
    bought = {l.code: l.shares for l in plan.lines}
    assert bought.get("360750", 0) >= 9
    assert al.CASH_CODE not in bought


# ---------------------------------------------------------------- 전체 실행
def test_end_to_end_with_synthetic_data(tmp_path, monkeypatch):
    prices = synth_prices()
    name = {e["code"]: e["name"] for e in UNIVERSE}

    def fake_history(code, count=4000):
        return data.PriceHistory(code=code, name=name[code], close=prices[code])

    monkeypatch.setattr(main, "naver_history", fake_history)
    monkeypatch.setattr(main, "fred_many", lambda ids: (synth_fred(), []))
    path = main.run(month="2099-01", with_news=False, out_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "이번 달 할 일" in text and "본인 IRP" in text and "남편 IRP" in text
    assert COMMENTARY_MARKER in text
    assert (tmp_path / "2099-01.json").exists()


def test_name_mismatch_is_excluded(tmp_path, monkeypatch):
    prices = synth_prices()
    name = {e["code"]: e["name"] for e in UNIVERSE}
    name["453870"] = "TIGER 차이나CSI300"  # 잘못된 코드 상황

    monkeypatch.setattr(
        main, "naver_history", lambda code, count=4000: data.PriceHistory(code, name[code], prices[code])
    )
    monkeypatch.setattr(main, "fred_many", lambda ids: (synth_fred(), []))
    path = main.run(month="2099-02", with_news=False, out_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "453870의 실제 종목명" in text
    assert "| TIGER 인도니프티50 |" not in text


def test_buy_plan_respects_max_lines():
    target = {"360750": 0.3, "379810": 0.1, "411060": 0.1, "114260": 0.2, al.CASH_CODE: 0.3}
    prices = {"360750": 22000, "379810": 21000, "411060": 25000, "114260": 60000, al.CASH_CODE: 55000}
    plan = al.buy_plan("spouse", target, prices, UNIVERSE, 200000, max_lines=3)
    assert 1 <= len(plan.lines) <= 3


def test_ledger_rotates_purchases(tmp_path, monkeypatch):
    prices = synth_prices()
    name = {e["code"]: e["name"] for e in UNIVERSE}
    monkeypatch.setattr(
        main, "naver_history", lambda code, count=4000: data.PriceHistory(code, name[code], prices[code])
    )
    monkeypatch.setattr(main, "fred_many", lambda ids: (synth_fred(), []))
    hold = tmp_path / "holdings.yaml"
    bought = []
    for m in ("2099-01", "2099-02", "2099-03"):
        main.run(month=m, with_news=False, out_dir=tmp_path, assume_last_plan=True, holdings_file=hold)
        snap = __import__("json").loads((tmp_path / f"{m}.json").read_text(encoding="utf-8"))
        bought.append({b["code"] for b in snap["plans"][1]["buys"]})
    assert len(set().union(*bought)) > len(bought[0])  # 종목이 돌아가며 채워짐
    h = yaml.safe_load(hold.read_text(encoding="utf-8"))
    assert h["applied_reports"] == ["2099-01", "2099-02"]
