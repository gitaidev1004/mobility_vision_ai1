# -*- coding: utf-8 -*-
"""전기차 배터리 SOH 추정 — 용량 감소 추세와 내부저항 증가 추세.

SOH(State of Health) 는 '현재 사용 가능 용량 / 신품 용량' 이다. 차량은
완전 충·방전을 거의 하지 않으므로 전체 용량을 직접 재기 어렵다. 그래서
실무에서는 두 가지를 쓴다.

  1) 부분 구간 쿨롱 카운팅: SOC 가 충분히 변한 충전 구간에서
     적산 전류(Ah) / SOC 변화량 으로 용량을 추정한다.
  2) 내부저항(DCIR) 추세: 부하 변화 시 ΔV/ΔI 로 저항을 구한다.
     저항 증가는 용량 감소보다 먼저 나타나는 선행 지표다.

이 파일은 numpy 만으로 동작한다.
실행:  python 09_parts/ev_battery_soh.py
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(31415)

NOMINAL_CAPACITY_AH = 180.0        # 신품 공칭 용량
NOMINAL_DCIR_MOHM = 48.0           # 신품 내부저항
EOL_SOH = 0.80                     # 보증·교체 기준(End of Life)


# ---------------------------------------------------- 1) 부분 쿨롱 카운팅
def capacity_from_charge_window(current_a: np.ndarray, dt_s: float,
                                soc_start: float, soc_end: float,
                                min_dsoc: float = 0.25) -> float | None:
    """부분 충전 구간에서 전체 용량(Ah)을 추정한다.

    적산 전하량 Q = Σ I·dt / 3600 [Ah] 이고, 그 구간이 SOC 를
    Δsoc 만큼 올렸다면 전체 용량은 Q / Δsoc 이다.
    Δsoc 가 작으면 SOC 추정 오차가 그대로 증폭되므로 거부한다.
    """
    dsoc = soc_end - soc_start
    if dsoc < min_dsoc:
        return None                          # 구간이 짧아 신뢰할 수 없다
    q_ah = float(np.sum(np.abs(current_a)) * dt_s / 3600.0)
    return q_ah / dsoc


def soh_from_capacity(capacity_ah: float,
                      nominal_ah: float = NOMINAL_CAPACITY_AH) -> float:
    """용량 기반 SOH. 1.0 을 넘지 않도록 자른다."""
    return float(min(capacity_ah / nominal_ah, 1.0))


# ------------------------------------------------------- 2) 내부저항(DCIR)
def dcir_from_pulse(v_before: float, v_after: float,
                    i_before: float, i_after: float) -> float | None:
    """부하 급변 구간에서 DCIR[mΩ] = |ΔV / ΔI| × 1000 을 구한다."""
    di = i_after - i_before
    if abs(di) < 5.0:                        # 전류 변화가 작으면 노이즈가 지배한다
        return None
    return abs((v_after - v_before) / di) * 1000.0


def soh_from_dcir(dcir_mohm: float, nominal_mohm: float = NOMINAL_DCIR_MOHM,
                  eol_ratio: float = 1.6) -> float:
    """저항 기반 SOH. 저항이 신품의 eol_ratio 배가 되면 0.8 로 본다.

    저항은 커질수록 나쁘므로 역방향 선형 사상을 쓴다.
    """
    r = dcir_mohm / nominal_mohm
    frac = (r - 1.0) / (eol_ratio - 1.0)     # 0=신품, 1=EOL
    return float(max(0.0, min(1.0, 1.0 - frac * (1.0 - EOL_SOH))))


# --------------------------------------------------------- 추세와 잔존 수명
def robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen 유사 중앙값 기울기. 이상치 한두 개에 흔들리지 않는다."""
    n = len(x)
    if n < 3:
        return 0.0
    slopes = []
    step = max(1, n // 60)                   # 전부 조합하면 O(n^2) 이므로 간격 샘플링
    for i in range(0, n - 1, step):
        for j in range(i + 1, n, step):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(slopes)) if slopes else 0.0


def months_to_eol(soh_now: float, slope_per_month: float,
                  eol: float = EOL_SOH) -> float | None:
    """현재 SOH 와 월간 감소율로 EOL 도달까지 남은 개월을 추정한다."""
    if slope_per_month >= -1e-6:
        return None                          # 감소 추세가 없으면 추정 불가
    if soh_now <= eol:
        return 0.0
    return float((soh_now - eol) / (-slope_per_month))


def fuse_soh(soh_cap: float | None, soh_dcir: float | None,
             w_cap: float = 0.7) -> float:
    """두 추정치를 결합한다. 용량 기반을 주, 저항 기반을 보조로 쓴다."""
    if soh_cap is None and soh_dcir is None:
        return float("nan")
    if soh_cap is None:
        return float(soh_dcir)
    if soh_dcir is None:
        return float(soh_cap)
    return float(w_cap * soh_cap + (1.0 - w_cap) * soh_dcir)


# --------------------------------------------------------------- 시뮬레이션
def simulate_pack_history(months: int = 36) -> list[dict]:
    """36개월 동안 월 1회 측정된 충전 구간 로그를 만든다."""
    logs = []
    for m in range(months):
        # 실제 용량은 캘린더 열화 + 사이클 열화로 완만히 감소한다.
        fade = 0.045 * (m / 12.0) ** 0.85            # 1년에 약 4.5%
        cap_true = NOMINAL_CAPACITY_AH * (1.0 - fade)
        # 저항은 용량보다 빠르게 증가한다(선행 지표).
        dcir_true = NOMINAL_DCIR_MOHM * (1.0 + 0.22 * (m / 12.0) ** 0.9)

        soc_start = float(RNG.uniform(0.15, 0.30))
        soc_end = float(RNG.uniform(0.80, 0.95))
        dsoc = soc_end - soc_start
        dt_s = 10.0                                   # 10초 샘플링
        n = 900
        # 충전 전류를 평탄하게 가정하고 목표 전하량에 맞춘다.
        q_target = cap_true * dsoc
        i_mean = q_target * 3600.0 / (n * dt_s)
        current = np.abs(RNG.normal(i_mean, i_mean * 0.04, n))

        # 가속 시 전류 급증 펄스에서 DCIR 을 측정한다.
        i0, i1 = 12.0, 190.0
        v0 = 372.0 - i0 * dcir_true / 1000.0
        v1 = 372.0 - i1 * dcir_true / 1000.0 + RNG.normal(0, 0.25)

        logs.append({"month": m, "soc_start": soc_start, "soc_end": soc_end,
                     "current": current, "dt_s": dt_s,
                     "v0": v0, "v1": v1, "i0": i0, "i1": i1,
                     "cap_true": cap_true, "dcir_true": dcir_true})
    return logs


def main() -> None:
    logs = simulate_pack_history(36)

    rows = []
    for g in logs:
        cap = capacity_from_charge_window(g["current"], g["dt_s"],
                                          g["soc_start"], g["soc_end"])
        dcir = dcir_from_pulse(g["v0"], g["v1"], g["i0"], g["i1"])
        s_cap = soh_from_capacity(cap) if cap else None
        s_dc = soh_from_dcir(dcir) if dcir else None
        rows.append({"month": g["month"], "cap": cap, "dcir": dcir,
                     "soh_cap": s_cap, "soh_dcir": s_dc,
                     "soh": fuse_soh(s_cap, s_dc),
                     "soh_true": g["cap_true"] / NOMINAL_CAPACITY_AH})

    print(f"신품 기준: 용량 {NOMINAL_CAPACITY_AH:.0f}Ah, "
          f"내부저항 {NOMINAL_DCIR_MOHM:.0f}mΩ, EOL SOH {EOL_SOH:.0%}\n")
    print(f"{'월':>4s}{'추정용량Ah':>11s}{'DCIR mΩ':>10s}"
          f"{'SOH(용량)':>10s}{'SOH(저항)':>10s}{'SOH(융합)':>10s}{'참값':>8s}")
    for r in rows[::6] + [rows[-1]]:
        print(f"{r['month']:4d}{r['cap']:11.1f}{r['dcir']:10.1f}"
              f"{r['soh_cap']:10.3f}{r['soh_dcir']:10.3f}"
              f"{r['soh']:10.3f}{r['soh_true']:8.3f}")

    # 추세 추정과 잔존 수명
    x = np.array([r["month"] for r in rows], dtype=float)
    y = np.array([r["soh"] for r in rows], dtype=float)
    slope = robust_slope(x, y)
    soh_now = float(np.mean(y[-3:]))         # 최근 3개월 평균으로 노이즈를 줄인다
    rem = months_to_eol(soh_now, slope)

    print(f"\n[추세 분석]")
    print(f"월간 SOH 감소율 : {slope * 100:+.4f}%p/월 "
          f"(연 {slope * 1200:+.2f}%p)")
    print(f"현재 SOH        : {soh_now:.3f}")
    print(f"EOL({EOL_SOH:.0%}) 도달까지: "
          + (f"약 {rem:.1f}개월 (약 {rem / 12.0:.1f}년)" if rem else "추정 불가"))

    # 추정 오차
    err = np.array([r["soh"] - r["soh_true"] for r in rows])
    print(f"\n[추정 오차] 평균 {err.mean() * 100:+.3f}%p / "
          f"절대평균 {np.abs(err).mean() * 100:.3f}%p / "
          f"최대 {np.abs(err).max() * 100:.3f}%p")

    print("\n[저항 지표의 선행성 — 36개월 변화율]")
    d_cap = rows[-1]["soh_cap"] - rows[0]["soh_cap"]
    d_dc = rows[-1]["dcir"] / rows[0]["dcir"] - 1.0
    print(f"  용량 기반 SOH : {d_cap * 100:+.2f}%p")
    print(f"  내부저항      : {d_dc * 100:+.2f}% 증가")
    print("  저항 변화율이 용량 변화율보다 커서 조기 경보에 유리하다.")


if __name__ == "__main__":
    main()
