package com.mobility.ai.service;

import com.mobility.ai.domain.ParkingPolicy;
import com.mobility.ai.domain.PassType;

/**
 * 주차 요금 계산 엔진 — <b>순수 함수</b>로 작성해 DB·스프링 컨텍스트 없이
 * 단위 테스트할 수 있게 한다(교재 6장 "요금 계산 로직 분리" 절에서 인용).
 *
 * <h2>계산 규칙</h2>
 * <ol>
 *   <li><b>무료 이용권</b>(REGULAR/STAFF, 유효기간 내) → 0원, 사유 {@code FREE_PASS}</li>
 *   <li><b>무료 시간 이내</b>: {@code minutes <= freeMinutes} → 0원
 *       (경계값: 무료시간과 "정확히 같으면" 무료)</li>
 *   <li><b>기본 시간 이내</b>: 무료시간 초과 ~ {@code baseMinutes} 이하 → {@code baseFee}</li>
 *   <li><b>기본 시간 초과</b>: 초과분을 {@code unitMinutes} 단위로 <b>올림</b>해
 *       {@code unitFee} 를 가산</li>
 *   <li><b>일일 상한</b>: 24시간(1,440분)을 1일로 보고, 시작된 날 수만큼
 *       {@code dailyMaxFee * 일수} 를 상한으로 적용</li>
 *   <li><b>방문권 할인</b>: VISITOR 는 위 결과에 {@link #VISITOR_DISCOUNT_RATE} 비율 할인</li>
 * </ol>
 *
 * <h2>무료 시간의 기준점</h2>
 * 무료 시간은 "면제 구간"이지 "요금 기준 시각의 이동"이 아니다. 즉 무료 30분을
 * 넘기면 전체 주차 시간(= 입차 시각 기준)에 대해 기본/추가 요금을 매긴다.
 * 교재 본문과 테스트가 이 해석을 공유한다.
 */
public final class ParkingFeeCalculator {

    /** 1일로 보는 분(24시간) */
    public static final int MINUTES_PER_DAY = 1440;

    /** 방문권 할인율(20%) */
    public static final double VISITOR_DISCOUNT_RATE = 0.20;

    private ParkingFeeCalculator() {
    }

    /**
     * 계산 결과.
     *
     * @param grossFee    할인 전 요금(원)
     * @param discount    할인 금액(원)
     * @param payableFee  최종 청구 금액(= grossFee - discount, 0 이상)
     * @param freeReason  전액 면제 사유(해당 없으면 null)
     */
    public record FeeResult(
            long grossFee,
            long discount,
            long payableFee,
            String freeReason
    ) {
    }

    /**
     * 요금을 계산한다.
     *
     * @param policy          적용 요금 정책(null 이면 전액 0원 처리)
     * @param parkedMinutes   주차 시간(분). 음수는 0으로 본다.
     * @param passType        차량 이용권(null 이면 NONE 취급)
     * @param freePassValid   정기권/임직원권이 유효한지 여부
     *                        ({@code Vehicle.hasValidFreePass(exitAt)} 결과)
     * @param visitorPassValid 방문권이 유효한지 여부
     *                        ({@code Vehicle.hasValidVisitorPass(exitAt)} 결과)
     * @return 계산 결과
     */
    public static FeeResult calculate(ParkingPolicy policy,
                                      long parkedMinutes,
                                      PassType passType,
                                      boolean freePassValid,
                                      boolean visitorPassValid) {
        long minutes = Math.max(0L, parkedMinutes);
        PassType type = passType == null ? PassType.NONE : passType;

        // 정책이 없으면 과금하지 않는다(초기 구축 단계 안전장치).
        if (policy == null) {
            return new FeeResult(0L, 0L, 0L, "NO_POLICY");
        }

        // 1) 무료 이용권(정기권/임직원) — 유효기간 내라면 전액 면제
        if (type.isFreePass() && freePassValid) {
            return new FeeResult(0L, 0L, 0L, "FREE_PASS");
        }

        // 2) 무료 시간 이내 (경계값: 정확히 같으면 무료)
        if (minutes <= policy.getFreeMinutes()) {
            return new FeeResult(0L, 0L, 0L, "FREE_MINUTES");
        }

        // 3) 기본 요금 + 4) 단위 요금
        long gross = computeGross(policy, minutes);

        // 5) 일일 상한
        gross = Math.min(gross, dailyCap(policy, minutes));

        // 6) 방문권 할인
        long discount = 0L;
        if (type == PassType.VISITOR && visitorPassValid) {
            discount = Math.round(gross * VISITOR_DISCOUNT_RATE);
        }

        long payable = Math.max(0L, gross - discount);
        return new FeeResult(gross, discount, payable, null);
    }

    /**
     * 상한 적용 전의 요금(기본 요금 + 단위 요금)을 구한다.
     *
     * @param policy  요금 정책
     * @param minutes 주차 시간(분, 무료 시간 초과가 확정된 값)
     * @return 할인/상한 적용 전 요금
     */
    private static long computeGross(ParkingPolicy policy, long minutes) {
        long fee = policy.getBaseFee();

        long overMinutes = minutes - policy.getBaseMinutes();
        if (overMinutes <= 0) {
            // 기본 시간 이내 → 기본 요금만
            return fee;
        }

        int unit = policy.getUnitMinutes();
        if (unit <= 0) {
            // 단위 시간이 0 이하인 잘못된 정책은 추가 과금을 생략한다.
            return fee;
        }

        // 초과분을 단위 시간으로 "올림" 처리 (예: 단위 10분, 초과 1분 → 1단위)
        long units = (overMinutes + unit - 1) / unit;
        return fee + units * policy.getUnitFee();
    }

    /**
     * 일일 상한 금액을 구한다. 24시간마다 상한이 1회분씩 늘어난다.
     *
     * <p>예) 상한 20,000원, 주차 30시간 → 2일분 = 40,000원
     *
     * @param policy  요금 정책
     * @param minutes 주차 시간(분)
     * @return 적용할 상한 금액
     */
    private static long dailyCap(ParkingPolicy policy, long minutes) {
        long days = (minutes + MINUTES_PER_DAY - 1) / MINUTES_PER_DAY;
        if (days < 1) {
            days = 1;
        }
        return policy.getDailyMaxFee() * days;
    }
}
