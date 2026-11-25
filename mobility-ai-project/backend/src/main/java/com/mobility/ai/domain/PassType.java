package com.mobility.ai.domain;

/**
 * 주차 이용권 종류. 계약서의 {@code vehicle.pass_type} 과 대응하며
 * 요금 계산 엔진({@code ParkingFeeCalculator})의 분기 기준이 된다.
 *
 * <ul>
 *   <li>NONE    : 일반 방문 차량, 정상 요금</li>
 *   <li>VISITOR : 방문권, 계산된 요금에서 비율 할인</li>
 *   <li>REGULAR : 월 정기권, 유효기간 내라면 무료</li>
 *   <li>STAFF   : 임직원, 유효기간 내라면 무료</li>
 * </ul>
 */
public enum PassType {
    NONE,
    VISITOR,
    REGULAR,
    STAFF;

    /**
     * 유효기간 내일 때 전액 면제되는 이용권인지 여부.
     *
     * @return REGULAR 또는 STAFF 이면 true
     */
    public boolean isFreePass() {
        return this == REGULAR || this == STAFF;
    }

    /**
     * 문자열을 안전하게 변환한다. 알 수 없는 값은 NONE 으로 본다.
     *
     * @param raw 입력 문자열
     * @return 매칭되는 이용권, 없으면 NONE
     */
    public static PassType from(String raw) {
        if (raw == null || raw.isBlank()) {
            return NONE;
        }
        for (PassType type : values()) {
            if (type.name().equalsIgnoreCase(raw.trim())) {
                return type;
            }
        }
        return NONE;
    }
}
