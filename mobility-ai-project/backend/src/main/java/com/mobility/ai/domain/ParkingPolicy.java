package com.mobility.ai.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/**
 * 주차 요금 정책. 테이블 {@code parking_policy}.
 * 활성({@code active=true}) 정책 1건을 요금 계산 엔진에 주입한다.
 *
 * <p>요금 구조:
 * <pre>
 *   무료 시간(free_minutes) 이내 → 0원
 *   기본 시간(base_minutes) 이내 → base_fee
 *   초과분은 unit_minutes 단위로 올림하여 unit_fee 를 가산
 *   1일(24시간) 단위로 daily_max_fee 상한 적용
 * </pre>
 */
@Entity
@Table(name = "parking_policy")
public class ParkingPolicy {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @Column(name = "name", nullable = false, length = 80)
    private String name;

    /** 무료 시간(분) */
    @Column(name = "free_minutes", nullable = false)
    private int freeMinutes = 30;

    /** 기본 시간(분) */
    @Column(name = "base_minutes", nullable = false)
    private int baseMinutes = 60;

    /** 기본 요금(원) */
    @Column(name = "base_fee", nullable = false)
    private long baseFee = 2000L;

    /** 추가 단위 시간(분) */
    @Column(name = "unit_minutes", nullable = false)
    private int unitMinutes = 10;

    /** 추가 단위 요금(원) */
    @Column(name = "unit_fee", nullable = false)
    private long unitFee = 500L;

    /** 1일 상한 요금(원) */
    @Column(name = "daily_max_fee", nullable = false)
    private long dailyMaxFee = 20000L;

    @Column(name = "active", nullable = false)
    private boolean active = true;

    protected ParkingPolicy() {
    }

    public ParkingPolicy(String name, int freeMinutes, int baseMinutes, long baseFee,
                         int unitMinutes, long unitFee, long dailyMaxFee, boolean active) {
        this.name = name;
        this.freeMinutes = freeMinutes;
        this.baseMinutes = baseMinutes;
        this.baseFee = baseFee;
        this.unitMinutes = unitMinutes;
        this.unitFee = unitFee;
        this.dailyMaxFee = dailyMaxFee;
        this.active = active;
    }

    public Long getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getFreeMinutes() {
        return freeMinutes;
    }

    public void setFreeMinutes(int freeMinutes) {
        this.freeMinutes = freeMinutes;
    }

    public int getBaseMinutes() {
        return baseMinutes;
    }

    public void setBaseMinutes(int baseMinutes) {
        this.baseMinutes = baseMinutes;
    }

    public long getBaseFee() {
        return baseFee;
    }

    public void setBaseFee(long baseFee) {
        this.baseFee = baseFee;
    }

    public int getUnitMinutes() {
        return unitMinutes;
    }

    public void setUnitMinutes(int unitMinutes) {
        this.unitMinutes = unitMinutes;
    }

    public long getUnitFee() {
        return unitFee;
    }

    public void setUnitFee(long unitFee) {
        this.unitFee = unitFee;
    }

    public long getDailyMaxFee() {
        return dailyMaxFee;
    }

    public void setDailyMaxFee(long dailyMaxFee) {
        this.dailyMaxFee = dailyMaxFee;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }
}
