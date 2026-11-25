package com.mobility.ai.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.time.LocalDateTime;

/**
 * 입차~출차 1회를 나타내는 주차 세션. 테이블 {@code parking_session}.
 * {@code exitAt == null} 이면 아직 주차 중(현재 주차 대수 집계 기준)이다.
 */
@Entity
@Table(name = "parking_session")
public class ParkingSession {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "vehicle_id")
    private Vehicle vehicle;

    @Column(name = "plate_number", length = 20)
    private String plateNumber;

    /** 입차 이벤트 id (FK 제약 없이 참조값만 보관) */
    @Column(name = "entry_event_id")
    private Long entryEventId;

    /** 출차 이벤트 id */
    @Column(name = "exit_event_id")
    private Long exitEventId;

    @Column(name = "entry_at", nullable = false)
    private LocalDateTime entryAt;

    @Column(name = "exit_at")
    private LocalDateTime exitAt;

    /** 주차 시간(분). 출차 시 계산되어 채워진다. */
    @Column(name = "parked_minutes")
    private Integer parkedMinutes;

    /** 최종 청구 금액(할인 반영 후) */
    @Column(name = "fee_amount")
    private Long feeAmount;

    /** 할인 금액 */
    @Column(name = "discount_amount", nullable = false)
    private long discountAmount = 0L;

    @Column(name = "paid", nullable = false)
    private boolean paid;

    /** CARD|CASH|PASS|MOBILE 등 */
    @Column(name = "payment_method", length = 20)
    private String paymentMethod;

    protected ParkingSession() {
    }

    public ParkingSession(String plateNumber, LocalDateTime entryAt) {
        this.plateNumber = plateNumber;
        this.entryAt = entryAt;
    }

    /**
     * 현재 주차 중인지 여부.
     *
     * @return 출차 시각이 없으면 true
     */
    public boolean isActive() {
        return this.exitAt == null;
    }

    public Long getId() {
        return id;
    }

    public Vehicle getVehicle() {
        return vehicle;
    }

    public void setVehicle(Vehicle vehicle) {
        this.vehicle = vehicle;
    }

    public String getPlateNumber() {
        return plateNumber;
    }

    public void setPlateNumber(String plateNumber) {
        this.plateNumber = plateNumber;
    }

    public Long getEntryEventId() {
        return entryEventId;
    }

    public void setEntryEventId(Long entryEventId) {
        this.entryEventId = entryEventId;
    }

    public Long getExitEventId() {
        return exitEventId;
    }

    public void setExitEventId(Long exitEventId) {
        this.exitEventId = exitEventId;
    }

    public LocalDateTime getEntryAt() {
        return entryAt;
    }

    public void setEntryAt(LocalDateTime entryAt) {
        this.entryAt = entryAt;
    }

    public LocalDateTime getExitAt() {
        return exitAt;
    }

    public void setExitAt(LocalDateTime exitAt) {
        this.exitAt = exitAt;
    }

    public Integer getParkedMinutes() {
        return parkedMinutes;
    }

    public void setParkedMinutes(Integer parkedMinutes) {
        this.parkedMinutes = parkedMinutes;
    }

    public Long getFeeAmount() {
        return feeAmount;
    }

    public void setFeeAmount(Long feeAmount) {
        this.feeAmount = feeAmount;
    }

    public long getDiscountAmount() {
        return discountAmount;
    }

    public void setDiscountAmount(long discountAmount) {
        this.discountAmount = discountAmount;
    }

    public boolean isPaid() {
        return paid;
    }

    public void setPaid(boolean paid) {
        this.paid = paid;
    }

    public String getPaymentMethod() {
        return paymentMethod;
    }

    public void setPaymentMethod(String paymentMethod) {
        this.paymentMethod = paymentMethod;
    }
}
