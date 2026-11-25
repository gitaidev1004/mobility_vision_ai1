package com.mobility.ai.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.LocalDateTime;

/**
 * 등록 차량 엔티티. 테이블 {@code vehicle}.
 * {@code plate_number} 가 업무 키이며 번호판 인식 결과와 매칭되는 기준이다.
 */
@Entity
@Table(name = "vehicle")
public class Vehicle {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    /** 차량 번호(유니크). 예: 12가3456 */
    @Column(name = "plate_number", nullable = false, unique = true, length = 20)
    private String plateNumber;

    @Column(name = "owner_name", nullable = false, length = 50)
    private String ownerName;

    @Column(name = "phone", length = 30)
    private String phone;

    /** SEDAN|SUV|TRUCK|VAN|EV 등 자유 문자열 */
    @Column(name = "vehicle_type", length = 30)
    private String vehicleType;

    @Column(name = "model_name", length = 80)
    private String modelName;

    @Column(name = "model_year")
    private Integer modelYear;

    @Column(name = "vin", length = 40)
    private String vin;

    /** 주차 이용권 종류. 요금 계산 분기에 사용. */
    @Enumerated(EnumType.STRING)
    @Column(name = "pass_type", nullable = false, length = 20)
    private PassType passType = PassType.NONE;

    /** 이용권 만료 시각. null 이면 만료 없음으로 본다. */
    @Column(name = "pass_expires_at")
    private LocalDateTime passExpiresAt;

    @Column(name = "active", nullable = false)
    private boolean active = true;

    @Column(name = "registered_at", nullable = false)
    private LocalDateTime registeredAt;

    protected Vehicle() {
    }

    public Vehicle(String plateNumber, String ownerName) {
        this.plateNumber = plateNumber;
        this.ownerName = ownerName;
    }

    @PrePersist
    void onCreate() {
        if (this.registeredAt == null) {
            this.registeredAt = LocalDateTime.now();
        }
        if (this.passType == null) {
            this.passType = PassType.NONE;
        }
    }

    /**
     * 기준 시각에 무료 이용권(정기권/임직원)이 유효한지 판단한다.
     * 요금 계산 엔진이 이 결과를 그대로 신뢰한다.
     *
     * @param at 판정 기준 시각(보통 출차 시각)
     * @return 무료 이용권이 유효하면 true
     */
    public boolean hasValidFreePass(LocalDateTime at) {
        if (this.passType == null || !this.passType.isFreePass()) {
            return false;
        }
        return this.passExpiresAt == null || this.passExpiresAt.isAfter(at);
    }

    /**
     * 기준 시각에 방문권이 유효한지 판단한다(할인 적용 대상).
     *
     * @param at 판정 기준 시각
     * @return 방문권이 유효하면 true
     */
    public boolean hasValidVisitorPass(LocalDateTime at) {
        if (this.passType != PassType.VISITOR) {
            return false;
        }
        return this.passExpiresAt == null || this.passExpiresAt.isAfter(at);
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getPlateNumber() {
        return plateNumber;
    }

    public void setPlateNumber(String plateNumber) {
        this.plateNumber = plateNumber;
    }

    public String getOwnerName() {
        return ownerName;
    }

    public void setOwnerName(String ownerName) {
        this.ownerName = ownerName;
    }

    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }

    public String getVehicleType() {
        return vehicleType;
    }

    public void setVehicleType(String vehicleType) {
        this.vehicleType = vehicleType;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public Integer getModelYear() {
        return modelYear;
    }

    public void setModelYear(Integer modelYear) {
        this.modelYear = modelYear;
    }

    public String getVin() {
        return vin;
    }

    public void setVin(String vin) {
        this.vin = vin;
    }

    public PassType getPassType() {
        return passType;
    }

    public void setPassType(PassType passType) {
        this.passType = passType;
    }

    public LocalDateTime getPassExpiresAt() {
        return passExpiresAt;
    }

    public void setPassExpiresAt(LocalDateTime passExpiresAt) {
        this.passExpiresAt = passExpiresAt;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    public LocalDateTime getRegisteredAt() {
        return registeredAt;
    }

    public void setRegisteredAt(LocalDateTime registeredAt) {
        this.registeredAt = registeredAt;
    }
}
