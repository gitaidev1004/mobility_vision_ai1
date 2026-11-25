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
 * 차량 텔레메트리 1건. 테이블 {@code sensor_reading}.
 * 예지보전(PdM) 모델의 입력 시계열이며 컬럼 구성은 계약서 2장과 1:1 이다.
 */
@Entity
@Table(name = "sensor_reading")
public class SensorReading {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "vehicle_id", nullable = false)
    private Vehicle vehicle;

    @Column(name = "measured_at", nullable = false)
    private LocalDateTime measuredAt;

    @Column(name = "odometer_km")
    private Integer odometerKm;

    /** 냉각수 온도(℃) */
    @Column(name = "engine_temp_c")
    private Double engineTempC;

    @Column(name = "engine_rpm")
    private Integer engineRpm;

    @Column(name = "speed_kph")
    private Double speedKph;

    @Column(name = "battery_voltage")
    private Double batteryVoltage;

    /** 배터리 건강도(0.0 ~ 1.0) */
    @Column(name = "battery_soh")
    private Double batterySoh;

    @Column(name = "tire_pressure_fl")
    private Double tirePressureFl;

    @Column(name = "tire_pressure_fr")
    private Double tirePressureFr;

    @Column(name = "tire_pressure_rl")
    private Double tirePressureRl;

    @Column(name = "tire_pressure_rr")
    private Double tirePressureRr;

    /** 브레이크 패드 잔량(mm) */
    @Column(name = "brake_pad_mm")
    private Double brakePadMm;

    /** 엔진오일 품질 지수(0.0 ~ 1.0) */
    @Column(name = "oil_quality")
    private Double oilQuality;

    /** 진동 RMS */
    @Column(name = "vibration_rms")
    private Double vibrationRms;

    /** 연료 잔량 비율(0.0 ~ 1.0) */
    @Column(name = "fuel_level")
    private Double fuelLevel;

    protected SensorReading() {
    }

    public SensorReading(Vehicle vehicle, LocalDateTime measuredAt) {
        this.vehicle = vehicle;
        this.measuredAt = measuredAt;
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

    public LocalDateTime getMeasuredAt() {
        return measuredAt;
    }

    public void setMeasuredAt(LocalDateTime measuredAt) {
        this.measuredAt = measuredAt;
    }

    public Integer getOdometerKm() {
        return odometerKm;
    }

    public void setOdometerKm(Integer odometerKm) {
        this.odometerKm = odometerKm;
    }

    public Double getEngineTempC() {
        return engineTempC;
    }

    public void setEngineTempC(Double engineTempC) {
        this.engineTempC = engineTempC;
    }

    public Integer getEngineRpm() {
        return engineRpm;
    }

    public void setEngineRpm(Integer engineRpm) {
        this.engineRpm = engineRpm;
    }

    public Double getSpeedKph() {
        return speedKph;
    }

    public void setSpeedKph(Double speedKph) {
        this.speedKph = speedKph;
    }

    public Double getBatteryVoltage() {
        return batteryVoltage;
    }

    public void setBatteryVoltage(Double batteryVoltage) {
        this.batteryVoltage = batteryVoltage;
    }

    public Double getBatterySoh() {
        return batterySoh;
    }

    public void setBatterySoh(Double batterySoh) {
        this.batterySoh = batterySoh;
    }

    public Double getTirePressureFl() {
        return tirePressureFl;
    }

    public void setTirePressureFl(Double tirePressureFl) {
        this.tirePressureFl = tirePressureFl;
    }

    public Double getTirePressureFr() {
        return tirePressureFr;
    }

    public void setTirePressureFr(Double tirePressureFr) {
        this.tirePressureFr = tirePressureFr;
    }

    public Double getTirePressureRl() {
        return tirePressureRl;
    }

    public void setTirePressureRl(Double tirePressureRl) {
        this.tirePressureRl = tirePressureRl;
    }

    public Double getTirePressureRr() {
        return tirePressureRr;
    }

    public void setTirePressureRr(Double tirePressureRr) {
        this.tirePressureRr = tirePressureRr;
    }

    public Double getBrakePadMm() {
        return brakePadMm;
    }

    public void setBrakePadMm(Double brakePadMm) {
        this.brakePadMm = brakePadMm;
    }

    public Double getOilQuality() {
        return oilQuality;
    }

    public void setOilQuality(Double oilQuality) {
        this.oilQuality = oilQuality;
    }

    public Double getVibrationRms() {
        return vibrationRms;
    }

    public void setVibrationRms(Double vibrationRms) {
        this.vibrationRms = vibrationRms;
    }

    public Double getFuelLevel() {
        return fuelLevel;
    }

    public void setFuelLevel(Double fuelLevel) {
        this.fuelLevel = fuelLevel;
    }
}
