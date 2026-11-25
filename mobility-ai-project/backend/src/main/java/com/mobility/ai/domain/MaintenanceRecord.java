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
import java.time.LocalDate;

/**
 * 정비 이력. 테이블 {@code maintenance_record}.
 * 예지보전 권고와 실제 정비 실행을 대조하는 데 사용한다.
 */
@Entity
@Table(name = "maintenance_record")
public class MaintenanceRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "vehicle_id", nullable = false)
    private Vehicle vehicle;

    /** 정비일(시각 없음) */
    @Column(name = "service_date", nullable = false)
    private LocalDate serviceDate;

    @Column(name = "odometer_km")
    private Integer odometerKm;

    /** 정비 항목(예: 엔진오일 교환) */
    @Column(name = "item", nullable = false, length = 80)
    private String item;

    @Column(name = "part_name", length = 80)
    private String partName;

    /** 비용(원) */
    @Column(name = "cost")
    private Long cost;

    @Column(name = "shop_name", length = 80)
    private String shopName;

    @Column(name = "memo", length = 500)
    private String memo;

    protected MaintenanceRecord() {
    }

    public MaintenanceRecord(Vehicle vehicle, LocalDate serviceDate, String item) {
        this.vehicle = vehicle;
        this.serviceDate = serviceDate;
        this.item = item;
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

    public LocalDate getServiceDate() {
        return serviceDate;
    }

    public void setServiceDate(LocalDate serviceDate) {
        this.serviceDate = serviceDate;
    }

    public Integer getOdometerKm() {
        return odometerKm;
    }

    public void setOdometerKm(Integer odometerKm) {
        this.odometerKm = odometerKm;
    }

    public String getItem() {
        return item;
    }

    public void setItem(String item) {
        this.item = item;
    }

    public String getPartName() {
        return partName;
    }

    public void setPartName(String partName) {
        this.partName = partName;
    }

    public Long getCost() {
        return cost;
    }

    public void setCost(Long cost) {
        this.cost = cost;
    }

    public String getShopName() {
        return shopName;
    }

    public void setShopName(String shopName) {
        this.shopName = shopName;
    }

    public String getMemo() {
        return memo;
    }

    public void setMemo(String memo) {
        this.memo = memo;
    }
}
