package com.mobility.ai.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.LocalDateTime;

/**
 * 차단기 출입 이벤트. 테이블 {@code access_event}.
 * 통행 허용 여부({@code allowed})와 거부 사유({@code reject_reason})를 남겨
 * 운영 화면에서 바로 추적할 수 있게 한다.
 */
@Entity
@Table(name = "access_event")
public class AccessEvent {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    /** 매칭된 차량. 미등록이면 null */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "vehicle_id")
    private Vehicle vehicle;

    @Column(name = "plate_number", length = 20)
    private String plateNumber;

    @Enumerated(EnumType.STRING)
    @Column(name = "event_type", nullable = false, length = 10)
    private EventType eventType;

    /** 게이트 이름(정문/후문/지하 등) */
    @Column(name = "gate_name", length = 50)
    private String gateName;

    @Column(name = "occurred_at", nullable = false)
    private LocalDateTime occurredAt;

    /** 이 이벤트를 유발한 인식 기록 */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "recognition_id")
    private PlateRecognition recognition;

    @Column(name = "confidence")
    private Double confidence;

    @Column(name = "allowed", nullable = false)
    private boolean allowed;

    /** 거부 사유 코드(예: NOT_REGISTERED, LOW_CONFIDENCE, AI_SERVICE_UNAVAILABLE) */
    @Column(name = "reject_reason", length = 100)
    private String rejectReason;

    protected AccessEvent() {
    }

    public AccessEvent(EventType eventType, String plateNumber, String gateName) {
        this.eventType = eventType;
        this.plateNumber = plateNumber;
        this.gateName = gateName;
    }

    @PrePersist
    void onCreate() {
        if (this.occurredAt == null) {
            this.occurredAt = LocalDateTime.now();
        }
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

    public EventType getEventType() {
        return eventType;
    }

    public void setEventType(EventType eventType) {
        this.eventType = eventType;
    }

    public String getGateName() {
        return gateName;
    }

    public void setGateName(String gateName) {
        this.gateName = gateName;
    }

    public LocalDateTime getOccurredAt() {
        return occurredAt;
    }

    public void setOccurredAt(LocalDateTime occurredAt) {
        this.occurredAt = occurredAt;
    }

    public PlateRecognition getRecognition() {
        return recognition;
    }

    public void setRecognition(PlateRecognition recognition) {
        this.recognition = recognition;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public boolean isAllowed() {
        return allowed;
    }

    public void setAllowed(boolean allowed) {
        this.allowed = allowed;
    }

    public String getRejectReason() {
        return rejectReason;
    }

    public void setRejectReason(String rejectReason) {
        this.rejectReason = rejectReason;
    }
}
