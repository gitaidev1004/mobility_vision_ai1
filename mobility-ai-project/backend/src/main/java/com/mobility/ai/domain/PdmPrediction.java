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
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

/**
 * 예지보전 예측 결과. 테이블 {@code pdm_prediction}.
 *
 * <p>{@code detail} 컬럼은 PostgreSQL JSONB 이며, 부품별 점수와 권고 목록을
 * <b>JSON 문자열 그대로</b> 보관한다. 필드 타입을 POJO 가 아닌 {@code String} 으로
 * 두고 {@code @JdbcTypeCode(SqlTypes.JSON)} 만 붙이면, Hibernate 는 jsonb 컬럼에
 * 문자열을 그대로 바인딩/추출한다. 객체 매핑을 ORM 에 맡기지 않으므로
 * 커스텀 {@code ObjectMapper} 빈이 필요 없다(계약서 0장 규칙).
 *
 * <p>JSON ↔ DTO 변환은 {@code com.mobility.ai.common.JsonCodec} 한 곳에서만 한다.
 */
@Entity
@Table(name = "pdm_prediction")
public class PdmPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "vehicle_id", nullable = false)
    private Vehicle vehicle;

    @Column(name = "predicted_at", nullable = false)
    private LocalDateTime predictedAt;

    /** 종합 건강 점수(0 ~ 100) */
    @Column(name = "health_score")
    private Double healthScore;

    /** 이상치 점수(0.0 ~ 1.0) */
    @Column(name = "anomaly_score")
    private Double anomalyScore;

    /** 고장 확률(0.0 ~ 1.0) */
    @Column(name = "failure_risk")
    private Double failureRisk;

    @Enumerated(EnumType.STRING)
    @Column(name = "risk_level", nullable = false, length = 10)
    private RiskLevel riskLevel;

    /** 잔존 수명(일) */
    @Column(name = "rul_days")
    private Integer rulDays;

    /** 가장 위험한 부품(engine|brake|battery|tire|oil ...) */
    @Column(name = "top_component", length = 50)
    private String topComponent;

    /** 부품/권고 상세 JSON 원문 */
    @Column(name = "detail", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    private String detail;

    @Column(name = "model_version", length = 50)
    private String modelVersion;

    protected PdmPrediction() {
    }

    public PdmPrediction(Vehicle vehicle, RiskLevel riskLevel) {
        this.vehicle = vehicle;
        this.riskLevel = riskLevel;
    }

    @PrePersist
    void onCreate() {
        if (this.predictedAt == null) {
            this.predictedAt = LocalDateTime.now();
        }
        if (this.riskLevel == null) {
            this.riskLevel = RiskLevel.LOW;
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

    public LocalDateTime getPredictedAt() {
        return predictedAt;
    }

    public void setPredictedAt(LocalDateTime predictedAt) {
        this.predictedAt = predictedAt;
    }

    public Double getHealthScore() {
        return healthScore;
    }

    public void setHealthScore(Double healthScore) {
        this.healthScore = healthScore;
    }

    public Double getAnomalyScore() {
        return anomalyScore;
    }

    public void setAnomalyScore(Double anomalyScore) {
        this.anomalyScore = anomalyScore;
    }

    public Double getFailureRisk() {
        return failureRisk;
    }

    public void setFailureRisk(Double failureRisk) {
        this.failureRisk = failureRisk;
    }

    public RiskLevel getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(RiskLevel riskLevel) {
        this.riskLevel = riskLevel;
    }

    public Integer getRulDays() {
        return rulDays;
    }

    public void setRulDays(Integer rulDays) {
        this.rulDays = rulDays;
    }

    public String getTopComponent() {
        return topComponent;
    }

    public void setTopComponent(String topComponent) {
        this.topComponent = topComponent;
    }

    public String getDetail() {
        return detail;
    }

    public void setDetail(String detail) {
        this.detail = detail;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }
}
