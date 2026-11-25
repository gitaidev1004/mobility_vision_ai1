package com.mobility.ai.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.LocalDateTime;

/**
 * 모델 성능 측정 1건. 테이블 {@code model_benchmark}.
 * AI 서비스의 {@code POST /api/v1/models/benchmark} 결과를 그대로 적재한다.
 */
@Entity
@Table(name = "model_benchmark")
public class ModelBenchmark {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @Column(name = "model_name", nullable = false, length = 80)
    private String modelName;

    /** 측정 과제(예: detect+ocr) */
    @Column(name = "task", nullable = false, length = 40)
    private String task;

    /** 검출 재현율 */
    @Column(name = "detect_recall")
    private Double detectRecall;

    /** 전체 문자열 정확 일치율 */
    @Column(name = "exact_match_rate")
    private Double exactMatchRate;

    @Column(name = "latency_p50_ms")
    private Double latencyP50Ms;

    @Column(name = "latency_p95_ms")
    private Double latencyP95Ms;

    @Column(name = "memory_mb")
    private Double memoryMb;

    @Column(name = "model_size_mb")
    private Double modelSizeMb;

    /** 양자화 적용 여부 */
    @Column(name = "quantized", nullable = false)
    private boolean quantized;

    @Column(name = "measured_at", nullable = false)
    private LocalDateTime measuredAt;

    protected ModelBenchmark() {
    }

    public ModelBenchmark(String modelName, String task) {
        this.modelName = modelName;
        this.task = task;
    }

    @PrePersist
    void onCreate() {
        if (this.measuredAt == null) {
            this.measuredAt = LocalDateTime.now();
        }
    }

    public Long getId() {
        return id;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getTask() {
        return task;
    }

    public void setTask(String task) {
        this.task = task;
    }

    public Double getDetectRecall() {
        return detectRecall;
    }

    public void setDetectRecall(Double detectRecall) {
        this.detectRecall = detectRecall;
    }

    public Double getExactMatchRate() {
        return exactMatchRate;
    }

    public void setExactMatchRate(Double exactMatchRate) {
        this.exactMatchRate = exactMatchRate;
    }

    public Double getLatencyP50Ms() {
        return latencyP50Ms;
    }

    public void setLatencyP50Ms(Double latencyP50Ms) {
        this.latencyP50Ms = latencyP50Ms;
    }

    public Double getLatencyP95Ms() {
        return latencyP95Ms;
    }

    public void setLatencyP95Ms(Double latencyP95Ms) {
        this.latencyP95Ms = latencyP95Ms;
    }

    public Double getMemoryMb() {
        return memoryMb;
    }

    public void setMemoryMb(Double memoryMb) {
        this.memoryMb = memoryMb;
    }

    public Double getModelSizeMb() {
        return modelSizeMb;
    }

    public void setModelSizeMb(Double modelSizeMb) {
        this.modelSizeMb = modelSizeMb;
    }

    public boolean isQuantized() {
        return quantized;
    }

    public void setQuantized(boolean quantized) {
        this.quantized = quantized;
    }

    public LocalDateTime getMeasuredAt() {
        return measuredAt;
    }

    public void setMeasuredAt(LocalDateTime measuredAt) {
        this.measuredAt = measuredAt;
    }
}
