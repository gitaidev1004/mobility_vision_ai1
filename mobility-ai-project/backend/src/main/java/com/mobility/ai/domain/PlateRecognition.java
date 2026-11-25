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
 * AI 번호판 인식 1건의 원본 기록. 테이블 {@code plate_recognition}.
 * 모델/지연시간/품질을 함께 저장해 재현과 성능 분석에 쓴다(6장 성능 절).
 */
@Entity
@Table(name = "plate_recognition")
public class PlateRecognition {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    /** 인식된 번호판. 미검출이면 null */
    @Column(name = "plate_number", length = 20)
    private String plateNumber;

    /** 0.0 ~ 1.0 */
    @Column(name = "confidence")
    private Double confidence;

    /** 사용한 검출기 이름(예: yolo, yolo-fallback) */
    @Column(name = "detector_name", length = 50)
    private String detectorName;

    /** 사용한 OCR 이름(예: crnn, crnn-fallback) */
    @Column(name = "ocr_name", length = 50)
    private String ocrName;

    /** 업로드 이미지 저장 경로 */
    @Column(name = "image_path", length = 300)
    private String imagePath;

    @Column(name = "detect_ms")
    private Double detectMs;

    @Column(name = "ocr_ms")
    private Double ocrMs;

    @Column(name = "total_ms")
    private Double totalMs;

    /** 영상 품질 점수(0.0 ~ 1.0) */
    @Column(name = "quality_score")
    private Double qualityScore;

    /** 한국 번호판 형식 적합 여부 */
    @Column(name = "valid_format", nullable = false)
    private boolean validFormat;

    /** 매칭된 등록 차량. 미등록 차량이면 null */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "matched_vehicle_id")
    private Vehicle matchedVehicle;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 20)
    private RecognitionStatus status;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    protected PlateRecognition() {
    }

    public PlateRecognition(RecognitionStatus status) {
        this.status = status;
    }

    @PrePersist
    void onCreate() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
        }
        if (this.status == null) {
            this.status = RecognitionStatus.NO_PLATE;
        }
    }

    public Long getId() {
        return id;
    }

    public String getPlateNumber() {
        return plateNumber;
    }

    public void setPlateNumber(String plateNumber) {
        this.plateNumber = plateNumber;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public String getDetectorName() {
        return detectorName;
    }

    public void setDetectorName(String detectorName) {
        this.detectorName = detectorName;
    }

    public String getOcrName() {
        return ocrName;
    }

    public void setOcrName(String ocrName) {
        this.ocrName = ocrName;
    }

    public String getImagePath() {
        return imagePath;
    }

    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }

    public Double getDetectMs() {
        return detectMs;
    }

    public void setDetectMs(Double detectMs) {
        this.detectMs = detectMs;
    }

    public Double getOcrMs() {
        return ocrMs;
    }

    public void setOcrMs(Double ocrMs) {
        this.ocrMs = ocrMs;
    }

    public Double getTotalMs() {
        return totalMs;
    }

    public void setTotalMs(Double totalMs) {
        this.totalMs = totalMs;
    }

    public Double getQualityScore() {
        return qualityScore;
    }

    public void setQualityScore(Double qualityScore) {
        this.qualityScore = qualityScore;
    }

    public boolean isValidFormat() {
        return validFormat;
    }

    public void setValidFormat(boolean validFormat) {
        this.validFormat = validFormat;
    }

    public Vehicle getMatchedVehicle() {
        return matchedVehicle;
    }

    public void setMatchedVehicle(Vehicle matchedVehicle) {
        this.matchedVehicle = matchedVehicle;
    }

    public RecognitionStatus getStatus() {
        return status;
    }

    public void setStatus(RecognitionStatus status) {
        this.status = status;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
}
