package com.mobility.ai.dto;

import com.mobility.ai.domain.PlateRecognition;
import com.mobility.ai.dto.VehicleDtos.VehicleSummary;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 번호판 인식 관련 DTO 모음.
 * 계약서 4장 {@code POST /api/v1/recognitions}, {@code GET /api/v1/recognitions}.
 */
public final class RecognitionDtos {

    private RecognitionDtos() {
    }

    /**
     * 검출 박스. 계약서 {@code detections:[{bbox, score}]} 와 1:1.
     *
     * @param bbox  [x1, y1, x2, y2]
     * @param score 검출 신뢰도
     */
    public record DetectionBox(
            List<Integer> bbox,
            Double score
    ) {
    }

    /**
     * 단계별 소요시간(ms). 계약서 {@code elapsedMs:{detect,ocr,total}} 와 1:1.
     *
     * @param detect 검출 시간
     * @param ocr    문자 인식 시간
     * @param total  총 시간
     */
    public record ElapsedMs(
            Double detect,
            Double ocr,
            Double total
    ) {
    }

    /**
     * 영상 품질. 계약서 {@code quality:{score,readable}} 와 1:1.
     *
     * @param score    품질 점수
     * @param readable 판독 가능 여부
     */
    public record QualityInfo(
            Double score,
            Boolean readable
    ) {
    }

    /**
     * 인식 + 출입/주차 처리 결과. 계약서 {@code RecognitionResultResponse} 와 1:1 이다.
     *
     * @param recognitionId    인식 기록 PK
     * @param plateNumber      인식된 번호판(null 가능)
     * @param confidence       신뢰도
     * @param status           SUCCESS|LOW_CONFIDENCE|NO_PLATE|FORMAT_INVALID
     * @param validFormat      형식 적합 여부
     * @param matchedVehicle   매칭된 차량(없으면 null)
     * @param eventId          생성/재사용된 출입 이벤트 PK
     * @param eventType        ENTRY|EXIT
     * @param allowed          통행 허용 여부
     * @param rejectReason     거부 사유(허용이면 null)
     * @param parkingSessionId 관련 주차 세션 PK
     * @param feeAmount        출차 시 계산된 요금(입차면 null)
     * @param parkedMinutes    주차 시간(분, 입차면 null)
     * @param detections       검출 박스 목록
     * @param elapsedMs        단계별 소요시간
     * @param quality          품질 정보
     */
    public record RecognitionResultResponse(
            Long recognitionId,
            String plateNumber,
            Double confidence,
            String status,
            Boolean validFormat,
            VehicleSummary matchedVehicle,
            Long eventId,
            String eventType,
            Boolean allowed,
            String rejectReason,
            Long parkingSessionId,
            Long feeAmount,
            Integer parkedMinutes,
            List<DetectionBox> detections,
            ElapsedMs elapsedMs,
            QualityInfo quality
    ) {
    }

    /**
     * 인식 이력 1건(목록용).
     *
     * @param id           PK
     * @param plateNumber  번호판
     * @param confidence   신뢰도
     * @param status       상태
     * @param validFormat  형식 적합 여부
     * @param detectorName 검출기
     * @param ocrName      OCR 이름
     * @param imagePath    저장 이미지 경로
     * @param detectMs     검출 시간
     * @param ocrMs        OCR 시간
     * @param totalMs      총 시간
     * @param qualityScore 품질 점수
     * @param matchedVehicleId 매칭 차량 PK(없으면 null)
     * @param createdAt    기록 시각
     */
    public record RecognitionResponse(
            Long id,
            String plateNumber,
            Double confidence,
            String status,
            boolean validFormat,
            String detectorName,
            String ocrName,
            String imagePath,
            Double detectMs,
            Double ocrMs,
            Double totalMs,
            Double qualityScore,
            Long matchedVehicleId,
            LocalDateTime createdAt
    ) {
        /**
         * 엔티티 → 목록 DTO 변환.
         *
         * <p>{@code matchedVehicle} 은 LAZY 연관이지만 여기서는 식별자만 쓰므로
         * 프록시 초기화 없이 {@code getId()} 만 호출한다.
         *
         * @param r 인식 엔티티
         * @return 목록 DTO
         */
        public static RecognitionResponse from(PlateRecognition r) {
            return new RecognitionResponse(
                    r.getId(),
                    r.getPlateNumber(),
                    r.getConfidence(),
                    r.getStatus() == null ? null : r.getStatus().name(),
                    r.isValidFormat(),
                    r.getDetectorName(),
                    r.getOcrName(),
                    r.getImagePath(),
                    r.getDetectMs(),
                    r.getOcrMs(),
                    r.getTotalMs(),
                    r.getQualityScore(),
                    r.getMatchedVehicle() == null ? null : r.getMatchedVehicle().getId(),
                    r.getCreatedAt()
            );
        }
    }
}
