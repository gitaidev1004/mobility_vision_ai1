package com.mobility.ai.dto;

import com.mobility.ai.domain.PdmPrediction;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 예지보전(PdM) DTO 모음. 계약서 4장 {@code /api/v1/pdm/**}.
 */
public final class PdmDtos {

    private PdmDtos() {
    }

    /**
     * 부품별 진단 1건. AI 응답 {@code components[]} 와 1:1.
     *
     * @param name   부품 이름(engine|brake|battery|tire|oil)
     * @param score  부품 점수(0~100)
     * @param status NORMAL|WARNING|CRITICAL
     * @param reason 판단 근거(한국어)
     */
    public record ComponentScore(
            String name,
            Double score,
            String status,
            String reason
    ) {
    }

    /**
     * 정비 권고 1건. AI 응답 {@code recommendations[]} 와 1:1.
     *
     * @param priority 우선순위(1이 가장 급함)
     * @param item     권고 항목
     * @param dueKm    권고 주행거리(km 이내)
     * @param dueDays  권고 기한(일 이내)
     */
    public record Recommendation(
            Integer priority,
            String item,
            Integer dueKm,
            Integer dueDays
    ) {
    }

    /**
     * 예측 결과 응답. 저장된 {@code pdm_prediction} 1건을 그대로 노출한다.
     *
     * @param id              PK
     * @param vehicleId       차량 PK
     * @param plateNumber     차량 번호
     * @param predictedAt     예측 시각
     * @param healthScore     종합 건강 점수
     * @param anomalyScore    이상치 점수
     * @param failureRisk     고장 확률
     * @param riskLevel       LOW|MEDIUM|HIGH|CRITICAL
     * @param rulDays         잔존 수명(일)
     * @param topComponent    최우선 점검 부품
     * @param components      부품별 진단
     * @param recommendations 정비 권고
     * @param modelVersion    모델 버전
     */
    public record PdmPredictionResponse(
            Long id,
            Long vehicleId,
            String plateNumber,
            LocalDateTime predictedAt,
            Double healthScore,
            Double anomalyScore,
            Double failureRisk,
            String riskLevel,
            Integer rulDays,
            String topComponent,
            List<ComponentScore> components,
            List<Recommendation> recommendations,
            String modelVersion
    ) {
        /**
         * 엔티티 + 복원된 상세를 합쳐 응답을 만든다.
         *
         * @param p               예측 엔티티
         * @param components      JSONB 에서 복원한 부품 목록(null 허용)
         * @param recommendations JSONB 에서 복원한 권고 목록(null 허용)
         * @return 응답 DTO
         */
        public static PdmPredictionResponse from(PdmPrediction p,
                                                 List<ComponentScore> components,
                                                 List<Recommendation> recommendations) {
            return new PdmPredictionResponse(
                    p.getId(),
                    p.getVehicle() == null ? null : p.getVehicle().getId(),
                    p.getVehicle() == null ? null : p.getVehicle().getPlateNumber(),
                    p.getPredictedAt(),
                    p.getHealthScore(),
                    p.getAnomalyScore(),
                    p.getFailureRisk(),
                    p.getRiskLevel() == null ? null : p.getRiskLevel().name(),
                    p.getRulDays(),
                    p.getTopComponent(),
                    components == null ? List.of() : components,
                    recommendations == null ? List.of() : recommendations,
                    p.getModelVersion()
            );
        }
    }

    /**
     * 위험 차량 목록 1건({@code GET /api/v1/pdm/risky}).
     *
     * @param vehicleId    차량 PK
     * @param plateNumber  차량 번호
     * @param ownerName    소유자
     * @param riskLevel    위험 등급
     * @param healthScore  건강 점수
     * @param failureRisk  고장 확률
     * @param rulDays      잔존 수명(일)
     * @param topComponent 최우선 점검 부품
     * @param predictedAt  예측 시각
     */
    public record RiskyVehicleResponse(
            Long vehicleId,
            String plateNumber,
            String ownerName,
            String riskLevel,
            Double healthScore,
            Double failureRisk,
            Integer rulDays,
            String topComponent,
            LocalDateTime predictedAt
    ) {
    }

    /**
     * 상세 JSONB 를 역직렬화하기 위한 래퍼.
     * {@code {"components":[...],"recommendations":[...]}} 구조와 1:1 이다.
     *
     * @param components      부품별 진단
     * @param recommendations 정비 권고
     */
    public record PdmDetail(
            List<ComponentScore> components,
            List<Recommendation> recommendations
    ) {
    }
}
