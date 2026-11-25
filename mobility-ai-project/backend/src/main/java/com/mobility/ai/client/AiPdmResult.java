package com.mobility.ai.client;

import com.mobility.ai.dto.PdmDtos.ComponentScore;
import com.mobility.ai.dto.PdmDtos.Recommendation;
import java.util.List;

/**
 * AI 서비스 {@code POST /api/v1/pdm/predict} 응답의 내부 표현.
 * 스네이크 케이스 키는 {@link AiInferenceClient} 가 직접 읽어 변환한다.
 *
 * @param vehicleId       차량 PK
 * @param healthScore     종합 건강 점수(0~100)
 * @param anomalyScore    이상치 점수(0.0~1.0)
 * @param failureRisk     고장 확률(0.0~1.0)
 * @param riskLevel       LOW|MEDIUM|HIGH|CRITICAL
 * @param rulDays         잔존 수명(일)
 * @param topComponent    최우선 점검 부품
 * @param components      부품별 진단
 * @param recommendations 정비 권고
 * @param modelVersion    모델 버전
 * @param available       AI 호출 성공 여부
 */
public record AiPdmResult(
        Long vehicleId,
        Double healthScore,
        Double anomalyScore,
        Double failureRisk,
        String riskLevel,
        Integer rulDays,
        String topComponent,
        List<ComponentScore> components,
        List<Recommendation> recommendations,
        String modelVersion,
        boolean available
) {
    /**
     * AI 서비스에 닿지 못했을 때의 보수적 폴백.
     * 점수를 모른다는 뜻으로 null 을 쓰지 않고, 중립값과 LOW 등급을 돌려주되
     * {@code modelVersion} 에 폴백임을 표시해 운영자가 구분할 수 있게 한다.
     *
     * @param vehicleId 대상 차량 PK
     * @return 폴백 결과
     */
    public static AiPdmResult unavailable(Long vehicleId) {
        return new AiPdmResult(
                vehicleId,
                null, null, null,
                "LOW",
                null, null,
                List.of(),
                List.of(),
                "unavailable",
                false
        );
    }
}
