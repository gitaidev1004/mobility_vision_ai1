package com.mobility.ai.client;

import java.util.List;

/**
 * AI 서비스 {@code POST /api/v1/lpr/recognize} 응답을 백엔드 내부 모델로 옮긴 값 객체.
 *
 * <p>AI 서비스의 JSON 은 스네이크 케이스(`plate_number`, `valid_format` …)이므로,
 * Jackson 애너테이션 없이 {@link AiInferenceClient} 가 {@code Map} 으로 받아
 * 키를 직접 읽어 이 record 를 조립한다. 덕분에 Jackson 버전/패키지 변화와
 * 무관하게 동작하고, 계약서의 "커스텀 ObjectMapper 금지" 규칙도 지킨다.
 *
 * @param plateNumber 인식된 번호판(미검출이면 null)
 * @param confidence  신뢰도 0.0 ~ 1.0
 * @param status      SUCCESS|LOW_CONFIDENCE|NO_PLATE|FORMAT_INVALID
 * @param validFormat 한국 번호판 형식 적합 여부
 * @param detector    사용한 검출기 이름(폴백이면 `-fallback` 접미사)
 * @param recognizer  사용한 OCR 이름
 * @param detectMs    검출 소요시간(ms)
 * @param ocrMs       문자 인식 소요시간(ms)
 * @param totalMs     총 소요시간(ms)
 * @param qualityScore 품질 점수
 * @param readable    판독 가능 여부
 * @param detections  검출 박스 목록
 * @param available   AI 서비스 호출이 성공했는지 여부(폴백 판별용)
 */
public record AiRecognitionResult(
        String plateNumber,
        Double confidence,
        String status,
        Boolean validFormat,
        String detector,
        String recognizer,
        Double detectMs,
        Double ocrMs,
        Double totalMs,
        Double qualityScore,
        Boolean readable,
        List<AiDetection> detections,
        boolean available
) {
    /**
     * 검출 박스 1개.
     *
     * @param bbox  [x1, y1, x2, y2]
     * @param score 검출 신뢰도
     * @param label 라벨(보통 "plate")
     */
    public record AiDetection(
            List<Integer> bbox,
            Double score,
            String label
    ) {
    }

    /**
     * AI 서비스에 닿지 못했을 때 쓰는 폴백 결과.
     * 계약서 요구대로 상태는 {@code NO_PLATE} 이며 {@code available=false} 다.
     *
     * @return 사용 불가 상태를 나타내는 결과
     */
    public static AiRecognitionResult unavailable() {
        return new AiRecognitionResult(
                null, 0.0, "NO_PLATE", false,
                "unavailable", "unavailable",
                0.0, 0.0, 0.0,
                0.0, false,
                List.of(),
                false
        );
    }
}
