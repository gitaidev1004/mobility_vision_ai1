package com.mobility.ai.domain;

/**
 * 번호판 인식 결과 상태. 계약서의 {@code plate_recognition.status} 와 동일하며
 * AI 서비스 응답의 {@code status} 필드와도 값이 같다.
 */
public enum RecognitionStatus {
    /** 정상 인식(형식 유효 + 신뢰도 충족) */
    SUCCESS,
    /** 글자는 읽었으나 신뢰도가 임계값 미만 */
    LOW_CONFIDENCE,
    /** 번호판 미검출 (AI 서비스 장애 시에도 이 값으로 기록) */
    NO_PLATE,
    /** 읽었으나 한국 번호판 형식이 아님 */
    FORMAT_INVALID;

    /**
     * AI 응답 문자열을 안전하게 변환한다. 알 수 없는 값은 NO_PLATE 로 본다.
     *
     * @param raw AI 응답의 status 문자열
     * @return 매칭되는 상태, 없으면 NO_PLATE
     */
    public static RecognitionStatus from(String raw) {
        if (raw == null || raw.isBlank()) {
            return NO_PLATE;
        }
        for (RecognitionStatus status : values()) {
            if (status.name().equalsIgnoreCase(raw.trim())) {
                return status;
            }
        }
        return NO_PLATE;
    }
}
