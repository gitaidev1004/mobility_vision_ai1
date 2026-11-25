package com.mobility.ai.domain;

/**
 * 예지보전 위험 등급. 계약서의 {@code pdm_prediction.risk_level} 과 동일하다.
 */
public enum RiskLevel {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL;

    /**
     * AI 서비스가 알 수 없는 문자열을 보내더라도 기동이 깨지지 않도록
     * 느슨하게 변환한다(기본값 LOW).
     *
     * @param raw AI 응답의 risk_level 문자열
     * @return 매칭되는 등급, 없으면 LOW
     */
    public static RiskLevel from(String raw) {
        if (raw == null || raw.isBlank()) {
            return LOW;
        }
        for (RiskLevel level : values()) {
            if (level.name().equalsIgnoreCase(raw.trim())) {
                return level;
            }
        }
        return LOW;
    }
}
