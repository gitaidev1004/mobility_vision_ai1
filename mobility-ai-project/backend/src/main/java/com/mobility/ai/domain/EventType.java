package com.mobility.ai.domain;

/**
 * 출입 이벤트 종류. 계약서의 {@code access_event.event_type} 과 동일하다.
 */
public enum EventType {
    /** 입차 */
    ENTRY,
    /** 출차 */
    EXIT
}
