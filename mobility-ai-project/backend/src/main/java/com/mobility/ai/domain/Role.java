package com.mobility.ai.domain;

/**
 * 사용자 권한. 계약서 2장의 {@code app_user.role} 값과 1:1 대응한다.
 * SecurityConfig 에서 {@code ROLE_} 접두사를 붙여 권한으로 매핑한다.
 */
public enum Role {
    /** 전체 관리 */
    ADMIN,
    /** 운영(등록/수정 가능) */
    OPERATOR,
    /** 조회 전용 */
    VIEWER
}
