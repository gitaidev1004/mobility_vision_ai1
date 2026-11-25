package com.mobility.ai.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * 인증 관련 요청/응답 DTO 모음.
 * 계약서 4장: {@code POST /api/v1/auth/login}
 */
public final class AuthDtos {

    private AuthDtos() {
    }

    /**
     * 로그인 요청.
     *
     * @param username 로그인 아이디
     * @param password 평문 비밀번호
     */
    public record LoginRequest(
            @NotBlank(message = "아이디는 필수입니다.") String username,
            @NotBlank(message = "비밀번호는 필수입니다.") String password
    ) {
    }

    /**
     * 로그인 응답. 계약서 JSON 과 필드명이 1:1 이다.
     *
     * <pre>
     * {"accessToken","tokenType":"Bearer","expiresIn":3600,"username","role","displayName"}
     * </pre>
     *
     * @param accessToken JWT 접근 토큰
     * @param tokenType   항상 "Bearer"
     * @param expiresIn   만료까지 남은 초
     * @param username    로그인 아이디
     * @param role        권한 문자열(ADMIN|OPERATOR|VIEWER)
     * @param displayName 표시 이름
     */
    public record LoginResponse(
            String accessToken,
            String tokenType,
            long expiresIn,
            String username,
            String role,
            String displayName
    ) {
    }

    /**
     * 현재 로그인 사용자 정보({@code GET /api/v1/auth/me}).
     *
     * @param username    로그인 아이디
     * @param role        권한
     * @param displayName 표시 이름
     */
    public record MeResponse(
            String username,
            String role,
            String displayName
    ) {
    }
}
