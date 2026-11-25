package com.mobility.ai.common;

import java.time.LocalDateTime;

/**
 * 공통 오류 응답 본문. 계약서 4장 말미의 스키마와 1:1 이다.
 *
 * <pre>
 * {"timestamp","status","error","message","path"}
 * </pre>
 *
 * @param timestamp 발생 시각
 * @param status    HTTP 상태 코드
 * @param error     HTTP 상태 사유구(예: Not Found)
 * @param message   사람이 읽을 메시지
 * @param path      요청 경로
 */
public record ApiError(
        LocalDateTime timestamp,
        int status,
        String error,
        String message,
        String path
) {
    /**
     * 현재 시각으로 오류 본문을 만든다.
     *
     * @param status  HTTP 상태 코드
     * @param error   상태 사유구
     * @param message 메시지
     * @param path    요청 경로
     * @return 오류 본문
     */
    public static ApiError of(int status, String error, String message, String path) {
        return new ApiError(LocalDateTime.now(), status, error, message, path);
    }
}
