package com.mobility.ai.common;

/**
 * 업무 규칙 위반(중복 등록, 이미 결제된 세션 등)에 사용한다.
 * {@link GlobalExceptionHandler} 가 400 Bad Request 로 변환한다.
 */
public class BusinessException extends RuntimeException {

    /**
     * @param message 사용자에게 보여줄 메시지
     */
    public BusinessException(String message) {
        super(message);
    }
}
