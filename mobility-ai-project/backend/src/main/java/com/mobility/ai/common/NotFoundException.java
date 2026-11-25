package com.mobility.ai.common;

/**
 * 조회 대상이 없을 때 던진다. {@link GlobalExceptionHandler} 가 404 로 변환한다.
 */
public class NotFoundException extends RuntimeException {

    /**
     * @param message 사용자에게 보여줄 메시지
     */
    public NotFoundException(String message) {
        super(message);
    }

    /**
     * "{리소스}을(를) 찾을 수 없습니다: {id}" 형태의 메시지를 만든다.
     *
     * @param resource 리소스 이름(예: 차량)
     * @param id       식별자
     * @return 예외 인스턴스
     */
    public static NotFoundException of(String resource, Object id) {
        return new NotFoundException(resource + "을(를) 찾을 수 없습니다: " + id);
    }
}
