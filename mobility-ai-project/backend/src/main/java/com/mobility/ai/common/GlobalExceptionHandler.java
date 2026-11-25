package com.mobility.ai.common;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.ConstraintViolationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.validation.FieldError;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;
import org.springframework.web.servlet.NoHandlerFoundException;

/**
 * 전역 예외 처리기. 모든 오류를 계약서의 {@link ApiError} 포맷으로 통일한다.
 *
 * <p>매핑 규칙:
 * <ul>
 *   <li>{@link NotFoundException} → 404</li>
 *   <li>{@link BusinessException}, 검증 실패, 타입 불일치 → 400</li>
 *   <li>{@link AuthenticationException} → 401</li>
 *   <li>{@link AccessDeniedException} → 403</li>
 *   <li>그 외 → 500 (스택트레이스는 로그에만 남긴다)</li>
 * </ul>
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /**
     * 404 : 리소스 없음
     *
     * @param ex      예외
     * @param request 요청
     * @return 404 응답
     */
    @ExceptionHandler(NotFoundException.class)
    public ResponseEntity<ApiError> handleNotFound(NotFoundException ex, HttpServletRequest request) {
        return build(HttpStatus.NOT_FOUND, ex.getMessage(), request);
    }

    /**
     * 400 : 업무 규칙 위반
     *
     * @param ex      예외
     * @param request 요청
     * @return 400 응답
     */
    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<ApiError> handleBusiness(BusinessException ex, HttpServletRequest request) {
        return build(HttpStatus.BAD_REQUEST, ex.getMessage(), request);
    }

    /**
     * 400 : {@code @Valid} 바디 검증 실패. 첫 번째 필드 오류를 메시지로 쓴다.
     *
     * @param ex      예외
     * @param request 요청
     * @return 400 응답
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiError> handleValidation(MethodArgumentNotValidException ex,
                                                     HttpServletRequest request) {
        String message = "요청 값이 올바르지 않습니다.";
        if (!ex.getBindingResult().getFieldErrors().isEmpty()) {
            FieldError first = ex.getBindingResult().getFieldErrors().get(0);
            message = first.getField() + ": " + first.getDefaultMessage();
        }
        return build(HttpStatus.BAD_REQUEST, message, request);
    }

    /**
     * 400 : 파라미터 제약 위반
     *
     * @param ex      예외
     * @param request 요청
     * @return 400 응답
     */
    @ExceptionHandler(ConstraintViolationException.class)
    public ResponseEntity<ApiError> handleConstraint(ConstraintViolationException ex,
                                                     HttpServletRequest request) {
        return build(HttpStatus.BAD_REQUEST, ex.getMessage(), request);
    }

    /**
     * 400 : 필수 파라미터 누락 / 타입 불일치 / 본문 파싱 실패
     *
     * @param ex      예외
     * @param request 요청
     * @return 400 응답
     */
    @ExceptionHandler({
            MissingServletRequestParameterException.class,
            MethodArgumentTypeMismatchException.class,
            HttpMessageNotReadableException.class,
            IllegalArgumentException.class
    })
    public ResponseEntity<ApiError> handleBadRequest(Exception ex, HttpServletRequest request) {
        return build(HttpStatus.BAD_REQUEST, ex.getMessage(), request);
    }

    /**
     * 401 : 인증 실패(잘못된 자격 증명, 토큰 없음/만료)
     *
     * @param ex      예외
     * @param request 요청
     * @return 401 응답
     */
    @ExceptionHandler(AuthenticationException.class)
    public ResponseEntity<ApiError> handleAuthentication(AuthenticationException ex,
                                                         HttpServletRequest request) {
        return build(HttpStatus.UNAUTHORIZED, "인증에 실패했습니다: " + ex.getMessage(), request);
    }

    /**
     * 403 : 권한 부족
     *
     * @param ex      예외
     * @param request 요청
     * @return 403 응답
     */
    @ExceptionHandler(AccessDeniedException.class)
    public ResponseEntity<ApiError> handleAccessDenied(AccessDeniedException ex,
                                                       HttpServletRequest request) {
        return build(HttpStatus.FORBIDDEN, "접근 권한이 없습니다.", request);
    }

    /**
     * 404 : 매핑되지 않은 경로
     *
     * @param ex      예외
     * @param request 요청
     * @return 404 응답
     */
    @ExceptionHandler(NoHandlerFoundException.class)
    public ResponseEntity<ApiError> handleNoHandler(NoHandlerFoundException ex,
                                                    HttpServletRequest request) {
        return build(HttpStatus.NOT_FOUND, "존재하지 않는 경로입니다.", request);
    }

    /**
     * 405 : 허용되지 않은 HTTP 메서드
     *
     * @param ex      예외
     * @param request 요청
     * @return 405 응답
     */
    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    public ResponseEntity<ApiError> handleMethodNotSupported(HttpRequestMethodNotSupportedException ex,
                                                             HttpServletRequest request) {
        return build(HttpStatus.METHOD_NOT_ALLOWED, ex.getMessage(), request);
    }

    /**
     * 415 : 지원하지 않는 Content-Type
     *
     * @param ex      예외
     * @param request 요청
     * @return 415 응답
     */
    @ExceptionHandler(HttpMediaTypeNotSupportedException.class)
    public ResponseEntity<ApiError> handleMediaTypeNotSupported(HttpMediaTypeNotSupportedException ex,
                                                                 HttpServletRequest request) {
        return build(HttpStatus.UNSUPPORTED_MEDIA_TYPE, ex.getMessage(), request);
    }

    /**
     * 400 : multipart 요청에서 필수 파트(예: {@code file})가 빠졌을 때
     *
     * @param ex      예외
     * @param request 요청
     * @return 400 응답
     */
    @ExceptionHandler(MissingServletRequestPartException.class)
    public ResponseEntity<ApiError> handleMissingPart(MissingServletRequestPartException ex,
                                                      HttpServletRequest request) {
        return build(HttpStatus.BAD_REQUEST, ex.getMessage(), request);
    }

    /**
     * 500 : 그 외 모든 예외. 상세 원인은 로그로만 남긴다.
     *
     * @param ex      예외
     * @param request 요청
     * @return 500 응답
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> handleUnexpected(Exception ex, HttpServletRequest request) {
        log.error("처리되지 않은 예외: {} {}", request.getMethod(), request.getRequestURI(), ex);
        return build(HttpStatus.INTERNAL_SERVER_ERROR, "서버 내부 오류가 발생했습니다.", request);
    }

    /**
     * 공통 응답 조립.
     *
     * @param status  상태
     * @param message 메시지
     * @param request 요청
     * @return 오류 응답
     */
    private ResponseEntity<ApiError> build(HttpStatus status, String message, HttpServletRequest request) {
        ApiError body = ApiError.of(
                status.value(),
                status.getReasonPhrase(),
                message == null ? status.getReasonPhrase() : message,
                request.getRequestURI()
        );
        return ResponseEntity.status(status).body(body);
    }
}
