package com.mobility.ai.common;

import java.util.List;
import java.util.function.Function;
import org.springframework.data.domain.Page;

/**
 * 페이지 응답 공통 포맷. 계약서 4장의
 * {@code {content,totalElements,totalPages,page,size}} 와 1:1 이다.
 *
 * <p>Spring Data 의 {@code Page} 를 그대로 직렬화하면 내부 구조가 노출되므로
 * 반드시 이 record 로 감싸 반환한다.
 *
 * @param content       현재 페이지 항목
 * @param totalElements 전체 건수
 * @param totalPages    전체 페이지 수
 * @param page          현재 페이지 번호(0-base)
 * @param size          페이지 크기
 * @param <T>           항목 타입
 */
public record PageResponse<T>(
        List<T> content,
        long totalElements,
        int totalPages,
        int page,
        int size
) {
    /**
     * 엔티티 Page 를 DTO Page 로 변환해 감싼다.
     *
     * @param source 원본 페이지
     * @param mapper 엔티티 → DTO 변환 함수
     * @param <E>    엔티티 타입
     * @param <D>    DTO 타입
     * @return 페이지 응답
     */
    public static <E, D> PageResponse<D> from(Page<E> source, Function<E, D> mapper) {
        return new PageResponse<>(
                source.getContent().stream().map(mapper).toList(),
                source.getTotalElements(),
                source.getTotalPages(),
                source.getNumber(),
                source.getSize()
        );
    }

    /**
     * 변환 없이 Page 를 그대로 감싼다.
     *
     * @param source 원본 페이지
     * @param <T>    항목 타입
     * @return 페이지 응답
     */
    public static <T> PageResponse<T> from(Page<T> source) {
        return new PageResponse<>(
                source.getContent(),
                source.getTotalElements(),
                source.getTotalPages(),
                source.getNumber(),
                source.getSize()
        );
    }

    /**
     * 리스트를 단일 페이지로 감싼다(페이징 없는 목록 API 용).
     *
     * @param items 항목 목록
     * @param <T>   항목 타입
     * @return 페이지 응답
     */
    public static <T> PageResponse<T> ofList(List<T> items) {
        return new PageResponse<>(items, items.size(), 1, 0, items.size());
    }
}
