package com.mobility.ai.common;

import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import tools.jackson.databind.ObjectMapper;

/**
 * JSONB 컬럼({@code pdm_prediction.detail})을 다루기 위한 아주 얇은 JSON 코덱.
 *
 * <p>Spring Boot 4 는 Jackson 3(<code>tools.jackson</code>)을 기본으로 쓴다.
 * 여기서는 <b>자동 구성된 ObjectMapper 빈을 주입만 해서</b> 사용하며,
 * 커스텀 ObjectMapper 빈을 새로 만들지 않는다(계약서 0장 규칙).
 *
 * <p>Jackson 관련 import 를 이 한 파일에만 모아 두어, Jackson 버전이 바뀌어도
 * 수정 지점이 한 곳에 머무르게 한다.
 */
@Component
public class JsonCodec {

    private static final Logger log = LoggerFactory.getLogger(JsonCodec.class);

    private final ObjectMapper objectMapper;

    /**
     * @param objectMapper Spring Boot 가 자동 구성한 Jackson 매퍼
     */
    public JsonCodec(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    /**
     * 객체를 JSON 문자열로 바꾼다. 실패하면 null 을 돌려주고 경고만 남긴다
     * (JSONB 상세 정보는 부가 정보이므로 요청 전체를 실패시키지 않는다).
     *
     * @param value 직렬화할 객체
     * @return JSON 문자열, 실패 시 null
     */
    public String toJson(Object value) {
        if (value == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            log.warn("JSON 직렬화 실패: {}", e.getMessage());
            return null;
        }
    }

    /**
     * JSON 문자열을 단일 객체로 복원한다.
     *
     * @param json JSON 문자열
     * @param type 대상 타입
     * @param <T>  대상 타입
     * @return 복원된 객체, 실패하거나 입력이 비면 null
     */
    public <T> T fromJson(String json, Class<T> type) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return objectMapper.readValue(json, type);
        } catch (Exception e) {
            log.warn("JSON 역직렬화 실패({}): {}", type.getSimpleName(), e.getMessage());
            return null;
        }
    }

    /**
     * JSON 배열 문자열을 리스트로 복원한다.
     *
     * @param json        JSON 배열 문자열
     * @param elementType 원소 타입
     * @param <T>         원소 타입
     * @return 복원된 리스트, 실패 시 빈 리스트
     */
    public <T> List<T> fromJsonList(String json, Class<T> elementType) {
        if (json == null || json.isBlank()) {
            return List.of();
        }
        try {
            T[] array = objectMapper.readValue(json, arrayTypeOf(elementType));
            return array == null ? List.of() : List.of(array);
        } catch (Exception e) {
            log.warn("JSON 배열 역직렬화 실패({}): {}", elementType.getSimpleName(), e.getMessage());
            return List.of();
        }
    }

    /**
     * {@code T} 의 배열 클래스를 얻는다.
     *
     * @param elementType 원소 타입
     * @param <T>         원소 타입
     * @return 배열 클래스
     */
    @SuppressWarnings("unchecked")
    private <T> Class<T[]> arrayTypeOf(Class<T> elementType) {
        return (Class<T[]>) java.lang.reflect.Array.newInstance(elementType, 0).getClass();
    }
}
