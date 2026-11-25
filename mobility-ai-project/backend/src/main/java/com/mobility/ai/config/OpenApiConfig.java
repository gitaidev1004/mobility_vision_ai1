package com.mobility.ai.config;

import java.util.List;
import java.util.Map;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * API 설명 문서(선택 구성).
 *
 * <p>springdoc-openapi 는 Spring Boot 4 / Jackson 3 조합에서 버전 정렬이 필요해
 * 교재 부록에서는 의존성을 추가하지 않는다. 대신 외부 라이브러리 없이
 * {@code GET /api/v1/meta/endpoints} 로 <b>엔드포인트 목록</b>만 돌려주는
 * 가벼운 문서 엔드포인트를 둔다. (인증 필요)
 *
 * <p>정식 OpenAPI 문서가 필요하면 {@code org.springdoc:springdoc-openapi-starter-webmvc-ui}
 * 를 추가하고 이 클래스를 제거하면 된다.
 */
@Configuration
public class OpenApiConfig {

    /**
     * 메타 정보 컨트롤러를 빈으로 등록한다.
     *
     * @return 메타 컨트롤러
     */
    @Bean
    public ApiMetaController apiMetaController() {
        return new ApiMetaController();
    }

    /**
     * 계약서 4장 표를 그대로 노출하는 간단한 문서 컨트롤러.
     */
    @RestController
    public static class ApiMetaController {

        /**
         * 백엔드 엔드포인트 목록.
         *
         * @return {@code {"version", "endpoints":[{"method","path","description"}]}}
         */
        @GetMapping("/api/v1/meta/endpoints")
        public Map<String, Object> endpoints() {
            List<Map<String, String>> list = List.of(
                    endpoint("POST", "/api/v1/auth/login", "로그인(JWT 발급)"),
                    endpoint("GET", "/api/v1/auth/me", "현재 로그인 사용자"),
                    endpoint("GET", "/api/v1/vehicles", "차량 목록(페이지)"),
                    endpoint("POST", "/api/v1/vehicles", "차량 등록"),
                    endpoint("GET", "/api/v1/vehicles/{id}", "차량 단건"),
                    endpoint("PUT", "/api/v1/vehicles/{id}", "차량 수정"),
                    endpoint("DELETE", "/api/v1/vehicles/{id}", "차량 삭제"),
                    endpoint("GET", "/api/v1/vehicles/by-plate/{plate}", "번호로 차량 조회"),
                    endpoint("POST", "/api/v1/recognitions", "번호판 인식 + 출입/주차 처리"),
                    endpoint("GET", "/api/v1/recognitions", "인식 이력"),
                    endpoint("GET", "/api/v1/recognitions/failures", "인식 실패 목록"),
                    endpoint("GET", "/api/v1/events", "출입 이벤트"),
                    endpoint("GET", "/api/v1/parking/sessions", "주차 세션"),
                    endpoint("POST", "/api/v1/parking/sessions/{id}/pay", "주차 요금 결제"),
                    endpoint("GET", "/api/v1/parking/status", "주차 현황"),
                    endpoint("GET", "/api/v1/maintenance", "정비 이력"),
                    endpoint("POST", "/api/v1/maintenance", "정비 등록"),
                    endpoint("GET", "/api/v1/sensors", "센서 시계열"),
                    endpoint("POST", "/api/v1/sensors", "센서 적재(배열)"),
                    endpoint("POST", "/api/v1/pdm/vehicles/{id}/predict", "예지보전 예측 실행"),
                    endpoint("GET", "/api/v1/pdm/vehicles/{id}/latest", "최신 예측"),
                    endpoint("GET", "/api/v1/pdm/risky", "위험 차량 목록"),
                    endpoint("GET", "/api/v1/dashboard/summary", "대시보드 요약"),
                    endpoint("GET", "/api/v1/benchmarks", "모델 성능 목록"),
                    endpoint("POST", "/api/v1/benchmarks/run", "벤치마크 실행 후 저장")
            );
            return Map.of("version", "1.0.0", "endpoints", list);
        }

        /**
         * 엔드포인트 1건을 Map 으로 만든다.
         *
         * @param method      HTTP 메서드
         * @param path        경로
         * @param description 설명
         * @return Map 표현
         */
        private Map<String, String> endpoint(String method, String path, String description) {
            return Map.of("method", method, "path", path, "description", description);
        }
    }
}
