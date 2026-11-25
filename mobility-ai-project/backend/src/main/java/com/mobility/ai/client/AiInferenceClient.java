package com.mobility.ai.client;

import com.mobility.ai.dto.PdmDtos.ComponentScore;
import com.mobility.ai.dto.PdmDtos.Recommendation;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;

/**
 * AI 추론 서비스(FastAPI) 호출 클라이언트.
 *
 * <p>설계 원칙
 * <ol>
 *   <li><b>장애 격리</b> — AI 서비스가 죽어 있어도 백엔드 API 는 200 을 유지해야 한다.
 *       모든 호출은 예외를 밖으로 던지지 않고 "사용 불가" 결과({@code available=false})를
 *       돌려준다. 호출자는 이 플래그만 보고 폴백 분기를 한다.</li>
 *   <li><b>스네이크 케이스 수동 매핑</b> — AI 응답 JSON 은 {@code plate_number} 처럼
 *       스네이크 케이스다. Jackson 애너테이션을 쓰지 않고 {@code Map} 으로 받아
 *       키를 직접 읽는다(계약서 0장: 커스텀 ObjectMapper 금지, Jackson import 최소화).</li>
 *   <li><b>타임아웃</b> — 연결/읽기 타임아웃은 {@code AiClientConfig} 가 주입한
 *       {@link RestClient} 에 이미 설정돼 있다.</li>
 * </ol>
 */
@Component
public class AiInferenceClient {

    private static final Logger log = LoggerFactory.getLogger(AiInferenceClient.class);

    /** 응답을 느슨한 Map 으로 받기 위한 타입 토큰 */
    private static final ParameterizedTypeReference<Map<String, Object>> MAP_TYPE =
            new ParameterizedTypeReference<Map<String, Object>>() {
            };

    /** 배열 응답(벤치마크)용 타입 토큰 */
    private static final ParameterizedTypeReference<List<Map<String, Object>>> LIST_MAP_TYPE =
            new ParameterizedTypeReference<List<Map<String, Object>>>() {
            };

    private final RestClient aiRestClient;

    /**
     * @param aiRestClient {@code AiClientConfig} 가 만든 AI 서비스 전용 RestClient
     */
    public AiInferenceClient(RestClient aiRestClient) {
        this.aiRestClient = aiRestClient;
    }

    // =================================================================
    // 1) 번호판 인식
    // =================================================================

    /**
     * 이미지 1장을 AI 서비스로 보내 번호판을 인식한다.
     *
     * @param imageBytes  이미지 바이트
     * @param filename    원본 파일명(AI 서비스 로그 추적용)
     * @param contentType 이미지 MIME 타입(null 이면 image/jpeg)
     * @param minConf     최소 신뢰도(null 이면 AI 기본값 0.25)
     * @param detector    검출기(yolo|mobilenet, null 이면 기본값)
     * @param recognizer  OCR(crnn|attention|paddle, null 이면 기본값)
     * @return 인식 결과. AI 서비스 장애 시 {@link AiRecognitionResult#unavailable()}
     */
    public AiRecognitionResult recognize(byte[] imageBytes,
                                         String filename,
                                         String contentType,
                                         Double minConf,
                                         String detector,
                                         String recognizer) {
        MultiValueMap<String, Object> form = new LinkedMultiValueMap<>();
        form.add("file", filePart(imageBytes, filename));
        if (minConf != null) {
            form.add("min_conf", String.valueOf(minConf));
        }
        if (detector != null && !detector.isBlank()) {
            form.add("detector", detector);
        }
        if (recognizer != null && !recognizer.isBlank()) {
            form.add("recognizer", recognizer);
        }

        try {
            Map<String, Object> body = aiRestClient.post()
                    .uri("/api/v1/lpr/recognize")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(form)
                    .retrieve()
                    .body(MAP_TYPE);
            return toRecognitionResult(body);
        } catch (Exception e) {
            log.warn("AI 번호판 인식 호출 실패 — 폴백 처리: {}", e.getMessage());
            return AiRecognitionResult.unavailable();
        }
    }

    /**
     * AI 응답 Map 을 내부 값 객체로 변환한다.
     *
     * @param body AI 응답 본문(null 가능)
     * @return 인식 결과
     */
    private AiRecognitionResult toRecognitionResult(Map<String, Object> body) {
        if (body == null || body.isEmpty()) {
            return AiRecognitionResult.unavailable();
        }

        Map<String, Object> elapsed = asMap(body.get("elapsed_ms"));
        Map<String, Object> quality = asMap(body.get("quality"));

        List<AiRecognitionResult.AiDetection> detections = new ArrayList<>();
        for (Object raw : asList(body.get("detections"))) {
            Map<String, Object> d = asMap(raw);
            if (d.isEmpty()) {
                continue;
            }
            List<Integer> bbox = new ArrayList<>();
            for (Object v : asList(d.get("bbox"))) {
                Integer i = asInt(v);
                if (i != null) {
                    bbox.add(i);
                }
            }
            detections.add(new AiRecognitionResult.AiDetection(
                    bbox, asDouble(d.get("score")), asString(d.get("label"))));
        }

        return new AiRecognitionResult(
                asString(body.get("plate_number")),
                asDouble(body.get("confidence")),
                asString(body.get("status")),
                asBoolean(body.get("valid_format")),
                asString(body.get("detector")),
                asString(body.get("recognizer")),
                asDouble(elapsed.get("detect")),
                asDouble(elapsed.get("ocr")),
                asDouble(elapsed.get("total")),
                asDouble(quality.get("score")),
                asBoolean(quality.get("readable")),
                detections,
                true
        );
    }

    // =================================================================
    // 2) 예지보전 예측
    // =================================================================

    /**
     * 최근 센서 시계열로 예지보전 예측을 요청한다.
     *
     * @param vehicleId  차량 PK
     * @param odometerKm 최신 주행거리(null 가능)
     * @param readings   센서 레코드 목록. 각 Map 의 키는 AI 계약서의 스네이크 케이스
     *                   ({@code measured_at}, {@code engine_temp_c} …)여야 한다.
     * @return 예측 결과. AI 서비스 장애 시 {@link AiPdmResult#unavailable(Long)}
     */
    public AiPdmResult predictPdm(Long vehicleId, Integer odometerKm, List<Map<String, Object>> readings) {
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("vehicle_id", vehicleId);
        request.put("odometer_km", odometerKm);
        request.put("readings", readings == null ? List.of() : readings);

        try {
            Map<String, Object> body = aiRestClient.post()
                    .uri("/api/v1/pdm/predict")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(request)
                    .retrieve()
                    .body(MAP_TYPE);
            return toPdmResult(vehicleId, body);
        } catch (Exception e) {
            log.warn("AI 예지보전 호출 실패 — 폴백 처리: {}", e.getMessage());
            return AiPdmResult.unavailable(vehicleId);
        }
    }

    /**
     * AI 예측 응답 Map 을 내부 값 객체로 변환한다.
     *
     * @param vehicleId 요청한 차량 PK(응답에 없을 때 사용)
     * @param body      응답 본문
     * @return 예측 결과
     */
    private AiPdmResult toPdmResult(Long vehicleId, Map<String, Object> body) {
        if (body == null || body.isEmpty()) {
            return AiPdmResult.unavailable(vehicleId);
        }

        List<ComponentScore> components = new ArrayList<>();
        for (Object raw : asList(body.get("components"))) {
            Map<String, Object> c = asMap(raw);
            if (c.isEmpty()) {
                continue;
            }
            components.add(new ComponentScore(
                    asString(c.get("name")),
                    asDouble(c.get("score")),
                    asString(c.get("status")),
                    asString(c.get("reason"))
            ));
        }

        List<Recommendation> recommendations = new ArrayList<>();
        for (Object raw : asList(body.get("recommendations"))) {
            Map<String, Object> r = asMap(raw);
            if (r.isEmpty()) {
                continue;
            }
            recommendations.add(new Recommendation(
                    asInt(r.get("priority")),
                    asString(r.get("item")),
                    asInt(r.get("due_km")),
                    asInt(r.get("due_days"))
            ));
        }

        Long resolvedVehicleId = asLong(body.get("vehicle_id"));
        return new AiPdmResult(
                resolvedVehicleId == null ? vehicleId : resolvedVehicleId,
                asDouble(body.get("health_score")),
                asDouble(body.get("anomaly_score")),
                asDouble(body.get("failure_risk")),
                asString(body.get("risk_level")),
                asInt(body.get("rul_days")),
                asString(body.get("top_component")),
                components,
                recommendations,
                asString(body.get("model_version")),
                true
        );
    }

    // =================================================================
    // 3) 모델 벤치마크
    // =================================================================

    /**
     * AI 서비스에서 벤치마크를 실행하고 결과 배열을 받는다.
     *
     * @return 벤치마크 결과 목록. 호출 실패 시 빈 리스트
     */
    public List<AiBenchmarkResult> runBenchmark() {
        try {
            List<Map<String, Object>> body = aiRestClient.post()
                    .uri("/api/v1/models/benchmark")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(Map.of())
                    .retrieve()
                    .body(LIST_MAP_TYPE);

            if (body == null) {
                return List.of();
            }
            List<AiBenchmarkResult> results = new ArrayList<>();
            for (Map<String, Object> row : body) {
                if (row == null || row.isEmpty()) {
                    continue;
                }
                results.add(new AiBenchmarkResult(
                        asString(row.get("model_name")),
                        asString(row.get("task")),
                        asDouble(row.get("detect_recall")),
                        asDouble(row.get("exact_match_rate")),
                        asDouble(row.get("latency_p50_ms")),
                        asDouble(row.get("latency_p95_ms")),
                        asDouble(row.get("memory_mb")),
                        asDouble(row.get("model_size_mb")),
                        Boolean.TRUE.equals(asBoolean(row.get("quantized")))
                ));
            }
            return results;
        } catch (Exception e) {
            log.warn("AI 벤치마크 호출 실패: {}", e.getMessage());
            return List.of();
        }
    }

    /**
     * AI 서비스 헬스 체크.
     *
     * @return 살아 있으면 true
     */
    public boolean isHealthy() {
        try {
            Map<String, Object> body = aiRestClient.get()
                    .uri("/health")
                    .retrieve()
                    .body(MAP_TYPE);
            return body != null && "UP".equalsIgnoreCase(asString(body.get("status")));
        } catch (Exception e) {
            log.debug("AI 헬스 체크 실패: {}", e.getMessage());
            return false;
        }
    }

    // =================================================================
    // 내부 유틸 — 느슨한 타입 변환
    // =================================================================

    /**
     * multipart 의 파일 파트를 만든다. {@code getFilename()} 을 재정의해야
     * {@code Content-Disposition} 에 filename 이 실린다.
     *
     * @param bytes    파일 내용
     * @param filename 파일명(null 이면 upload.jpg)
     * @return 리소스 파트
     */
    private ByteArrayResource filePart(byte[] bytes, String filename) {
        final String name = (filename == null || filename.isBlank()) ? "upload.jpg" : filename;
        return new ByteArrayResource(bytes == null ? new byte[0] : bytes) {
            @Override
            public String getFilename() {
                return name;
            }
        };
    }

    /**
     * 임의 객체를 {@code Map<String,Object>} 로 간주한다.
     *
     * @param value 원본 값
     * @return 변환된 Map, 불가능하면 빈 Map
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> asMap(Object value) {
        if (value instanceof Map<?, ?> map) {
            return (Map<String, Object>) map;
        }
        return Map.of();
    }

    /**
     * 임의 객체를 List 로 간주한다.
     *
     * @param value 원본 값
     * @return 변환된 List, 불가능하면 빈 List
     */
    private List<Object> asList(Object value) {
        if (value instanceof List<?> list) {
            return new ArrayList<>(list);
        }
        return List.of();
    }

    /**
     * 문자열 변환(null 안전).
     *
     * @param value 원본 값
     * @return 문자열 또는 null
     */
    private String asString(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    /**
     * Double 변환(null 안전, 문자열 숫자도 허용).
     *
     * @param value 원본 값
     * @return Double 또는 null
     */
    private Double asDouble(Object value) {
        if (value instanceof Number n) {
            return n.doubleValue();
        }
        if (value instanceof String s && !s.isBlank()) {
            try {
                return Double.valueOf(s);
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    /**
     * Integer 변환(null 안전).
     *
     * @param value 원본 값
     * @return Integer 또는 null
     */
    private Integer asInt(Object value) {
        if (value instanceof Number n) {
            return n.intValue();
        }
        if (value instanceof String s && !s.isBlank()) {
            try {
                return (int) Double.parseDouble(s);
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    /**
     * Long 변환(null 안전).
     *
     * @param value 원본 값
     * @return Long 또는 null
     */
    private Long asLong(Object value) {
        if (value instanceof Number n) {
            return n.longValue();
        }
        if (value instanceof String s && !s.isBlank()) {
            try {
                return (long) Double.parseDouble(s);
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    /**
     * Boolean 변환(null 안전).
     *
     * @param value 원본 값
     * @return Boolean 또는 null
     */
    private Boolean asBoolean(Object value) {
        if (value instanceof Boolean b) {
            return b;
        }
        if (value instanceof String s && !s.isBlank()) {
            return Boolean.valueOf(s);
        }
        return null;
    }
}
