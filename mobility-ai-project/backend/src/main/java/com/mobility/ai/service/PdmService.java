package com.mobility.ai.service;

import com.mobility.ai.client.AiInferenceClient;
import com.mobility.ai.client.AiPdmResult;
import com.mobility.ai.common.BusinessException;
import com.mobility.ai.common.JsonCodec;
import com.mobility.ai.common.NotFoundException;
import com.mobility.ai.common.PageResponse;
import com.mobility.ai.config.AppProperties;
import com.mobility.ai.domain.PdmPrediction;
import com.mobility.ai.domain.RiskLevel;
import com.mobility.ai.domain.SensorReading;
import com.mobility.ai.domain.Vehicle;
import com.mobility.ai.dto.PdmDtos.ComponentScore;
import com.mobility.ai.dto.PdmDtos.PdmDetail;
import com.mobility.ai.dto.PdmDtos.PdmPredictionResponse;
import com.mobility.ai.dto.PdmDtos.Recommendation;
import com.mobility.ai.dto.PdmDtos.RiskyVehicleResponse;
import com.mobility.ai.repository.PdmPredictionRepository;
import com.mobility.ai.repository.SensorReadingRepository;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 예지보전(PdM) 서비스 — 교재 6장 "예지보전 연계" 절 인용 대상.
 *
 * <h2>처리 순서</h2>
 * <ol>
 *   <li>차량의 최근 센서 {@code app.pdm.readings-window} 건을 읽는다</li>
 *   <li>AI 계약서 형식(스네이크 케이스)으로 변환해 {@code POST /api/v1/pdm/predict} 호출</li>
 *   <li>결과를 {@code pdm_prediction} 에 저장(부품/권고는 {@code detail} JSONB)</li>
 *   <li>응답 조립</li>
 * </ol>
 *
 * <p>AI 장애 시에도 500 을 던지지 않는다. {@code riskLevel=LOW},
 * {@code modelVersion="unavailable"} 로 기록해 운영자가 폴백 여부를 구분할 수 있게 한다.
 */
@Service
public class PdmService {

    private static final Logger log = LoggerFactory.getLogger(PdmService.class);

    private final AiInferenceClient aiInferenceClient;
    private final PdmPredictionRepository pdmPredictionRepository;
    private final SensorReadingRepository sensorReadingRepository;
    private final VehicleService vehicleService;
    private final JsonCodec jsonCodec;
    private final AppProperties appProperties;

    /**
     * @param aiInferenceClient       AI 호출 클라이언트
     * @param pdmPredictionRepository 예측 리포지토리
     * @param sensorReadingRepository 센서 리포지토리
     * @param vehicleService          차량 조회
     * @param jsonCodec               JSONB 직렬화/역직렬화
     * @param appProperties           {@code app.*} 설정
     */
    public PdmService(AiInferenceClient aiInferenceClient,
                      PdmPredictionRepository pdmPredictionRepository,
                      SensorReadingRepository sensorReadingRepository,
                      VehicleService vehicleService,
                      JsonCodec jsonCodec,
                      AppProperties appProperties) {
        this.aiInferenceClient = aiInferenceClient;
        this.pdmPredictionRepository = pdmPredictionRepository;
        this.sensorReadingRepository = sensorReadingRepository;
        this.vehicleService = vehicleService;
        this.jsonCodec = jsonCodec;
        this.appProperties = appProperties;
    }

    /**
     * 예측을 실행하고 저장한다.
     *
     * @param vehicleId 차량 PK
     * @return 저장된 예측 응답
     * @throws NotFoundException 차량이 없을 때
     * @throws BusinessException 센서 데이터가 전혀 없을 때
     */
    @Transactional
    public PdmPredictionResponse predict(Long vehicleId) {
        Vehicle vehicle = vehicleService.getEntity(vehicleId);

        int window = appProperties.getPdm().getReadingsWindow();
        List<SensorReading> recent = sensorReadingRepository.findRecentByVehicle(
                vehicleId, PageRequest.of(0, Math.max(1, window)));

        if (recent.isEmpty()) {
            throw new BusinessException(
                    "예측에 사용할 센서 데이터가 없습니다. 먼저 POST /api/v1/sensors 로 데이터를 적재하세요.");
        }

        // AI 는 시간 오름차순 시계열을 기대하므로 뒤집어 보낸다.
        List<SensorReading> ordered = new ArrayList<>(recent);
        Collections.reverse(ordered);

        Integer odometerKm = recent.get(0).getOdometerKm();
        AiPdmResult ai = aiInferenceClient.predictPdm(vehicleId, odometerKm, toAiReadings(ordered));

        if (!ai.available()) {
            log.warn("AI 예지보전 서비스를 사용할 수 없어 폴백 결과를 저장합니다: vehicleId={}", vehicleId);
        }

        PdmPrediction prediction = new PdmPrediction(vehicle, RiskLevel.from(ai.riskLevel()));
        prediction.setHealthScore(ai.healthScore());
        prediction.setAnomalyScore(ai.anomalyScore());
        prediction.setFailureRisk(ai.failureRisk());
        prediction.setRulDays(ai.rulDays());
        prediction.setTopComponent(ai.topComponent());
        prediction.setModelVersion(ai.modelVersion());
        prediction.setDetail(jsonCodec.toJson(
                new PdmDetail(ai.components(), ai.recommendations())));

        prediction = pdmPredictionRepository.save(prediction);
        return PdmPredictionResponse.from(prediction, ai.components(), ai.recommendations());
    }

    /**
     * 차량의 최신 예측을 조회한다.
     *
     * @param vehicleId 차량 PK
     * @return 최신 예측 응답
     * @throws NotFoundException 차량이 없거나 예측 기록이 없을 때
     */
    @Transactional(readOnly = true)
    public PdmPredictionResponse latest(Long vehicleId) {
        vehicleService.getEntity(vehicleId);

        List<PdmPrediction> found = pdmPredictionRepository.findLatestByVehicle(
                vehicleId, PageRequest.of(0, 1));
        if (found.isEmpty()) {
            throw new NotFoundException("차량 " + vehicleId + " 의 예지보전 예측 기록이 없습니다.");
        }

        PdmPrediction prediction = found.get(0);
        PdmDetail detail = jsonCodec.fromJson(prediction.getDetail(), PdmDetail.class);
        return PdmPredictionResponse.from(
                prediction,
                detail == null ? List.of() : safeComponents(detail),
                detail == null ? List.of() : safeRecommendations(detail)
        );
    }

    /**
     * 위험 차량 목록. 차량별 최신 예측 기준으로 {@code level} 이상만 돌려준다.
     *
     * @param level LOW|MEDIUM|HIGH|CRITICAL (null 이면 HIGH)
     * @param page  페이지 번호
     * @param size  페이지 크기
     * @return 위험 차량 페이지
     */
    @Transactional(readOnly = true)
    public PageResponse<RiskyVehicleResponse> risky(String level, int page, int size) {
        int minSeverity = severityOf(level == null || level.isBlank() ? "HIGH" : level);
        int safeSize = normalizeSize(size);
        int safePage = Math.max(0, page);

        List<Object[]> rows = pdmPredictionRepository.findRiskyVehicles(
                minSeverity, safeSize, safePage * safeSize);
        long total = pdmPredictionRepository.countRiskyVehiclesAtLeast(minSeverity);

        List<RiskyVehicleResponse> content = new ArrayList<>(rows.size());
        for (Object[] row : rows) {
            content.add(new RiskyVehicleResponse(
                    toLong(row[0]),
                    toStr(row[1]),
                    toStr(row[2]),
                    toStr(row[3]),
                    toDouble(row[4]),
                    toDouble(row[5]),
                    toInt(row[6]),
                    toStr(row[7]),
                    toDateTime(row[8])
            ));
        }

        int totalPages = safeSize == 0 ? 0 : (int) Math.ceil((double) total / (double) safeSize);
        return new PageResponse<>(content, total, totalPages, safePage, safeSize);
    }

    /**
     * 차량별 최신 예측 기준 위험 등급 분포(대시보드 {@code riskDistribution}).
     * 4개 등급 키를 항상 포함한다(없는 등급은 0).
     *
     * @return 등급 → 차량 수
     */
    @Transactional(readOnly = true)
    public Map<String, Long> riskDistribution() {
        Map<String, Long> distribution = new LinkedHashMap<>();
        for (RiskLevel level : RiskLevel.values()) {
            distribution.put(level.name(), 0L);
        }
        for (Object[] row : pdmPredictionRepository.countLatestGroupByRiskLevel()) {
            String key = toStr(row[0]);
            if (key != null && distribution.containsKey(key)) {
                distribution.put(key, toLongOrZero(row[1]));
            }
        }
        return distribution;
    }

    /**
     * HIGH/CRITICAL 차량 수(대시보드 {@code riskyVehicles}).
     *
     * @return 위험 차량 수
     */
    @Transactional(readOnly = true)
    public long countRiskyVehicles() {
        return pdmPredictionRepository.countRiskyVehicles();
    }

    // =================================================================
    // 내부 변환
    // =================================================================

    /**
     * 센서 엔티티 목록을 AI 계약서의 스네이크 케이스 Map 목록으로 바꾼다.
     *
     * @param readings 센서 목록(시간 오름차순)
     * @return AI 요청 {@code readings} 배열
     */
    private List<Map<String, Object>> toAiReadings(List<SensorReading> readings) {
        List<Map<String, Object>> list = new ArrayList<>(readings.size());
        for (SensorReading s : readings) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("measured_at", s.getMeasuredAt() == null ? null : s.getMeasuredAt().toString());
            m.put("engine_temp_c", s.getEngineTempC());
            m.put("engine_rpm", s.getEngineRpm());
            m.put("speed_kph", s.getSpeedKph());
            m.put("battery_voltage", s.getBatteryVoltage());
            m.put("battery_soh", s.getBatterySoh());
            m.put("tire_pressure_fl", s.getTirePressureFl());
            m.put("tire_pressure_fr", s.getTirePressureFr());
            m.put("tire_pressure_rl", s.getTirePressureRl());
            m.put("tire_pressure_rr", s.getTirePressureRr());
            m.put("brake_pad_mm", s.getBrakePadMm());
            m.put("oil_quality", s.getOilQuality());
            m.put("vibration_rms", s.getVibrationRms());
            m.put("fuel_level", s.getFuelLevel());
            list.add(m);
        }
        return list;
    }

    /**
     * 위험 등급을 숫자 심각도로 환산한다(네이티브 쿼리와 동일 규칙).
     *
     * @param level 등급 문자열
     * @return 1(LOW) ~ 4(CRITICAL)
     */
    private int severityOf(String level) {
        return switch (RiskLevel.from(level)) {
            case CRITICAL -> 4;
            case HIGH -> 3;
            case MEDIUM -> 2;
            case LOW -> 1;
        };
    }

    /**
     * 상세에서 부품 목록을 안전하게 꺼낸다.
     *
     * @param detail 복원된 상세
     * @return 부품 목록(null 이면 빈 목록)
     */
    private List<ComponentScore> safeComponents(PdmDetail detail) {
        return detail.components() == null ? List.of() : detail.components();
    }

    /**
     * 상세에서 권고 목록을 안전하게 꺼낸다.
     *
     * @param detail 복원된 상세
     * @return 권고 목록(null 이면 빈 목록)
     */
    private List<Recommendation> safeRecommendations(PdmDetail detail) {
        return detail.recommendations() == null ? List.of() : detail.recommendations();
    }

    /**
     * 네이티브 결과를 Long 으로 변환한다.
     *
     * @param value 원본 값
     * @return Long 또는 null
     */
    private Long toLong(Object value) {
        return value instanceof Number n ? n.longValue() : null;
    }

    /**
     * 네이티브 결과를 long 으로 변환한다(null 은 0).
     *
     * @param value 원본 값
     * @return long 값
     */
    private long toLongOrZero(Object value) {
        return value instanceof Number n ? n.longValue() : 0L;
    }

    /**
     * 네이티브 결과를 Integer 로 변환한다.
     *
     * @param value 원본 값
     * @return Integer 또는 null
     */
    private Integer toInt(Object value) {
        return value instanceof Number n ? n.intValue() : null;
    }

    /**
     * 네이티브 결과를 Double 로 변환한다.
     *
     * @param value 원본 값
     * @return Double 또는 null
     */
    private Double toDouble(Object value) {
        return value instanceof Number n ? n.doubleValue() : null;
    }

    /**
     * 네이티브 결과를 String 으로 변환한다.
     *
     * @param value 원본 값
     * @return 문자열 또는 null
     */
    private String toStr(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    /**
     * 네이티브 결과를 LocalDateTime 으로 변환한다.
     * JDBC 드라이버에 따라 {@link Timestamp} 또는 {@link LocalDateTime} 이 온다.
     *
     * @param value 원본 값
     * @return LocalDateTime 또는 null
     */
    private LocalDateTime toDateTime(Object value) {
        if (value instanceof LocalDateTime ldt) {
            return ldt;
        }
        if (value instanceof Timestamp ts) {
            return ts.toLocalDateTime();
        }
        return null;
    }

    /**
     * 페이지 크기를 1~200 으로 제한한다.
     *
     * @param size 요청 크기
     * @return 보정된 크기
     */
    private int normalizeSize(int size) {
        if (size <= 0) {
            return 20;
        }
        return Math.min(size, 200);
    }
}
