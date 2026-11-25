package com.mobility.ai.service;

import com.mobility.ai.common.BusinessException;
import com.mobility.ai.domain.SensorReading;
import com.mobility.ai.domain.Vehicle;
import com.mobility.ai.dto.SensorDtos.SensorRequest;
import com.mobility.ai.dto.SensorDtos.SensorResponse;
import com.mobility.ai.repository.SensorReadingRepository;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 센서 시계열 서비스. 계약서 4장 {@code /api/v1/sensors}.
 */
@Service
public class SensorService {

    private final SensorReadingRepository sensorReadingRepository;
    private final VehicleService vehicleService;

    /**
     * @param sensorReadingRepository 센서 리포지토리
     * @param vehicleService          차량 조회(존재 검증)
     */
    public SensorService(SensorReadingRepository sensorReadingRepository,
                         VehicleService vehicleService) {
        this.sensorReadingRepository = sensorReadingRepository;
        this.vehicleService = vehicleService;
    }

    /**
     * 센서 시계열 조회(최근 데이터 우선).
     *
     * @param vehicleId 차량 PK(null 이면 전체 차량)
     * @param limit     최대 건수(1~1000, 기본 100)
     * @return 센서 응답 목록
     */
    @Transactional(readOnly = true)
    public List<SensorResponse> list(Long vehicleId, int limit) {
        PageRequest pageable = PageRequest.of(0, normalizeLimit(limit));
        List<SensorReading> readings = vehicleId == null
                ? sensorReadingRepository.findRecent(pageable)
                : sensorReadingRepository.findRecentByVehicle(vehicleId, pageable);
        return readings.stream().map(SensorResponse::from).toList();
    }

    /**
     * 센서 데이터를 일괄 적재한다.
     *
     * <p>같은 차량이 반복되는 배열을 받으므로 차량 조회 결과를 캐시해
     * 불필요한 SELECT 를 줄인다.
     *
     * @param requests 적재할 센서 목록
     * @return 저장된 센서 응답 목록
     */
    @Transactional
    public List<SensorResponse> ingest(List<SensorRequest> requests) {
        if (requests == null || requests.isEmpty()) {
            return List.of();
        }

        Map<Long, Vehicle> cache = new HashMap<>();
        List<SensorReading> entities = new ArrayList<>(requests.size());

        for (int i = 0; i < requests.size(); i++) {
            SensorRequest r = requests.get(i);

            // 배열 원소 검증은 @Valid 가 내려가지 않으므로 직접 한다.
            // 어느 원소가 왜 잘못됐는지 알려 주어야 디버깅이 가능하다.
            if (r == null) {
                throw new BusinessException("센서 배열의 " + i + "번째 원소가 null 입니다.");
            }
            if (r.vehicleId() == null) {
                throw new BusinessException("센서 배열 " + i + "번째: vehicleId 는 필수입니다.");
            }
            if (r.measuredAt() == null) {
                throw new BusinessException("센서 배열 " + i + "번째: measuredAt 은 필수입니다.");
            }

            Vehicle vehicle = cache.computeIfAbsent(r.vehicleId(), vehicleService::getEntity);

            SensorReading reading = new SensorReading(vehicle, r.measuredAt());
            reading.setOdometerKm(r.odometerKm());
            reading.setEngineTempC(r.engineTempC());
            reading.setEngineRpm(r.engineRpm());
            reading.setSpeedKph(r.speedKph());
            reading.setBatteryVoltage(r.batteryVoltage());
            reading.setBatterySoh(r.batterySoh());
            reading.setTirePressureFl(r.tirePressureFl());
            reading.setTirePressureFr(r.tirePressureFr());
            reading.setTirePressureRl(r.tirePressureRl());
            reading.setTirePressureRr(r.tirePressureRr());
            reading.setBrakePadMm(r.brakePadMm());
            reading.setOilQuality(r.oilQuality());
            reading.setVibrationRms(r.vibrationRms());
            reading.setFuelLevel(r.fuelLevel());
            entities.add(reading);
        }

        return sensorReadingRepository.saveAll(entities).stream()
                .map(SensorResponse::from)
                .toList();
    }

    /**
     * 조회 건수를 1~1000 으로 제한한다.
     *
     * @param limit 요청 건수
     * @return 보정된 건수
     */
    private int normalizeLimit(int limit) {
        if (limit <= 0) {
            return 100;
        }
        return Math.min(limit, 1000);
    }
}
