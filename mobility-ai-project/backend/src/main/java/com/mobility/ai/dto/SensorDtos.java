package com.mobility.ai.dto;

import com.mobility.ai.domain.SensorReading;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;

/**
 * 센서 시계열 DTO 모음. 계약서 4장 {@code /api/v1/sensors}.
 * 필드 구성은 {@code sensor_reading} 테이블과 1:1 이다.
 */
public final class SensorDtos {

    private SensorDtos() {
    }

    /**
     * 센서 적재 요청 1건(배열로 POST 된다).
     *
     * @param vehicleId      차량 PK(필수)
     * @param measuredAt     측정 시각(필수)
     * @param odometerKm     주행거리
     * @param engineTempC    냉각수 온도
     * @param engineRpm      엔진 RPM
     * @param speedKph       속도
     * @param batteryVoltage 배터리 전압
     * @param batterySoh     배터리 건강도
     * @param tirePressureFl 타이어 공기압 전좌
     * @param tirePressureFr 타이어 공기압 전우
     * @param tirePressureRl 타이어 공기압 후좌
     * @param tirePressureRr 타이어 공기압 후우
     * @param brakePadMm     브레이크 패드 잔량
     * @param oilQuality     오일 품질 지수
     * @param vibrationRms   진동 RMS
     * @param fuelLevel      연료 잔량
     */
    public record SensorRequest(
            @NotNull(message = "차량 ID는 필수입니다.") Long vehicleId,
            @NotNull(message = "측정 시각은 필수입니다.") LocalDateTime measuredAt,
            Integer odometerKm,
            Double engineTempC,
            Integer engineRpm,
            Double speedKph,
            Double batteryVoltage,
            Double batterySoh,
            Double tirePressureFl,
            Double tirePressureFr,
            Double tirePressureRl,
            Double tirePressureRr,
            Double brakePadMm,
            Double oilQuality,
            Double vibrationRms,
            Double fuelLevel
    ) {
    }

    /**
     * 센서 시계열 응답 1건.
     *
     * @param id             PK
     * @param vehicleId      차량 PK
     * @param measuredAt     측정 시각
     * @param odometerKm     주행거리
     * @param engineTempC    냉각수 온도
     * @param engineRpm      엔진 RPM
     * @param speedKph       속도
     * @param batteryVoltage 배터리 전압
     * @param batterySoh     배터리 건강도
     * @param tirePressureFl 타이어 공기압 전좌
     * @param tirePressureFr 타이어 공기압 전우
     * @param tirePressureRl 타이어 공기압 후좌
     * @param tirePressureRr 타이어 공기압 후우
     * @param brakePadMm     브레이크 패드 잔량
     * @param oilQuality     오일 품질 지수
     * @param vibrationRms   진동 RMS
     * @param fuelLevel      연료 잔량
     */
    public record SensorResponse(
            Long id,
            Long vehicleId,
            LocalDateTime measuredAt,
            Integer odometerKm,
            Double engineTempC,
            Integer engineRpm,
            Double speedKph,
            Double batteryVoltage,
            Double batterySoh,
            Double tirePressureFl,
            Double tirePressureFr,
            Double tirePressureRl,
            Double tirePressureRr,
            Double brakePadMm,
            Double oilQuality,
            Double vibrationRms,
            Double fuelLevel
    ) {
        /**
         * 엔티티 → DTO 변환.
         *
         * @param s 센서 엔티티
         * @return DTO
         */
        public static SensorResponse from(SensorReading s) {
            return new SensorResponse(
                    s.getId(),
                    s.getVehicle() == null ? null : s.getVehicle().getId(),
                    s.getMeasuredAt(),
                    s.getOdometerKm(),
                    s.getEngineTempC(),
                    s.getEngineRpm(),
                    s.getSpeedKph(),
                    s.getBatteryVoltage(),
                    s.getBatterySoh(),
                    s.getTirePressureFl(),
                    s.getTirePressureFr(),
                    s.getTirePressureRl(),
                    s.getTirePressureRr(),
                    s.getBrakePadMm(),
                    s.getOilQuality(),
                    s.getVibrationRms(),
                    s.getFuelLevel()
            );
        }
    }
}
