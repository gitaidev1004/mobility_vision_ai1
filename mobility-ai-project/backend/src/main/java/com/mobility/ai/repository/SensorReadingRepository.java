package com.mobility.ai.repository;

import com.mobility.ai.domain.SensorReading;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/**
 * 센서 시계열 리포지토리.
 */
public interface SensorReadingRepository extends JpaRepository<SensorReading, Long> {

    /**
     * 차량별 최근 센서 데이터를 측정 시각 내림차순으로 읽는다.
     * {@code PdmService} 가 최근 N건을 모아 AI 로 보낸다.
     *
     * @param vehicleId 차량 PK
     * @param pageable  크기 제한용 페이지 정보
     * @return 최근 센서 목록
     */
    @Query("""
            SELECT s FROM SensorReading s
            WHERE s.vehicle.id = :vehicleId
            ORDER BY s.measuredAt DESC
            """)
    List<SensorReading> findRecentByVehicle(@Param("vehicleId") Long vehicleId, Pageable pageable);

    /**
     * 전체 차량의 최근 센서 데이터(차량 미지정 조회용).
     *
     * @param pageable 크기 제한용 페이지 정보
     * @return 최근 센서 목록
     */
    @Query("SELECT s FROM SensorReading s ORDER BY s.measuredAt DESC")
    List<SensorReading> findRecent(Pageable pageable);
}
