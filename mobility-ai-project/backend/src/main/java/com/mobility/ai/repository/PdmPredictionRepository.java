package com.mobility.ai.repository;

import com.mobility.ai.domain.PdmPrediction;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/**
 * 예지보전 예측 리포지토리.
 *
 * <p>대시보드의 위험 분포와 위험 차량 목록은 "차량별 최신 예측 1건"을 기준으로
 * 집계해야 하므로, PostgreSQL 의 {@code DISTINCT ON} 을 쓰는 네이티브 쿼리를 둔다.
 */
public interface PdmPredictionRepository extends JpaRepository<PdmPrediction, Long> {

    /**
     * 차량의 최신 예측을 읽는다(크기 1 Pageable 로 호출).
     *
     * @param vehicleId 차량 PK
     * @param pageable  크기 제한용 페이지 정보
     * @return 최신 예측 목록
     */
    @Query("""
            SELECT p FROM PdmPrediction p
            WHERE p.vehicle.id = :vehicleId
            ORDER BY p.predictedAt DESC
            """)
    List<PdmPrediction> findLatestByVehicle(@Param("vehicleId") Long vehicleId, Pageable pageable);

    /**
     * 차량별 최신 예측을 기준으로 위험 등급별 차량 수를 센다.
     * 각 행은 {@code [risk_level(String), count(long)]} 이다.
     *
     * @return 등급별 집계 행
     */
    @Query(value = """
            SELECT latest.risk_level, COUNT(*)
            FROM (
                SELECT DISTINCT ON (vehicle_id) vehicle_id, risk_level
                FROM pdm_prediction
                ORDER BY vehicle_id, predicted_at DESC
            ) AS latest
            GROUP BY latest.risk_level
            """, nativeQuery = true)
    List<Object[]> countLatestGroupByRiskLevel();

    /**
     * 차량별 최신 예측이 HIGH 또는 CRITICAL 인 차량 수(대시보드 {@code riskyVehicles}).
     *
     * @return 위험 차량 수
     */
    @Query(value = """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT ON (vehicle_id) vehicle_id, risk_level
                FROM pdm_prediction
                ORDER BY vehicle_id, predicted_at DESC
            ) AS latest
            WHERE latest.risk_level IN ('HIGH', 'CRITICAL')
            """, nativeQuery = true)
    long countRiskyVehicles();

    /**
     * 위험 차량 목록. 차량별 최신 예측 중 심각도가 {@code minSeverity} 이상인 건만
     * 심각도 내림차순 → 건강점수 오름차순으로 돌려준다.
     *
     * <p>심각도는 LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4 로 환산한다.
     * 각 행은 다음 순서다:
     * {@code [vehicle_id(Number), plate_number(String), owner_name(String),
     *          risk_level(String), health_score(Number), failure_risk(Number),
     *          rul_days(Number), top_component(String), predicted_at(Timestamp)]}
     *
     * @param minSeverity 최소 심각도(1~4)
     * @param limit       최대 건수
     * @param offset      건너뛸 건수
     * @return 위험 차량 행 목록
     */
    @Query(value = """
            SELECT latest.vehicle_id,
                   v.plate_number,
                   v.owner_name,
                   latest.risk_level,
                   latest.health_score,
                   latest.failure_risk,
                   latest.rul_days,
                   latest.top_component,
                   latest.predicted_at
            FROM (
                SELECT DISTINCT ON (vehicle_id)
                       vehicle_id, risk_level, health_score, failure_risk,
                       rul_days, top_component, predicted_at,
                       CASE risk_level
                           WHEN 'CRITICAL' THEN 4
                           WHEN 'HIGH'     THEN 3
                           WHEN 'MEDIUM'   THEN 2
                           ELSE 1
                       END AS severity
                FROM pdm_prediction
                ORDER BY vehicle_id, predicted_at DESC
            ) AS latest
            JOIN vehicle v ON v.id = latest.vehicle_id
            WHERE latest.severity >= :minSeverity
            ORDER BY latest.severity DESC, latest.health_score ASC NULLS LAST
            LIMIT :limit OFFSET :offset
            """, nativeQuery = true)
    List<Object[]> findRiskyVehicles(@Param("minSeverity") int minSeverity,
                                     @Param("limit") int limit,
                                     @Param("offset") int offset);

    /**
     * {@link #findRiskyVehicles(int, int, int)} 의 전체 건수(페이징용).
     *
     * @param minSeverity 최소 심각도(1~4)
     * @return 전체 건수
     */
    @Query(value = """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT ON (vehicle_id) vehicle_id,
                       CASE risk_level
                           WHEN 'CRITICAL' THEN 4
                           WHEN 'HIGH'     THEN 3
                           WHEN 'MEDIUM'   THEN 2
                           ELSE 1
                       END AS severity
                FROM pdm_prediction
                ORDER BY vehicle_id, predicted_at DESC
            ) AS latest
            WHERE latest.severity >= :minSeverity
            """, nativeQuery = true)
    long countRiskyVehiclesAtLeast(@Param("minSeverity") int minSeverity);
}
