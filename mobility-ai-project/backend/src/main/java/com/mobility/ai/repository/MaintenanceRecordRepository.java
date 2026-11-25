package com.mobility.ai.repository;

import com.mobility.ai.domain.MaintenanceRecord;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/**
 * 정비 이력 리포지토리.
 */
public interface MaintenanceRecordRepository extends JpaRepository<MaintenanceRecord, Long> {

    /**
     * 차량별 정비 이력을 검색한다. {@code vehicleId} 가 null 이면 전체.
     *
     * <p>{@code JOIN FETCH} 로 차량을 함께 읽어 DTO 변환 시 N+1 을 막는다.
     *
     * @param vehicleId 차량 PK(null 허용)
     * @param pageable  페이지 정보
     * @return 정비 이력 페이지
     */
    @Query(value = """
            SELECT m FROM MaintenanceRecord m JOIN FETCH m.vehicle v
            WHERE (:vehicleId IS NULL OR v.id = :vehicleId)
            """,
            countQuery = """
            SELECT COUNT(m) FROM MaintenanceRecord m
            WHERE (:vehicleId IS NULL OR m.vehicle.id = :vehicleId)
            """)
    Page<MaintenanceRecord> search(@Param("vehicleId") Long vehicleId, Pageable pageable);
}
