package com.mobility.ai.repository;

import com.mobility.ai.domain.Vehicle;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/**
 * 차량 리포지토리.
 */
public interface VehicleRepository extends JpaRepository<Vehicle, Long> {

    /**
     * 번호판으로 차량을 찾는다(인식 결과 매칭의 핵심 경로).
     *
     * @param plateNumber 차량 번호
     * @return 차량(없으면 empty)
     */
    Optional<Vehicle> findByPlateNumber(String plateNumber);

    /**
     * 번호판 중복 확인.
     *
     * @param plateNumber 차량 번호
     * @return 있으면 true
     */
    boolean existsByPlateNumber(String plateNumber);

    /**
     * 번호판 또는 소유자 이름 부분 일치 검색(대소문자 무시).
     * {@code q} 가 비어 있으면 전체를 돌려준다.
     *
     * @param q        검색어(null/빈 문자열 허용)
     * @param pageable 페이지 정보
     * @return 차량 페이지
     */
    @Query("""
            SELECT v FROM Vehicle v
            WHERE :q IS NULL OR :q = ''
               OR LOWER(v.plateNumber) LIKE LOWER(CONCAT('%', :q, '%'))
               OR LOWER(v.ownerName)   LIKE LOWER(CONCAT('%', :q, '%'))
            """)
    Page<Vehicle> search(@Param("q") String q, Pageable pageable);
}
