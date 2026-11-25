package com.mobility.ai.repository;

import com.mobility.ai.domain.ParkingSession;
import java.time.LocalDateTime;
import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/**
 * 주차 세션 리포지토리. 주차 현황/매출 집계 쿼리를 포함한다.
 */
public interface ParkingSessionRepository extends JpaRepository<ParkingSession, Long> {

    /**
     * 주차 중(출차 시각 없음) 세션만 검색한다.
     *
     * <p>세 가지 조건을 하나의 JPQL 에 {@code :active IS NULL OR :active = TRUE ...}
     * 로 합치지 않는다. 그렇게 쓰면 Boolean 파라미터가 엔티티 속성과 비교되지 않아
     * Hibernate 가 파라미터 타입을 추론할 수 없고, 방언에 따라 기동이 실패한다.
     * 메서드를 분리하는 편이 생성되는 SQL 도 단순하다.
     *
     * @param pageable 페이지 정보
     * @return 주차 중 세션 페이지
     */
    @Query("SELECT s FROM ParkingSession s WHERE s.exitAt IS NULL")
    Page<ParkingSession> findActiveSessions(Pageable pageable);

    /**
     * 출차 완료 세션만 검색한다.
     *
     * @param pageable 페이지 정보
     * @return 완료 세션 페이지
     */
    @Query("SELECT s FROM ParkingSession s WHERE s.exitAt IS NOT NULL")
    Page<ParkingSession> findClosedSessions(Pageable pageable);

    /**
     * 번호판으로 열린(주차 중) 세션을 최신순으로 찾는다.
     * 출차 처리 시 크기 1 의 Pageable 로 호출해 최신 1건을 집는다.
     *
     * @param plateNumber 번호판
     * @param pageable    크기 제한용 페이지 정보
     * @return 열린 세션 목록
     */
    @Query("""
            SELECT s FROM ParkingSession s
            WHERE s.plateNumber = :plateNumber AND s.exitAt IS NULL
            ORDER BY s.entryAt DESC
            """)
    List<ParkingSession> findOpenByPlateNumber(@Param("plateNumber") String plateNumber, Pageable pageable);

    /**
     * 현재 주차 대수(출차 시각이 없는 세션 수).
     *
     * @return 주차 중 세션 수
     */
    @Query("SELECT COUNT(s) FROM ParkingSession s WHERE s.exitAt IS NULL")
    long countActive();

    /**
     * 미결제 세션 수: 출차했고 요금이 0보다 큰데 아직 결제되지 않은 건.
     *
     * @return 미결제 건수
     */
    @Query("""
            SELECT COUNT(s) FROM ParkingSession s
            WHERE s.exitAt IS NOT NULL AND s.paid = false
              AND s.feeAmount IS NOT NULL AND s.feeAmount > 0
            """)
    long countUnpaid();

    /**
     * 기간 내 결제 완료 매출 합계. 데이터가 없으면 null.
     *
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 매출 합계 또는 null
     */
    @Query("""
            SELECT SUM(s.feeAmount) FROM ParkingSession s
            WHERE s.paid = true AND s.exitAt >= :from AND s.exitAt < :to
            """)
    Long sumPaidFeeInRange(@Param("from") LocalDateTime from, @Param("to") LocalDateTime to);
}
