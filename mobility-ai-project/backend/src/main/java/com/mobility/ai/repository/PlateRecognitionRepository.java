package com.mobility.ai.repository;

import com.mobility.ai.domain.PlateRecognition;
import com.mobility.ai.domain.RecognitionStatus;
import java.time.LocalDateTime;
import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/**
 * 번호판 인식 이력 리포지토리. 대시보드 집계 쿼리도 함께 둔다.
 */
public interface PlateRecognitionRepository extends JpaRepository<PlateRecognition, Long> {

    /**
     * 기간 + 상태로 인식 이력을 검색한다. 모든 조건은 선택적이다.
     *
     * @param from     시작 시각(포함, null 허용)
     * @param to       종료 시각(포함, null 허용)
     * @param status   상태(null 허용)
     * @param pageable 페이지 정보
     * @return 인식 이력 페이지
     */
    @Query("""
            SELECT r FROM PlateRecognition r
            WHERE (:from IS NULL OR r.createdAt >= :from)
              AND (:to   IS NULL OR r.createdAt <= :to)
              AND (:status IS NULL OR r.status = :status)
            """)
    Page<PlateRecognition> search(@Param("from") LocalDateTime from,
                                  @Param("to") LocalDateTime to,
                                  @Param("status") RecognitionStatus status,
                                  Pageable pageable);

    /**
     * 인식 실패 목록(SUCCESS 가 아닌 모든 상태).
     *
     * @param pageable 페이지 정보
     * @return 실패 이력 페이지
     */
    @Query("SELECT r FROM PlateRecognition r WHERE r.status <> com.mobility.ai.domain.RecognitionStatus.SUCCESS")
    Page<PlateRecognition> findFailures(Pageable pageable);

    /**
     * 기간 내 전체 인식 건수.
     *
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 건수
     */
    @Query("SELECT COUNT(r) FROM PlateRecognition r WHERE r.createdAt >= :from AND r.createdAt < :to")
    long countInRange(@Param("from") LocalDateTime from, @Param("to") LocalDateTime to);

    /**
     * 기간 내 특정 상태 건수.
     *
     * @param status 상태
     * @param from   시작 시각(포함)
     * @param to     종료 시각(미포함)
     * @return 건수
     */
    @Query("""
            SELECT COUNT(r) FROM PlateRecognition r
            WHERE r.status = :status AND r.createdAt >= :from AND r.createdAt < :to
            """)
    long countByStatusInRange(@Param("status") RecognitionStatus status,
                              @Param("from") LocalDateTime from,
                              @Param("to") LocalDateTime to);

    /**
     * 기간 내 실패(SUCCESS 가 아닌) 건수.
     *
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 건수
     */
    @Query("""
            SELECT COUNT(r) FROM PlateRecognition r
            WHERE r.status <> com.mobility.ai.domain.RecognitionStatus.SUCCESS
              AND r.createdAt >= :from AND r.createdAt < :to
            """)
    long countFailuresInRange(@Param("from") LocalDateTime from, @Param("to") LocalDateTime to);

    /**
     * 기간 내 평균 총 지연(ms). 데이터가 없으면 null.
     *
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 평균 지연 또는 null
     */
    @Query("""
            SELECT AVG(r.totalMs) FROM PlateRecognition r
            WHERE r.totalMs IS NOT NULL AND r.createdAt >= :from AND r.createdAt < :to
            """)
    Double avgTotalMsInRange(@Param("from") LocalDateTime from, @Param("to") LocalDateTime to);

    /**
     * 상태별 건수 집계(기간 내). 각 행은 {@code [RecognitionStatus, Long]} 이다.
     *
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 집계 행 목록
     */
    @Query("""
            SELECT r.status, COUNT(r) FROM PlateRecognition r
            WHERE r.createdAt >= :from AND r.createdAt < :to
            GROUP BY r.status
            """)
    List<Object[]> countGroupByStatus(@Param("from") LocalDateTime from, @Param("to") LocalDateTime to);
}
