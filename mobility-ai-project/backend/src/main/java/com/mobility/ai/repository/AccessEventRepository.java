package com.mobility.ai.repository;

import com.mobility.ai.domain.AccessEvent;
import com.mobility.ai.domain.EventType;
import java.time.LocalDateTime;
import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

/**
 * 출입 이벤트 리포지토리. 중복 제거와 대시보드 집계 쿼리를 포함한다.
 */
public interface AccessEventRepository extends JpaRepository<AccessEvent, Long> {

    /**
     * 기간 + 타입으로 이벤트를 검색한다. 모든 조건은 선택적이다.
     *
     * @param type     이벤트 타입(null 허용)
     * @param from     시작 시각(포함, null 허용)
     * @param to       종료 시각(포함, null 허용)
     * @param pageable 페이지 정보
     * @return 이벤트 페이지
     */
    @Query("""
            SELECT e FROM AccessEvent e
            WHERE (:type IS NULL OR e.eventType = :type)
              AND (:from IS NULL OR e.occurredAt >= :from)
              AND (:to   IS NULL OR e.occurredAt <= :to)
            """)
    Page<AccessEvent> search(@Param("type") EventType type,
                             @Param("from") LocalDateTime from,
                             @Param("to") LocalDateTime to,
                             Pageable pageable);

    /**
     * 중복 이벤트 탐지: 동일 번호판 + 동일 게이트 + 동일 타입이면서
     * {@code since} 이후에 발생한 가장 최근 이벤트를 찾는다.
     *
     * <p>{@code RecognitionService} 가 30초 창으로 호출해, 존재하면 새 이벤트를
     * 만들지 않고 기존 이벤트를 재사용한다.
     *
     * @param plateNumber 번호판
     * @param gateName    게이트 이름(null 허용 — null 끼리도 동일 게이트로 본다)
     * @param eventType   이벤트 타입
     * @param since       이 시각 이후(포함)
     * @param pageable    크기 1 의 페이지 정보를 넘겨 최신 1건만 받는다
     * @return 가장 최근 중복 이벤트 목록(없으면 빈 리스트)
     */
    @Query("""
            SELECT e FROM AccessEvent e
            WHERE e.plateNumber = :plateNumber
              AND e.eventType = :eventType
              AND ((:gateName IS NULL AND e.gateName IS NULL) OR (e.gateName = :gateName))
              AND e.occurredAt >= :since
            ORDER BY e.occurredAt DESC
            """)
    List<AccessEvent> findRecentDuplicates(@Param("plateNumber") String plateNumber,
                                           @Param("gateName") String gateName,
                                           @Param("eventType") EventType eventType,
                                           @Param("since") LocalDateTime since,
                                           Pageable pageable);

    /**
     * 기간 내 타입별 건수.
     *
     * @param type 이벤트 타입
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 건수
     */
    @Query("""
            SELECT COUNT(e) FROM AccessEvent e
            WHERE e.eventType = :type AND e.occurredAt >= :from AND e.occurredAt < :to
            """)
    long countByTypeInRange(@Param("type") EventType type,
                            @Param("from") LocalDateTime from,
                            @Param("to") LocalDateTime to);

    /**
     * 최근 이벤트 N건(대시보드 {@code recentEvents}).
     *
     * @param pageable 크기 제한용 페이지 정보(예: PageRequest.of(0, 10))
     * @return 발생 시각 내림차순 목록
     */
    @Query("SELECT e FROM AccessEvent e ORDER BY e.occurredAt DESC")
    List<AccessEvent> findRecent(Pageable pageable);

    /**
     * 기간 내 시간대별 입/출차 건수 집계(네이티브 쿼리).
     * 각 행은 {@code [hour(int), entries(long), exits(long)]} 이다.
     *
     * <p>{@code EXTRACT(HOUR ...)} 는 numeric 을 돌려주므로 int 로 캐스팅한다.
     *
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 시간대 집계 행
     */
    @Query(value = """
            SELECT CAST(EXTRACT(HOUR FROM occurred_at) AS int) AS hour,
                   COUNT(*) FILTER (WHERE event_type = 'ENTRY') AS entries,
                   COUNT(*) FILTER (WHERE event_type = 'EXIT')  AS exits
            FROM access_event
            WHERE occurred_at >= :from AND occurred_at < :to
            GROUP BY 1
            ORDER BY 1
            """, nativeQuery = true)
    List<Object[]> countHourlyTraffic(@Param("from") LocalDateTime from, @Param("to") LocalDateTime to);
}
