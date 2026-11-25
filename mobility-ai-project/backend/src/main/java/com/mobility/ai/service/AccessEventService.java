package com.mobility.ai.service;

import com.mobility.ai.common.PageResponse;
import com.mobility.ai.config.AppProperties;
import com.mobility.ai.domain.AccessEvent;
import com.mobility.ai.domain.EventType;
import com.mobility.ai.domain.PlateRecognition;
import com.mobility.ai.domain.Vehicle;
import com.mobility.ai.dto.EventDtos.EventResponse;
import com.mobility.ai.dto.EventDtos.RecentEvent;
import com.mobility.ai.dto.ParkingDtos.HourlyTraffic;
import com.mobility.ai.repository.AccessEventRepository;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 출입 이벤트 서비스.
 *
 * <h2>중복 이벤트 제거</h2>
 * 차단기 카메라는 한 대의 차량에 대해 짧은 간격으로 여러 프레임을 올리는 일이 흔하다.
 * 그래서 <b>동일 번호판 + 동일 게이트 + 동일 타입</b> 이벤트가
 * {@code app.recognition.dedup-window-seconds}(기본 30초) 이내에 이미 있으면
 * 새 이벤트를 만들지 않고 <b>기존 이벤트를 그대로 돌려준다</b>.
 * 덕분에 주차 세션이 중복 생성되거나 이중 과금되는 일이 없다.
 */
@Service
public class AccessEventService {

    private static final Logger log = LoggerFactory.getLogger(AccessEventService.class);

    private final AccessEventRepository accessEventRepository;
    private final AppProperties appProperties;

    /**
     * @param accessEventRepository 이벤트 리포지토리
     * @param appProperties         {@code app.*} 설정(중복 창)
     */
    public AccessEventService(AccessEventRepository accessEventRepository,
                              AppProperties appProperties) {
        this.accessEventRepository = accessEventRepository;
        this.appProperties = appProperties;
    }

    /**
     * 이벤트 생성 결과.
     *
     * @param event      생성되었거나 재사용된 이벤트
     * @param duplicated 기존 이벤트를 재사용했으면 true
     */
    public record EventCreation(AccessEvent event, boolean duplicated) {
    }

    /**
     * 출입 이벤트를 기록한다. 중복 창 안의 동일 이벤트가 있으면 재사용한다.
     *
     * @param eventType    ENTRY|EXIT
     * @param plate        번호판(null 이면 중복 판정을 건너뛴다)
     * @param gateName     게이트 이름
     * @param vehicle      매칭 차량(미등록이면 null)
     * @param recognition  유발한 인식 기록(없으면 null)
     * @param confidence   신뢰도
     * @param allowed      통행 허용 여부
     * @param rejectReason 거부 사유(허용이면 null)
     * @return 생성/재사용 결과
     */
    @Transactional
    public EventCreation record(EventType eventType,
                                String plate,
                                String gateName,
                                Vehicle vehicle,
                                PlateRecognition recognition,
                                Double confidence,
                                boolean allowed,
                                String rejectReason) {
        LocalDateTime now = LocalDateTime.now();

        // 번호판을 못 읽은 건은 서로 구분할 수 없으므로 중복 제거를 적용하지 않는다.
        if (plate != null && !plate.isBlank()) {
            int window = appProperties.getRecognition().getDedupWindowSeconds();
            LocalDateTime since = now.minusSeconds(Math.max(0, window));
            List<AccessEvent> duplicates = accessEventRepository.findRecentDuplicates(
                    plate, gateName, eventType, since, PageRequest.of(0, 1));
            if (!duplicates.isEmpty()) {
                AccessEvent existing = duplicates.get(0);
                log.info("중복 출입 이벤트 감지({}초 이내) — 기존 이벤트 재사용: id={}, plate={}, gate={}, type={}",
                        window, existing.getId(), plate, gateName, eventType);
                return new EventCreation(existing, true);
            }
        }

        AccessEvent event = new AccessEvent(eventType, plate, gateName);
        event.setVehicle(vehicle);
        event.setRecognition(recognition);
        event.setConfidence(confidence);
        event.setAllowed(allowed);
        event.setRejectReason(rejectReason);
        event.setOccurredAt(now);
        return new EventCreation(accessEventRepository.save(event), false);
    }

    /**
     * 출입 이벤트 검색.
     *
     * @param type ENTRY|EXIT 문자열(null/빈값이면 전체)
     * @param from 시작 시각(null 허용)
     * @param to   종료 시각(null 허용)
     * @param page 페이지 번호
     * @param size 페이지 크기
     * @return 이벤트 페이지 응답
     */
    @Transactional(readOnly = true)
    public PageResponse<EventResponse> search(String type, LocalDateTime from, LocalDateTime to,
                                              int page, int size) {
        EventType eventType = parseEventType(type);
        PageRequest pageable = PageRequest.of(
                Math.max(0, page),
                normalizeSize(size),
                Sort.by(Sort.Direction.DESC, "occurredAt").and(Sort.by(Sort.Direction.DESC, "id"))
        );
        Page<AccessEvent> result = accessEventRepository.search(eventType, from, to, pageable);
        return PageResponse.from(result, EventResponse::from);
    }

    /**
     * 최근 이벤트 N건(대시보드용).
     *
     * @param limit 최대 건수
     * @return 축약 이벤트 목록
     */
    @Transactional(readOnly = true)
    public List<RecentEvent> recent(int limit) {
        List<AccessEvent> events = accessEventRepository.findRecent(
                PageRequest.of(0, normalizeSize(limit)));
        return events.stream().map(RecentEvent::from).toList();
    }

    /**
     * 기간 내 타입별 건수.
     *
     * @param type 이벤트 타입
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 건수
     */
    @Transactional(readOnly = true)
    public long countInRange(EventType type, LocalDateTime from, LocalDateTime to) {
        return accessEventRepository.countByTypeInRange(type, from, to);
    }

    /**
     * 시간대별 통행량. DB 에 없는 시간대도 0 으로 채워 0~23 전부를 돌려준다
     * (프런트엔드 SVG 차트가 빈 구간을 다루지 않아도 되게 하기 위함).
     *
     * @param from 시작 시각(포함)
     * @param to   종료 시각(미포함)
     * @return 24개 구간 목록
     */
    @Transactional(readOnly = true)
    public List<HourlyTraffic> hourlyTraffic(LocalDateTime from, LocalDateTime to) {
        long[] entries = new long[24];
        long[] exits = new long[24];

        for (Object[] row : accessEventRepository.countHourlyTraffic(from, to)) {
            Integer hour = toInt(row[0]);
            if (hour == null || hour < 0 || hour > 23) {
                continue;
            }
            entries[hour] = toLong(row[1]);
            exits[hour] = toLong(row[2]);
        }

        List<HourlyTraffic> list = new ArrayList<>(24);
        for (int h = 0; h < 24; h++) {
            list.add(new HourlyTraffic(h, entries[h], exits[h]));
        }
        return list;
    }

    /**
     * 문자열을 이벤트 타입으로 변환한다.
     *
     * @param raw 입력 문자열
     * @return 타입, 비었거나 알 수 없으면 null(= 전체 조회)
     */
    private EventType parseEventType(String raw) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        for (EventType type : EventType.values()) {
            if (type.name().equalsIgnoreCase(raw.trim())) {
                return type;
            }
        }
        return null;
    }

    /**
     * 네이티브 쿼리 결과를 Integer 로 변환한다.
     *
     * @param value 원본 값
     * @return Integer 또는 null
     */
    private Integer toInt(Object value) {
        return value instanceof Number n ? n.intValue() : null;
    }

    /**
     * 네이티브 쿼리 결과를 long 으로 변환한다.
     *
     * @param value 원본 값
     * @return long 값(변환 불가면 0)
     */
    private long toLong(Object value) {
        return value instanceof Number n ? n.longValue() : 0L;
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
