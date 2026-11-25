package com.mobility.ai.web;

import com.mobility.ai.common.PageResponse;
import com.mobility.ai.dto.EventDtos.EventResponse;
import com.mobility.ai.service.AccessEventService;
import java.time.LocalDateTime;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 출입 이벤트 컨트롤러.
 *
 * <pre>
 * GET /api/v1/events?type=&from=&to=&page=&size=
 * </pre>
 */
@RestController
@RequestMapping("/api/v1/events")
public class EventController {

    private final AccessEventService accessEventService;

    /**
     * @param accessEventService 이벤트 서비스
     */
    public EventController(AccessEventService accessEventService) {
        this.accessEventService = accessEventService;
    }

    /**
     * 출입 이벤트 조회.
     *
     * @param type ENTRY|EXIT (선택)
     * @param from 시작 시각(ISO-8601, 선택)
     * @param to   종료 시각(ISO-8601, 선택)
     * @param page 페이지 번호
     * @param size 페이지 크기
     * @return 이벤트 페이지
     */
    @GetMapping
    public PageResponse<EventResponse> list(
            @RequestParam(name = "type", required = false) String type,
            @RequestParam(name = "from", required = false)
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime from,
            @RequestParam(name = "to", required = false)
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime to,
            @RequestParam(name = "page", defaultValue = "0") int page,
            @RequestParam(name = "size", defaultValue = "20") int size) {
        return accessEventService.search(type, from, to, page, size);
    }
}
