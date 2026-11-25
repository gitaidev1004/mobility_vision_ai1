package com.mobility.ai.web;

import com.mobility.ai.common.PageResponse;
import com.mobility.ai.dto.ParkingDtos.ParkingSessionResponse;
import com.mobility.ai.dto.ParkingDtos.ParkingStatusResponse;
import com.mobility.ai.dto.ParkingDtos.PayRequest;
import com.mobility.ai.service.ParkingService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 주차 컨트롤러.
 *
 * <pre>
 * GET  /api/v1/parking/sessions?active=true&page=&size=
 * POST /api/v1/parking/sessions/{id}/pay   {"paymentMethod":"CARD","discountAmount":0}
 * GET  /api/v1/parking/status
 * </pre>
 */
@RestController
@RequestMapping("/api/v1/parking")
public class ParkingController {

    private final ParkingService parkingService;

    /**
     * @param parkingService 주차 서비스
     */
    public ParkingController(ParkingService parkingService) {
        this.parkingService = parkingService;
    }

    /**
     * 주차 세션 목록.
     *
     * @param active true=주차 중만, false=완료만, 생략=전체
     * @param page   페이지 번호
     * @param size   페이지 크기
     * @return 세션 페이지
     */
    @GetMapping("/sessions")
    public PageResponse<ParkingSessionResponse> sessions(
            @RequestParam(name = "active", required = false) Boolean active,
            @RequestParam(name = "page", defaultValue = "0") int page,
            @RequestParam(name = "size", defaultValue = "20") int size) {
        return parkingService.list(active, page, size);
    }

    /**
     * 주차 요금 결제.
     *
     * @param id      세션 PK
     * @param request 결제 수단과 추가 할인
     * @return 결제 반영된 세션
     */
    @PostMapping("/sessions/{id}/pay")
    public ParkingSessionResponse pay(@PathVariable("id") Long id,
                                      @Valid @RequestBody PayRequest request) {
        return parkingService.pay(id, request);
    }

    /**
     * 주차 현황(점유율/미결제/시간대별 통행량).
     *
     * @return 현황 응답
     */
    @GetMapping("/status")
    public ParkingStatusResponse status() {
        return parkingService.status();
    }
}
