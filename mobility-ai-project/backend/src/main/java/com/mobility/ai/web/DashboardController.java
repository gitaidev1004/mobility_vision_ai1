package com.mobility.ai.web;

import com.mobility.ai.dto.DashboardDtos.DashboardSummaryResponse;
import com.mobility.ai.service.DashboardService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 대시보드 컨트롤러.
 *
 * <pre>
 * GET /api/v1/dashboard/summary
 * </pre>
 *
 * <p>계약서 4장대로 이 경로는 permitAll 이 아니다 — 로그인(JWT)이 필요하다.
 */
@RestController
@RequestMapping("/api/v1/dashboard")
public class DashboardController {

    private final DashboardService dashboardService;

    /**
     * @param dashboardService 대시보드 서비스
     */
    public DashboardController(DashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    /**
     * 대시보드 요약 지표.
     *
     * @return 요약 응답
     */
    @GetMapping("/summary")
    public DashboardSummaryResponse summary() {
        return dashboardService.summary();
    }
}
