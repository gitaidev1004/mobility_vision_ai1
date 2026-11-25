package com.mobility.ai.dto;

import com.mobility.ai.dto.EventDtos.RecentEvent;
import com.mobility.ai.dto.ParkingDtos.HourlyTraffic;
import java.util.List;
import java.util.Map;

/**
 * 대시보드 DTO. 계약서 4장 {@code GET /api/v1/dashboard/summary} 와 1:1 이다.
 */
public final class DashboardDtos {

    private DashboardDtos() {
    }

    /**
     * 대시보드 요약.
     *
     * <pre>
     * {"totalVehicles","todayEntries","todayExits","currentlyParked",
     *  "recognitionRate","failureCount","unpaidCount","todayRevenue",
     *  "riskyVehicles","avgLatencyMs","recentEvents","hourlyTraffic","riskDistribution"}
     * </pre>
     *
     * @param totalVehicles    등록 차량 수
     * @param todayEntries     오늘 입차 건수
     * @param todayExits       오늘 출차 건수
     * @param currentlyParked  현재 주차 대수
     * @param recognitionRate  오늘 인식 성공률(0.0 ~ 1.0)
     * @param failureCount     오늘 인식 실패 건수
     * @param unpaidCount      미결제 세션 수
     * @param todayRevenue     오늘 결제 매출(원)
     * @param riskyVehicles    HIGH/CRITICAL 차량 수
     * @param avgLatencyMs     오늘 평균 인식 지연(ms)
     * @param recentEvents     최근 출입 이벤트 10건
     * @param hourlyTraffic    오늘 시간대별 통행량(0~23)
     * @param riskDistribution 위험 등급 분포
     */
    public record DashboardSummaryResponse(
            long totalVehicles,
            long todayEntries,
            long todayExits,
            long currentlyParked,
            double recognitionRate,
            long failureCount,
            long unpaidCount,
            long todayRevenue,
            long riskyVehicles,
            double avgLatencyMs,
            List<RecentEvent> recentEvents,
            List<HourlyTraffic> hourlyTraffic,
            Map<String, Long> riskDistribution
    ) {
    }
}
