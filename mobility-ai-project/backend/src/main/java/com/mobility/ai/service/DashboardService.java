package com.mobility.ai.service;

import com.mobility.ai.domain.EventType;
import com.mobility.ai.domain.RecognitionStatus;
import com.mobility.ai.dto.DashboardDtos.DashboardSummaryResponse;
import com.mobility.ai.dto.EventDtos.RecentEvent;
import com.mobility.ai.dto.ParkingDtos.HourlyTraffic;
import com.mobility.ai.repository.ParkingSessionRepository;
import com.mobility.ai.repository.PlateRecognitionRepository;
import com.mobility.ai.repository.VehicleRepository;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 대시보드 집계 서비스. 계약서 4장 {@code GET /api/v1/dashboard/summary} 스키마를
 * 필드 하나까지 그대로 채운다(교재 6장 "대시보드 집계" 절 인용 대상).
 *
 * <p>모든 "오늘" 기준 지표는 서버 로컬 날짜의 00:00(포함) ~ 다음날 00:00(미포함)이다.
 * 집계는 리포지토리의 JPQL/네이티브 쿼리로 수행하고, 이 클래스는 조립만 한다.
 */
@Service
public class DashboardService {

    /** 최근 이벤트 노출 건수 */
    private static final int RECENT_EVENT_LIMIT = 10;

    private final VehicleRepository vehicleRepository;
    private final PlateRecognitionRepository plateRecognitionRepository;
    private final ParkingSessionRepository parkingSessionRepository;
    private final AccessEventService accessEventService;
    private final PdmService pdmService;

    /**
     * @param vehicleRepository          차량 리포지토리
     * @param plateRecognitionRepository 인식 리포지토리
     * @param parkingSessionRepository   주차 세션 리포지토리
     * @param accessEventService         이벤트 집계
     * @param pdmService                 위험도 집계
     */
    public DashboardService(VehicleRepository vehicleRepository,
                            PlateRecognitionRepository plateRecognitionRepository,
                            ParkingSessionRepository parkingSessionRepository,
                            AccessEventService accessEventService,
                            PdmService pdmService) {
        this.vehicleRepository = vehicleRepository;
        this.plateRecognitionRepository = plateRecognitionRepository;
        this.parkingSessionRepository = parkingSessionRepository;
        this.accessEventService = accessEventService;
        this.pdmService = pdmService;
    }

    /**
     * 대시보드 요약을 만든다.
     *
     * @return 요약 응답
     */
    @Transactional(readOnly = true)
    public DashboardSummaryResponse summary() {
        LocalDateTime dayStart = LocalDate.now().atStartOfDay();
        LocalDateTime dayEnd = dayStart.plusDays(1);

        long totalVehicles = vehicleRepository.count();
        long todayEntries = accessEventService.countInRange(EventType.ENTRY, dayStart, dayEnd);
        long todayExits = accessEventService.countInRange(EventType.EXIT, dayStart, dayEnd);
        long currentlyParked = parkingSessionRepository.countActive();

        // 인식률 = 오늘 SUCCESS / 오늘 전체. 오늘 인식이 없으면 0.0 으로 둔다.
        long totalRecognitions = plateRecognitionRepository.countInRange(dayStart, dayEnd);
        long successCount = plateRecognitionRepository.countByStatusInRange(
                RecognitionStatus.SUCCESS, dayStart, dayEnd);
        double recognitionRate = totalRecognitions == 0
                ? 0.0
                : round3((double) successCount / (double) totalRecognitions);

        long failureCount = plateRecognitionRepository.countFailuresInRange(dayStart, dayEnd);
        long unpaidCount = parkingSessionRepository.countUnpaid();

        Long revenue = parkingSessionRepository.sumPaidFeeInRange(dayStart, dayEnd);
        long todayRevenue = revenue == null ? 0L : revenue;

        long riskyVehicles = pdmService.countRiskyVehicles();

        Double avgLatency = plateRecognitionRepository.avgTotalMsInRange(dayStart, dayEnd);
        double avgLatencyMs = avgLatency == null ? 0.0 : round1(avgLatency);

        List<RecentEvent> recentEvents = accessEventService.recent(RECENT_EVENT_LIMIT);
        List<HourlyTraffic> hourlyTraffic = accessEventService.hourlyTraffic(dayStart, dayEnd);
        Map<String, Long> riskDistribution = pdmService.riskDistribution();

        return new DashboardSummaryResponse(
                totalVehicles,
                todayEntries,
                todayExits,
                currentlyParked,
                recognitionRate,
                failureCount,
                unpaidCount,
                todayRevenue,
                riskyVehicles,
                avgLatencyMs,
                recentEvents,
                hourlyTraffic,
                riskDistribution
        );
    }

    /**
     * 소수 첫째 자리 반올림(지연 ms 표기용).
     *
     * @param value 원본
     * @return 반올림 값
     */
    private double round1(double value) {
        return Math.round(value * 10.0) / 10.0;
    }

    /**
     * 소수 셋째 자리 반올림(비율 표기용).
     *
     * @param value 원본
     * @return 반올림 값
     */
    private double round3(double value) {
        return Math.round(value * 1000.0) / 1000.0;
    }
}
