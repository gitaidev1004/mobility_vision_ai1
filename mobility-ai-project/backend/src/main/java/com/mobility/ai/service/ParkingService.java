package com.mobility.ai.service;

import com.mobility.ai.common.BusinessException;
import com.mobility.ai.common.NotFoundException;
import com.mobility.ai.common.PageResponse;
import com.mobility.ai.config.AppProperties;
import com.mobility.ai.domain.AccessEvent;
import com.mobility.ai.domain.ParkingPolicy;
import com.mobility.ai.domain.ParkingSession;
import com.mobility.ai.domain.Vehicle;
import com.mobility.ai.dto.ParkingDtos.HourlyTraffic;
import com.mobility.ai.dto.ParkingDtos.ParkingSessionResponse;
import com.mobility.ai.dto.ParkingDtos.ParkingStatusResponse;
import com.mobility.ai.dto.ParkingDtos.PayRequest;
import com.mobility.ai.repository.ParkingPolicyRepository;
import com.mobility.ai.repository.ParkingSessionRepository;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 주차 세션/요금 서비스. 요금 계산은 순수 함수인
 * {@link ParkingFeeCalculator} 에 위임하고, 이 클래스는 DB 상태 전이만 담당한다
 * (교재 6장 "주차 요금 처리" 절 인용 대상).
 */
@Service
public class ParkingService {

    private static final Logger log = LoggerFactory.getLogger(ParkingService.class);

    private final ParkingSessionRepository parkingSessionRepository;
    private final ParkingPolicyRepository parkingPolicyRepository;
    private final AccessEventService accessEventService;
    private final AppProperties appProperties;

    /**
     * @param parkingSessionRepository 주차 세션 리포지토리
     * @param parkingPolicyRepository  요금 정책 리포지토리
     * @param accessEventService       시간대별 통행량 집계용
     * @param appProperties            {@code app.*} 설정(수용 대수)
     */
    public ParkingService(ParkingSessionRepository parkingSessionRepository,
                          ParkingPolicyRepository parkingPolicyRepository,
                          AccessEventService accessEventService,
                          AppProperties appProperties) {
        this.parkingSessionRepository = parkingSessionRepository;
        this.parkingPolicyRepository = parkingPolicyRepository;
        this.accessEventService = accessEventService;
        this.appProperties = appProperties;
    }

    // =================================================================
    // 세션 생성/종료 — RecognitionService 가 호출한다
    // =================================================================

    /**
     * 입차 세션을 만든다. 같은 번호판으로 열린 세션이 이미 있으면
     * 새로 만들지 않고 기존 세션을 그대로 돌려준다(중복 입차 방어).
     *
     * @param vehicle    매칭된 차량(미등록이면 null)
     * @param plate      번호판
     * @param entryEvent 입차 이벤트
     * @return 생성되었거나 재사용된 세션
     */
    @Transactional
    public ParkingSession openSession(Vehicle vehicle, String plate, AccessEvent entryEvent) {
        List<ParkingSession> open = parkingSessionRepository
                .findOpenByPlateNumber(plate, PageRequest.of(0, 1));
        if (!open.isEmpty()) {
            log.debug("이미 열린 주차 세션이 있어 재사용합니다: plate={}, sessionId={}",
                    plate, open.get(0).getId());
            return open.get(0);
        }

        LocalDateTime entryAt = entryEvent == null || entryEvent.getOccurredAt() == null
                ? LocalDateTime.now()
                : entryEvent.getOccurredAt();

        ParkingSession session = new ParkingSession(plate, entryAt);
        session.setVehicle(vehicle);
        session.setEntryEventId(entryEvent == null ? null : entryEvent.getId());
        session.setDiscountAmount(0L);
        session.setPaid(false);
        return parkingSessionRepository.save(session);
    }

    /**
     * 출차 처리: 열린 세션을 찾아 종료하고 요금을 계산한다.
     *
     * @param vehicle   매칭된 차량(미등록이면 null)
     * @param plate     번호판
     * @param exitEvent 출차 이벤트
     * @return 종료된 세션. 열린 세션이 없으면 null(출차만 찍힌 비정상 흐름)
     */
    @Transactional
    public ParkingSession closeSession(Vehicle vehicle, String plate, AccessEvent exitEvent) {
        List<ParkingSession> open = parkingSessionRepository
                .findOpenByPlateNumber(plate, PageRequest.of(0, 1));
        if (open.isEmpty()) {
            log.warn("열린 주차 세션 없이 출차 이벤트가 발생했습니다: plate={}", plate);
            return null;
        }

        ParkingSession session = open.get(0);
        LocalDateTime exitAt = exitEvent == null || exitEvent.getOccurredAt() == null
                ? LocalDateTime.now()
                : exitEvent.getOccurredAt();

        // 시계 역전(출차가 입차보다 이름) 상황에서도 음수가 되지 않게 한다.
        long minutes = Duration.between(session.getEntryAt(), exitAt).toMinutes();
        if (minutes < 0) {
            minutes = 0;
        }

        ParkingPolicy policy = activePolicy();
        ParkingFeeCalculator.FeeResult fee = ParkingFeeCalculator.calculate(
                policy,
                minutes,
                vehicle == null ? null : vehicle.getPassType(),
                vehicle != null && vehicle.hasValidFreePass(exitAt),
                vehicle != null && vehicle.hasValidVisitorPass(exitAt)
        );

        session.setExitAt(exitAt);
        session.setExitEventId(exitEvent == null ? null : exitEvent.getId());
        session.setParkedMinutes((int) minutes);
        session.setFeeAmount(fee.payableFee());
        session.setDiscountAmount(fee.discount());

        // 요금이 0원이면 결제 대상이 아니므로 결제 완료로 마감한다.
        if (fee.payableFee() == 0L) {
            session.setPaid(true);
            session.setPaymentMethod(fee.freeReason() == null ? "FREE" : fee.freeReason());
        }

        log.info("출차 처리: plate={}, 주차 {}분, 요금 {}원(할인 {}원)",
                plate, minutes, fee.payableFee(), fee.discount());
        return parkingSessionRepository.save(session);
    }

    // =================================================================
    // 조회/결제 API
    // =================================================================

    /**
     * 주차 세션 목록.
     *
     * @param active true=주차 중만, false=완료만, null=전체
     * @param page   페이지 번호
     * @param size   페이지 크기
     * @return 세션 페이지 응답
     */
    @Transactional(readOnly = true)
    public PageResponse<ParkingSessionResponse> list(Boolean active, int page, int size) {
        PageRequest pageable = PageRequest.of(
                Math.max(0, page),
                normalizeSize(size),
                Sort.by(Sort.Direction.DESC, "entryAt").and(Sort.by(Sort.Direction.DESC, "id"))
        );

        Page<ParkingSession> result;
        if (active == null) {
            result = parkingSessionRepository.findAll(pageable);
        } else if (active) {
            result = parkingSessionRepository.findActiveSessions(pageable);
        } else {
            result = parkingSessionRepository.findClosedSessions(pageable);
        }
        return PageResponse.from(result, ParkingSessionResponse::from);
    }

    /**
     * 주차 요금 결제.
     *
     * @param sessionId 세션 PK
     * @param request   결제 수단과 추가 할인
     * @return 결제 반영된 세션
     * @throws NotFoundException 세션이 없을 때
     * @throws BusinessException 아직 출차하지 않았거나 이미 결제된 세션일 때
     */
    @Transactional
    public ParkingSessionResponse pay(Long sessionId, PayRequest request) {
        ParkingSession session = parkingSessionRepository.findById(sessionId)
                .orElseThrow(() -> NotFoundException.of("주차 세션", sessionId));

        if (session.getExitAt() == null) {
            throw new BusinessException("아직 출차하지 않은 세션은 결제할 수 없습니다.");
        }
        if (session.isPaid()) {
            throw new BusinessException("이미 결제된 세션입니다.");
        }

        long extraDiscount = request.discountAmount() == null ? 0L : request.discountAmount();
        long currentFee = session.getFeeAmount() == null ? 0L : session.getFeeAmount();
        if (extraDiscount > currentFee) {
            throw new BusinessException(
                    "할인 금액이 청구 금액(" + currentFee + "원)보다 클 수 없습니다.");
        }

        session.setDiscountAmount(session.getDiscountAmount() + extraDiscount);
        session.setFeeAmount(currentFee - extraDiscount);
        session.setPaid(true);
        session.setPaymentMethod(request.paymentMethod());

        log.info("결제 완료: sessionId={}, 수단={}, 최종 {}원",
                sessionId, request.paymentMethod(), session.getFeeAmount());
        return ParkingSessionResponse.from(parkingSessionRepository.save(session));
    }

    /**
     * 주차 현황. 계약서 {@code GET /api/v1/parking/status} 스키마와 1:1 이다.
     *
     * @return 현황 응답
     */
    @Transactional(readOnly = true)
    public ParkingStatusResponse status() {
        int capacity = appProperties.getParking().getCapacity();
        long occupied = parkingSessionRepository.countActive();
        long available = Math.max(0L, capacity - occupied);
        double occupancyRate = capacity <= 0
                ? 0.0
                : round3((double) occupied / (double) capacity);
        long unpaidCount = parkingSessionRepository.countUnpaid();

        LocalDateTime from = LocalDate.now().atStartOfDay();
        List<HourlyTraffic> byHour = accessEventService.hourlyTraffic(from, from.plusDays(1));

        return new ParkingStatusResponse(capacity, occupied, available, occupancyRate, unpaidCount, byHour);
    }

    /**
     * 활성 요금 정책 1건을 읽는다.
     *
     * @return 활성 정책. 없으면 null(요금 계산기가 0원 처리)
     */
    @Transactional(readOnly = true)
    public ParkingPolicy activePolicy() {
        List<ParkingPolicy> active = parkingPolicyRepository.findActive(PageRequest.of(0, 1));
        return active.isEmpty() ? null : active.get(0);
    }

    /**
     * 소수 셋째 자리까지 반올림한다(점유율 표기용).
     *
     * @param value 원본 값
     * @return 반올림 값
     */
    private double round3(double value) {
        return Math.round(value * 1000.0) / 1000.0;
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
