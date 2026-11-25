package com.mobility.ai.dto;

import com.mobility.ai.domain.ParkingSession;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.PositiveOrZero;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 주차 관련 DTO 모음. 계약서 4장 {@code /api/v1/parking/**}.
 */
public final class ParkingDtos {

    private ParkingDtos() {
    }

    /**
     * 주차 세션 1건.
     *
     * @param id              PK
     * @param vehicleId       차량 PK
     * @param plateNumber     번호판
     * @param entryEventId    입차 이벤트 PK
     * @param exitEventId     출차 이벤트 PK
     * @param entryAt         입차 시각
     * @param exitAt          출차 시각(주차 중이면 null)
     * @param parkedMinutes   주차 시간(분)
     * @param feeAmount       청구 금액
     * @param discountAmount  할인 금액
     * @param paid            결제 여부
     * @param paymentMethod   결제 수단
     * @param active          주차 중 여부
     */
    public record ParkingSessionResponse(
            Long id,
            Long vehicleId,
            String plateNumber,
            Long entryEventId,
            Long exitEventId,
            LocalDateTime entryAt,
            LocalDateTime exitAt,
            Integer parkedMinutes,
            Long feeAmount,
            long discountAmount,
            boolean paid,
            String paymentMethod,
            boolean active
    ) {
        /**
         * 엔티티 → DTO 변환.
         *
         * @param s 세션 엔티티
         * @return DTO
         */
        public static ParkingSessionResponse from(ParkingSession s) {
            return new ParkingSessionResponse(
                    s.getId(),
                    s.getVehicle() == null ? null : s.getVehicle().getId(),
                    s.getPlateNumber(),
                    s.getEntryEventId(),
                    s.getExitEventId(),
                    s.getEntryAt(),
                    s.getExitAt(),
                    s.getParkedMinutes(),
                    s.getFeeAmount(),
                    s.getDiscountAmount(),
                    s.isPaid(),
                    s.getPaymentMethod(),
                    s.isActive()
            );
        }
    }

    /**
     * 결제 요청. 계약서 {@code {"paymentMethod":"CARD","discountAmount":0}}.
     *
     * @param paymentMethod  결제 수단(CARD|CASH|PASS|MOBILE)
     * @param discountAmount 추가 할인 금액(0 이상)
     */
    public record PayRequest(
            @NotBlank(message = "결제 수단은 필수입니다.") String paymentMethod,
            @PositiveOrZero(message = "할인 금액은 0 이상이어야 합니다.") Long discountAmount
    ) {
    }

    /**
     * 시간대별 통행량 1건. 계약서 {@code byHour}/{@code hourlyTraffic} 공용.
     *
     * @param hour    0~23 시
     * @param entries 입차 건수
     * @param exits   출차 건수
     */
    public record HourlyTraffic(
            int hour,
            long entries,
            long exits
    ) {
    }

    /**
     * 주차 현황. 계약서 {@code GET /api/v1/parking/status} 와 1:1.
     *
     * @param capacity       총 수용 대수
     * @param occupied       현재 주차 대수
     * @param available      남은 자리
     * @param occupancyRate  점유율(0.0 ~ 1.0)
     * @param unpaidCount    미결제 세션 수
     * @param byHour         시간대별 통행량(0~23 전부)
     */
    public record ParkingStatusResponse(
            int capacity,
            long occupied,
            long available,
            double occupancyRate,
            long unpaidCount,
            List<HourlyTraffic> byHour
    ) {
    }
}
