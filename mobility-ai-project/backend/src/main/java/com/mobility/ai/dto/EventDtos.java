package com.mobility.ai.dto;

import com.mobility.ai.domain.AccessEvent;
import java.time.LocalDateTime;

/**
 * 출입 이벤트 DTO 모음. 계약서 4장 {@code GET /api/v1/events}.
 */
public final class EventDtos {

    private EventDtos() {
    }

    /**
     * 출입 이벤트 1건.
     *
     * @param id            PK
     * @param vehicleId     차량 PK(미등록이면 null)
     * @param plateNumber   번호판
     * @param eventType     ENTRY|EXIT
     * @param gateName      게이트 이름
     * @param occurredAt    발생 시각
     * @param recognitionId 인식 기록 PK
     * @param confidence    신뢰도
     * @param allowed       통행 허용 여부
     * @param rejectReason  거부 사유
     */
    public record EventResponse(
            Long id,
            Long vehicleId,
            String plateNumber,
            String eventType,
            String gateName,
            LocalDateTime occurredAt,
            Long recognitionId,
            Double confidence,
            boolean allowed,
            String rejectReason
    ) {
        /**
         * 엔티티 → DTO 변환. LAZY 연관은 식별자만 읽는다.
         *
         * @param e 이벤트 엔티티
         * @return DTO
         */
        public static EventResponse from(AccessEvent e) {
            return new EventResponse(
                    e.getId(),
                    e.getVehicle() == null ? null : e.getVehicle().getId(),
                    e.getPlateNumber(),
                    e.getEventType() == null ? null : e.getEventType().name(),
                    e.getGateName(),
                    e.getOccurredAt(),
                    e.getRecognition() == null ? null : e.getRecognition().getId(),
                    e.getConfidence(),
                    e.isAllowed(),
                    e.getRejectReason()
            );
        }
    }

    /**
     * 대시보드 최근 이벤트 축약형.
     * 계약서 {@code recentEvents:[{id,plateNumber,eventType,gateName,occurredAt,allowed}]} 와 1:1.
     *
     * @param id          PK
     * @param plateNumber 번호판
     * @param eventType   ENTRY|EXIT
     * @param gateName    게이트 이름
     * @param occurredAt  발생 시각
     * @param allowed     허용 여부
     */
    public record RecentEvent(
            Long id,
            String plateNumber,
            String eventType,
            String gateName,
            LocalDateTime occurredAt,
            boolean allowed
    ) {
        /**
         * 엔티티 → 축약 DTO 변환.
         *
         * @param e 이벤트 엔티티
         * @return 축약 DTO
         */
        public static RecentEvent from(AccessEvent e) {
            return new RecentEvent(
                    e.getId(),
                    e.getPlateNumber(),
                    e.getEventType() == null ? null : e.getEventType().name(),
                    e.getGateName(),
                    e.getOccurredAt(),
                    e.isAllowed()
            );
        }
    }
}
