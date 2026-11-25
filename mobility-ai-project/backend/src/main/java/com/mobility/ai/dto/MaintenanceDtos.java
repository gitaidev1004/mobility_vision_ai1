package com.mobility.ai.dto;

import com.mobility.ai.domain.MaintenanceRecord;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;
import jakarta.validation.constraints.Size;
import java.time.LocalDate;

/**
 * 정비 이력 DTO 모음. 계약서 4장 {@code /api/v1/maintenance}.
 */
public final class MaintenanceDtos {

    private MaintenanceDtos() {
    }

    /**
     * 정비 등록 요청.
     *
     * @param vehicleId   차량 PK(필수)
     * @param serviceDate 정비일(필수)
     * @param odometerKm  주행거리
     * @param item        정비 항목(필수)
     * @param partName    부품명
     * @param cost        비용
     * @param shopName    정비소
     * @param memo        메모
     */
    public record MaintenanceRequest(
            @NotNull(message = "차량 ID는 필수입니다.") Long vehicleId,
            @NotNull(message = "정비일은 필수입니다.") LocalDate serviceDate,
            @PositiveOrZero(message = "주행거리는 0 이상이어야 합니다.") Integer odometerKm,
            @NotBlank(message = "정비 항목은 필수입니다.") @Size(max = 80) String item,
            @Size(max = 80) String partName,
            @PositiveOrZero(message = "비용은 0 이상이어야 합니다.") Long cost,
            @Size(max = 80) String shopName,
            @Size(max = 500) String memo
    ) {
    }

    /**
     * 정비 이력 응답.
     *
     * @param id          PK
     * @param vehicleId   차량 PK
     * @param plateNumber 차량 번호
     * @param serviceDate 정비일
     * @param odometerKm  주행거리
     * @param item        정비 항목
     * @param partName    부품명
     * @param cost        비용
     * @param shopName    정비소
     * @param memo        메모
     */
    public record MaintenanceResponse(
            Long id,
            Long vehicleId,
            String plateNumber,
            LocalDate serviceDate,
            Integer odometerKm,
            String item,
            String partName,
            Long cost,
            String shopName,
            String memo
    ) {
        /**
         * 엔티티 → DTO 변환.
         *
         * <p>{@code vehicle} 은 LAZY 이므로 번호판을 읽을 때 프록시가 초기화된다.
         * 서비스 계층에서 트랜잭션 안에서 변환하도록 호출 순서를 맞춘다.
         *
         * @param m 정비 엔티티
         * @return DTO
         */
        public static MaintenanceResponse from(MaintenanceRecord m) {
            return new MaintenanceResponse(
                    m.getId(),
                    m.getVehicle() == null ? null : m.getVehicle().getId(),
                    m.getVehicle() == null ? null : m.getVehicle().getPlateNumber(),
                    m.getServiceDate(),
                    m.getOdometerKm(),
                    m.getItem(),
                    m.getPartName(),
                    m.getCost(),
                    m.getShopName(),
                    m.getMemo()
            );
        }
    }
}
