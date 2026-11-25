package com.mobility.ai.dto;

import com.mobility.ai.domain.Vehicle;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;

/**
 * 차량 관련 DTO 모음. 계약서 4장 {@code /api/v1/vehicles} 계열.
 */
public final class VehicleDtos {

    private VehicleDtos() {
    }

    /**
     * 차량 등록/수정 요청.
     *
     * @param plateNumber   차량 번호(필수)
     * @param ownerName     소유자(필수)
     * @param phone         연락처
     * @param vehicleType   차종
     * @param modelName     모델명
     * @param modelYear     연식
     * @param vin           차대번호
     * @param passType      이용권(NONE|VISITOR|REGULAR|STAFF)
     * @param passExpiresAt 이용권 만료 시각
     * @param active        사용 여부(null 이면 true)
     */
    public record VehicleRequest(
            @NotBlank(message = "차량 번호는 필수입니다.")
            @Size(max = 20, message = "차량 번호는 20자 이하여야 합니다.")
            String plateNumber,

            @NotBlank(message = "소유자 이름은 필수입니다.")
            @Size(max = 50, message = "소유자 이름은 50자 이하여야 합니다.")
            String ownerName,

            @Size(max = 30) String phone,
            @Size(max = 30) String vehicleType,
            @Size(max = 80) String modelName,

            @Min(value = 1900, message = "연식이 올바르지 않습니다.")
            @Max(value = 2100, message = "연식이 올바르지 않습니다.")
            Integer modelYear,

            @Size(max = 40) String vin,
            String passType,
            LocalDateTime passExpiresAt,
            Boolean active
    ) {
    }

    /**
     * 차량 상세/목록 응답.
     *
     * @param id            PK
     * @param plateNumber   차량 번호
     * @param ownerName     소유자
     * @param phone         연락처
     * @param vehicleType   차종
     * @param modelName     모델명
     * @param modelYear     연식
     * @param vin           차대번호
     * @param passType      이용권
     * @param passExpiresAt 이용권 만료 시각
     * @param active        사용 여부
     * @param registeredAt  등록 시각
     */
    public record VehicleResponse(
            Long id,
            String plateNumber,
            String ownerName,
            String phone,
            String vehicleType,
            String modelName,
            Integer modelYear,
            String vin,
            String passType,
            LocalDateTime passExpiresAt,
            boolean active,
            LocalDateTime registeredAt
    ) {
        /**
         * 엔티티 → 응답 변환.
         *
         * @param v 차량 엔티티
         * @return 응답 DTO
         */
        public static VehicleResponse from(Vehicle v) {
            return new VehicleResponse(
                    v.getId(),
                    v.getPlateNumber(),
                    v.getOwnerName(),
                    v.getPhone(),
                    v.getVehicleType(),
                    v.getModelName(),
                    v.getModelYear(),
                    v.getVin(),
                    v.getPassType() == null ? null : v.getPassType().name(),
                    v.getPassExpiresAt(),
                    v.isActive(),
                    v.getRegisteredAt()
            );
        }
    }

    /**
     * 인식 결과에 포함되는 축약 차량 정보.
     * 계약서 {@code RecognitionResultResponse.matchedVehicle} 와 1:1 이다.
     *
     * @param id          PK
     * @param plateNumber 차량 번호
     * @param ownerName   소유자
     * @param passType    이용권
     */
    public record VehicleSummary(
            Long id,
            String plateNumber,
            String ownerName,
            String passType
    ) {
        /**
         * 엔티티 → 축약 응답 변환. 입력이 null 이면 null 을 돌려준다.
         *
         * @param v 차량 엔티티(nullable)
         * @return 축약 DTO 또는 null
         */
        public static VehicleSummary from(Vehicle v) {
            if (v == null) {
                return null;
            }
            return new VehicleSummary(
                    v.getId(),
                    v.getPlateNumber(),
                    v.getOwnerName(),
                    v.getPassType() == null ? null : v.getPassType().name()
            );
        }
    }
}
