package com.mobility.ai.service;

import com.mobility.ai.common.BusinessException;
import com.mobility.ai.common.NotFoundException;
import com.mobility.ai.common.PageResponse;
import com.mobility.ai.domain.PassType;
import com.mobility.ai.domain.Vehicle;
import com.mobility.ai.dto.VehicleDtos.VehicleRequest;
import com.mobility.ai.dto.VehicleDtos.VehicleResponse;
import com.mobility.ai.repository.VehicleRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 차량 CRUD 서비스. 계약서 4장 {@code /api/v1/vehicles} 계열을 담당한다.
 *
 * <p>번호판은 저장 전에 {@link PlateFormatValidator#normalize(String)} 로 공백/하이픈을
 * 제거해 정규화한다. 인식 결과와의 매칭이 공백 유무로 깨지지 않게 하기 위함이다.
 */
@Service
public class VehicleService {

    private final VehicleRepository vehicleRepository;

    /**
     * @param vehicleRepository 차량 리포지토리
     */
    public VehicleService(VehicleRepository vehicleRepository) {
        this.vehicleRepository = vehicleRepository;
    }

    /**
     * 차량 목록 검색.
     *
     * @param q    번호판/소유자 부분 검색어(null 허용)
     * @param page 페이지 번호(0-base)
     * @param size 페이지 크기
     * @return 차량 페이지 응답
     */
    @Transactional(readOnly = true)
    public PageResponse<VehicleResponse> search(String q, int page, int size) {
        PageRequest pageable = PageRequest.of(
                Math.max(0, page),
                normalizeSize(size),
                Sort.by(Sort.Direction.DESC, "registeredAt").and(Sort.by(Sort.Direction.DESC, "id"))
        );
        Page<Vehicle> result = vehicleRepository.search(q == null ? "" : q.trim(), pageable);
        return PageResponse.from(result, VehicleResponse::from);
    }

    /**
     * 단건 조회.
     *
     * @param id 차량 PK
     * @return 차량 응답
     * @throws NotFoundException 없을 때
     */
    @Transactional(readOnly = true)
    public VehicleResponse get(Long id) {
        return VehicleResponse.from(getEntity(id));
    }

    /**
     * 번호판으로 조회.
     *
     * @param plate 차량 번호(공백/하이픈 허용)
     * @return 차량 응답
     * @throws NotFoundException 없을 때
     */
    @Transactional(readOnly = true)
    public VehicleResponse getByPlate(String plate) {
        String normalized = PlateFormatValidator.normalize(plate);
        if (normalized == null) {
            throw new NotFoundException("차량 번호가 비어 있습니다.");
        }
        Vehicle vehicle = vehicleRepository.findByPlateNumber(normalized)
                .orElseThrow(() -> NotFoundException.of("차량", normalized));
        return VehicleResponse.from(vehicle);
    }

    /**
     * 차량 등록.
     *
     * @param request 등록 요청
     * @return 생성된 차량 응답
     * @throws BusinessException 번호판이 이미 등록되어 있을 때
     */
    @Transactional
    public VehicleResponse create(VehicleRequest request) {
        String plate = PlateFormatValidator.normalize(request.plateNumber());
        if (plate == null) {
            throw new BusinessException("차량 번호가 비어 있습니다.");
        }
        if (vehicleRepository.existsByPlateNumber(plate)) {
            throw new BusinessException("이미 등록된 차량 번호입니다: " + plate);
        }

        Vehicle vehicle = new Vehicle(plate, request.ownerName());
        apply(vehicle, request);
        return VehicleResponse.from(vehicleRepository.save(vehicle));
    }

    /**
     * 차량 수정.
     *
     * @param id      차량 PK
     * @param request 수정 요청
     * @return 수정된 차량 응답
     * @throws NotFoundException 차량이 없을 때
     * @throws BusinessException 바꾸려는 번호판이 다른 차량과 중복일 때
     */
    @Transactional
    public VehicleResponse update(Long id, VehicleRequest request) {
        Vehicle vehicle = getEntity(id);

        String plate = PlateFormatValidator.normalize(request.plateNumber());
        if (plate == null) {
            throw new BusinessException("차량 번호가 비어 있습니다.");
        }
        if (!plate.equals(vehicle.getPlateNumber()) && vehicleRepository.existsByPlateNumber(plate)) {
            throw new BusinessException("이미 등록된 차량 번호입니다: " + plate);
        }

        vehicle.setPlateNumber(plate);
        vehicle.setOwnerName(request.ownerName());
        apply(vehicle, request);
        return VehicleResponse.from(vehicleRepository.save(vehicle));
    }

    /**
     * 차량 삭제.
     *
     * @param id 차량 PK
     * @throws NotFoundException 없을 때
     */
    @Transactional
    public void delete(Long id) {
        Vehicle vehicle = getEntity(id);
        vehicleRepository.delete(vehicle);
    }

    /**
     * 엔티티 조회(내부/다른 서비스 공용).
     *
     * @param id 차량 PK
     * @return 차량 엔티티
     * @throws NotFoundException 없을 때
     */
    @Transactional(readOnly = true)
    public Vehicle getEntity(Long id) {
        return vehicleRepository.findById(id)
                .orElseThrow(() -> NotFoundException.of("차량", id));
    }

    /**
     * 요청 값을 엔티티에 반영한다(번호판/소유자는 호출부에서 처리).
     *
     * @param vehicle 대상 엔티티
     * @param request 요청
     */
    private void apply(Vehicle vehicle, VehicleRequest request) {
        vehicle.setOwnerName(request.ownerName());
        vehicle.setPhone(request.phone());
        vehicle.setVehicleType(request.vehicleType());
        vehicle.setModelName(request.modelName());
        vehicle.setModelYear(request.modelYear());
        vehicle.setVin(request.vin());
        vehicle.setPassType(PassType.from(request.passType()));
        vehicle.setPassExpiresAt(request.passExpiresAt());
        vehicle.setActive(request.active() == null || request.active());
    }

    /**
     * 페이지 크기를 1~200 으로 제한한다(과도한 조회 방지).
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
