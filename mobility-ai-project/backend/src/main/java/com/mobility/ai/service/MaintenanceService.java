package com.mobility.ai.service;

import com.mobility.ai.common.PageResponse;
import com.mobility.ai.domain.MaintenanceRecord;
import com.mobility.ai.domain.Vehicle;
import com.mobility.ai.dto.MaintenanceDtos.MaintenanceRequest;
import com.mobility.ai.dto.MaintenanceDtos.MaintenanceResponse;
import com.mobility.ai.repository.MaintenanceRecordRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 정비 이력 서비스. 계약서 4장 {@code /api/v1/maintenance}.
 */
@Service
public class MaintenanceService {

    private final MaintenanceRecordRepository maintenanceRecordRepository;
    private final VehicleService vehicleService;

    /**
     * @param maintenanceRecordRepository 정비 리포지토리
     * @param vehicleService              차량 조회(존재 검증)
     */
    public MaintenanceService(MaintenanceRecordRepository maintenanceRecordRepository,
                              VehicleService vehicleService) {
        this.maintenanceRecordRepository = maintenanceRecordRepository;
        this.vehicleService = vehicleService;
    }

    /**
     * 정비 이력 검색.
     *
     * @param vehicleId 차량 PK(null 이면 전체)
     * @param page      페이지 번호
     * @param size      페이지 크기
     * @return 정비 이력 페이지
     */
    @Transactional(readOnly = true)
    public PageResponse<MaintenanceResponse> search(Long vehicleId, int page, int size) {
        PageRequest pageable = PageRequest.of(
                Math.max(0, page),
                normalizeSize(size),
                Sort.by(Sort.Direction.DESC, "serviceDate").and(Sort.by(Sort.Direction.DESC, "id"))
        );
        Page<MaintenanceRecord> result = maintenanceRecordRepository.search(vehicleId, pageable);
        return PageResponse.from(result, MaintenanceResponse::from);
    }

    /**
     * 정비 이력 등록.
     *
     * @param request 등록 요청
     * @return 생성된 정비 이력
     */
    @Transactional
    public MaintenanceResponse create(MaintenanceRequest request) {
        Vehicle vehicle = vehicleService.getEntity(request.vehicleId());

        MaintenanceRecord record =
                new MaintenanceRecord(vehicle, request.serviceDate(), request.item());
        record.setOdometerKm(request.odometerKm());
        record.setPartName(request.partName());
        record.setCost(request.cost());
        record.setShopName(request.shopName());
        record.setMemo(request.memo());

        return MaintenanceResponse.from(maintenanceRecordRepository.save(record));
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
