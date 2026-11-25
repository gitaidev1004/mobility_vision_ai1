package com.mobility.ai.web;

import com.mobility.ai.common.PageResponse;
import com.mobility.ai.dto.MaintenanceDtos.MaintenanceRequest;
import com.mobility.ai.dto.MaintenanceDtos.MaintenanceResponse;
import com.mobility.ai.service.MaintenanceService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

/**
 * 정비 이력 컨트롤러.
 *
 * <pre>
 * GET  /api/v1/maintenance?vehicleId=&page=&size=
 * POST /api/v1/maintenance
 * </pre>
 */
@RestController
@RequestMapping("/api/v1/maintenance")
public class MaintenanceController {

    private final MaintenanceService maintenanceService;

    /**
     * @param maintenanceService 정비 서비스
     */
    public MaintenanceController(MaintenanceService maintenanceService) {
        this.maintenanceService = maintenanceService;
    }

    /**
     * 정비 이력 조회.
     *
     * @param vehicleId 차량 PK(생략 시 전체)
     * @param page      페이지 번호
     * @param size      페이지 크기
     * @return 정비 이력 페이지
     */
    @GetMapping
    public PageResponse<MaintenanceResponse> list(
            @RequestParam(name = "vehicleId", required = false) Long vehicleId,
            @RequestParam(name = "page", defaultValue = "0") int page,
            @RequestParam(name = "size", defaultValue = "20") int size) {
        return maintenanceService.search(vehicleId, page, size);
    }

    /**
     * 정비 이력 등록.
     *
     * @param request 등록 요청
     * @return 생성된 정비 이력(201)
     */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public MaintenanceResponse create(@Valid @RequestBody MaintenanceRequest request) {
        return maintenanceService.create(request);
    }
}
