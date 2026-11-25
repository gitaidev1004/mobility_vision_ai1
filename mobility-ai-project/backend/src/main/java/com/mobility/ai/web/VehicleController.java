package com.mobility.ai.web;

import com.mobility.ai.common.PageResponse;
import com.mobility.ai.dto.VehicleDtos.VehicleRequest;
import com.mobility.ai.dto.VehicleDtos.VehicleResponse;
import com.mobility.ai.service.VehicleService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

/**
 * 차량 컨트롤러. 계약서 4장의 경로/메서드와 1:1 이다.
 *
 * <pre>
 * GET    /api/v1/vehicles?q=&page=0&size=20
 * POST   /api/v1/vehicles
 * GET    /api/v1/vehicles/{id}
 * PUT    /api/v1/vehicles/{id}
 * DELETE /api/v1/vehicles/{id}
 * GET    /api/v1/vehicles/by-plate/{plate}
 * </pre>
 */
@RestController
@RequestMapping("/api/v1/vehicles")
public class VehicleController {

    private final VehicleService vehicleService;

    /**
     * @param vehicleService 차량 서비스
     */
    public VehicleController(VehicleService vehicleService) {
        this.vehicleService = vehicleService;
    }

    /**
     * 차량 목록(페이지).
     *
     * @param q    번호판/소유자 검색어
     * @param page 페이지 번호(0-base)
     * @param size 페이지 크기
     * @return 페이지 응답 {@code {content,totalElements,totalPages,page,size}}
     */
    @GetMapping
    public PageResponse<VehicleResponse> list(
            @RequestParam(name = "q", required = false) String q,
            @RequestParam(name = "page", defaultValue = "0") int page,
            @RequestParam(name = "size", defaultValue = "20") int size) {
        return vehicleService.search(q, page, size);
    }

    /**
     * 차량 등록.
     *
     * @param request 등록 요청
     * @return 생성된 차량(201)
     */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public VehicleResponse create(@Valid @RequestBody VehicleRequest request) {
        return vehicleService.create(request);
    }

    /**
     * 차량 단건 조회.
     *
     * @param id 차량 PK
     * @return 차량
     */
    @GetMapping("/{id}")
    public VehicleResponse get(@PathVariable("id") Long id) {
        return vehicleService.get(id);
    }

    /**
     * 차량 수정.
     *
     * @param id      차량 PK
     * @param request 수정 요청
     * @return 수정된 차량
     */
    @PutMapping("/{id}")
    public VehicleResponse update(@PathVariable("id") Long id,
                                  @Valid @RequestBody VehicleRequest request) {
        return vehicleService.update(id, request);
    }

    /**
     * 차량 삭제.
     *
     * @param id 차량 PK
     */
    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@PathVariable("id") Long id) {
        vehicleService.delete(id);
    }

    /**
     * 번호판으로 차량 조회.
     *
     * @param plate 차량 번호
     * @return 차량
     */
    @GetMapping("/by-plate/{plate}")
    public VehicleResponse getByPlate(@PathVariable("plate") String plate) {
        return vehicleService.getByPlate(plate);
    }
}
