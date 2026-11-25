package com.mobility.ai.web;

import com.mobility.ai.common.PageResponse;
import com.mobility.ai.dto.PdmDtos.PdmPredictionResponse;
import com.mobility.ai.dto.PdmDtos.RiskyVehicleResponse;
import com.mobility.ai.service.PdmService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * 예지보전 컨트롤러.
 *
 * <pre>
 * POST /api/v1/pdm/vehicles/{id}/predict
 * GET  /api/v1/pdm/vehicles/{id}/latest
 * GET  /api/v1/pdm/risky?level=HIGH&page=&size=
 * </pre>
 */
@RestController
@RequestMapping("/api/v1/pdm")
public class PdmController {

    private final PdmService pdmService;

    /**
     * @param pdmService 예지보전 서비스
     */
    public PdmController(PdmService pdmService) {
        this.pdmService = pdmService;
    }

    /**
     * 예측을 실행하고 저장한다(AI 호출).
     *
     * @param id 차량 PK
     * @return 저장된 예측
     */
    @PostMapping("/vehicles/{id}/predict")
    public PdmPredictionResponse predict(@PathVariable("id") Long id) {
        return pdmService.predict(id);
    }

    /**
     * 최신 예측 조회.
     *
     * @param id 차량 PK
     * @return 최신 예측
     */
    @GetMapping("/vehicles/{id}/latest")
    public PdmPredictionResponse latest(@PathVariable("id") Long id) {
        return pdmService.latest(id);
    }

    /**
     * 위험 차량 목록.
     *
     * @param level 최소 위험 등급(기본 HIGH)
     * @param page  페이지 번호
     * @param size  페이지 크기
     * @return 위험 차량 페이지
     */
    @GetMapping("/risky")
    public PageResponse<RiskyVehicleResponse> risky(
            @RequestParam(name = "level", defaultValue = "HIGH") String level,
            @RequestParam(name = "page", defaultValue = "0") int page,
            @RequestParam(name = "size", defaultValue = "20") int size) {
        return pdmService.risky(level, page, size);
    }
}
