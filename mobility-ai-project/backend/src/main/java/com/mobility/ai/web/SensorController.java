package com.mobility.ai.web;

import com.mobility.ai.dto.SensorDtos.SensorRequest;
import com.mobility.ai.dto.SensorDtos.SensorResponse;
import com.mobility.ai.service.SensorService;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

/**
 * 센서 시계열 컨트롤러.
 *
 * <pre>
 * GET  /api/v1/sensors?vehicleId=&limit=100
 * POST /api/v1/sensors            (배열 본문)
 * </pre>
 *
 * <p><b>배열 본문의 검증</b>: {@code @Valid} 는 {@code @RequestBody List<...>} 의
 * <i>원소</i>까지 자동으로 내려가지 않는다. 원소 단위 Bean Validation 을 걸려면
 * 메서드 검증(프록시) 설정이 필요하고 그 동작은 스프링 마이너 버전에 따라 달라진다.
 * 교재 부록은 어느 버전에서도 같게 동작해야 하므로, 여기서는 애너테이션에 의존하지 않고
 * {@code SensorService.ingest} 가 <b>원소를 명시적으로 검증</b>한다
 * (몇 번째 원소가 왜 잘못됐는지 메시지에 담아 준다).
 */
@RestController
@RequestMapping("/api/v1/sensors")
public class SensorController {

    private final SensorService sensorService;

    /**
     * @param sensorService 센서 서비스
     */
    public SensorController(SensorService sensorService) {
        this.sensorService = sensorService;
    }

    /**
     * 센서 시계열 조회(최근 데이터 우선).
     *
     * @param vehicleId 차량 PK(생략 시 전체 차량)
     * @param limit     최대 건수(기본 100)
     * @return 센서 목록
     */
    @GetMapping
    public List<SensorResponse> list(
            @RequestParam(name = "vehicleId", required = false) Long vehicleId,
            @RequestParam(name = "limit", defaultValue = "100") int limit) {
        return sensorService.list(vehicleId, limit);
    }

    /**
     * 센서 데이터 일괄 적재.
     *
     * @param requests 센서 배열
     * @return 저장된 센서 목록(201)
     */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public List<SensorResponse> ingest(@RequestBody List<SensorRequest> requests) {
        return sensorService.ingest(requests);
    }
}
