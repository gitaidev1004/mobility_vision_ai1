package com.mobility.ai.web;

import com.mobility.ai.dto.BenchmarkDtos.BenchmarkResponse;
import com.mobility.ai.service.BenchmarkService;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 모델 벤치마크 컨트롤러.
 *
 * <pre>
 * GET  /api/v1/benchmarks
 * POST /api/v1/benchmarks/run
 * </pre>
 */
@RestController
@RequestMapping("/api/v1/benchmarks")
public class BenchmarkController {

    private final BenchmarkService benchmarkService;

    /**
     * @param benchmarkService 벤치마크 서비스
     */
    public BenchmarkController(BenchmarkService benchmarkService) {
        this.benchmarkService = benchmarkService;
    }

    /**
     * 저장된 모델 성능 목록.
     *
     * @return 벤치마크 목록
     */
    @GetMapping
    public List<BenchmarkResponse> list() {
        return benchmarkService.list();
    }

    /**
     * AI 서비스에서 벤치마크를 실행하고 결과를 저장한다.
     *
     * @return 이번 실행 결과 목록
     */
    @PostMapping("/run")
    public List<BenchmarkResponse> run() {
        return benchmarkService.run();
    }
}
