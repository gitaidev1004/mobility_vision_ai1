package com.mobility.ai.service;

import com.mobility.ai.client.AiBenchmarkResult;
import com.mobility.ai.client.AiInferenceClient;
import com.mobility.ai.common.BusinessException;
import com.mobility.ai.domain.ModelBenchmark;
import com.mobility.ai.dto.BenchmarkDtos.BenchmarkResponse;
import com.mobility.ai.repository.ModelBenchmarkRepository;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 모델 벤치마크 서비스. 계약서 4장 {@code /api/v1/benchmarks}.
 * 교재 6장 "모델 성능 비교" 절에서 이 결과 표를 그대로 인용한다.
 */
@Service
public class BenchmarkService {

    private static final Logger log = LoggerFactory.getLogger(BenchmarkService.class);

    private final ModelBenchmarkRepository modelBenchmarkRepository;
    private final AiInferenceClient aiInferenceClient;

    /**
     * @param modelBenchmarkRepository 벤치마크 리포지토리
     * @param aiInferenceClient        AI 호출 클라이언트
     */
    public BenchmarkService(ModelBenchmarkRepository modelBenchmarkRepository,
                            AiInferenceClient aiInferenceClient) {
        this.modelBenchmarkRepository = modelBenchmarkRepository;
        this.aiInferenceClient = aiInferenceClient;
    }

    /**
     * 저장된 벤치마크 전체 목록(최신 측정 우선).
     *
     * @return 벤치마크 응답 목록
     */
    @Transactional(readOnly = true)
    public List<BenchmarkResponse> list() {
        return modelBenchmarkRepository.findAllOrderByMeasuredAtDesc().stream()
                .map(BenchmarkResponse::from)
                .toList();
    }

    /**
     * AI 서비스에서 벤치마크를 실행하고 결과를 저장한다.
     *
     * @return 이번 실행으로 저장된 결과 목록
     * @throws BusinessException AI 서비스가 결과를 돌려주지 못했을 때
     *                          (이 API 는 명시적 실행 요청이므로, 조용히 빈 배열을
     *                           돌려주기보다 사유를 알려 주는 편이 운영에 유용하다)
     */
    @Transactional
    public List<BenchmarkResponse> run() {
        List<AiBenchmarkResult> results = aiInferenceClient.runBenchmark();
        if (results.isEmpty()) {
            throw new BusinessException(
                    "AI 서비스에서 벤치마크 결과를 받지 못했습니다. AI 서비스(8000) 상태를 확인하세요.");
        }

        LocalDateTime measuredAt = LocalDateTime.now();
        List<ModelBenchmark> entities = new ArrayList<>(results.size());
        for (AiBenchmarkResult r : results) {
            ModelBenchmark b = new ModelBenchmark(
                    r.modelName() == null ? "unknown" : r.modelName(),
                    r.task() == null ? "unknown" : r.task());
            b.setDetectRecall(r.detectRecall());
            b.setExactMatchRate(r.exactMatchRate());
            b.setLatencyP50Ms(r.latencyP50Ms());
            b.setLatencyP95Ms(r.latencyP95Ms());
            b.setMemoryMb(r.memoryMb());
            b.setModelSizeMb(r.modelSizeMb());
            b.setQuantized(r.quantized());
            b.setMeasuredAt(measuredAt);
            entities.add(b);
        }

        log.info("벤치마크 {}건을 저장했습니다.", entities.size());
        return modelBenchmarkRepository.saveAll(entities).stream()
                .map(BenchmarkResponse::from)
                .toList();
    }
}
