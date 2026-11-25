package com.mobility.ai.dto;

import com.mobility.ai.domain.ModelBenchmark;
import java.time.LocalDateTime;

/**
 * 모델 벤치마크 DTO 모음. 계약서 4장 {@code /api/v1/benchmarks}.
 */
public final class BenchmarkDtos {

    private BenchmarkDtos() {
    }

    /**
     * 벤치마크 1건 응답.
     *
     * @param id             PK
     * @param modelName      모델 이름
     * @param task           측정 과제
     * @param detectRecall   검출 재현율
     * @param exactMatchRate 문자열 정확 일치율
     * @param latencyP50Ms   지연 p50
     * @param latencyP95Ms   지연 p95
     * @param memoryMb       메모리 사용량
     * @param modelSizeMb    모델 크기
     * @param quantized      양자화 여부
     * @param measuredAt     측정 시각
     */
    public record BenchmarkResponse(
            Long id,
            String modelName,
            String task,
            Double detectRecall,
            Double exactMatchRate,
            Double latencyP50Ms,
            Double latencyP95Ms,
            Double memoryMb,
            Double modelSizeMb,
            boolean quantized,
            LocalDateTime measuredAt
    ) {
        /**
         * 엔티티 → DTO 변환.
         *
         * @param b 벤치마크 엔티티
         * @return DTO
         */
        public static BenchmarkResponse from(ModelBenchmark b) {
            return new BenchmarkResponse(
                    b.getId(),
                    b.getModelName(),
                    b.getTask(),
                    b.getDetectRecall(),
                    b.getExactMatchRate(),
                    b.getLatencyP50Ms(),
                    b.getLatencyP95Ms(),
                    b.getMemoryMb(),
                    b.getModelSizeMb(),
                    b.isQuantized(),
                    b.getMeasuredAt()
            );
        }
    }
}
