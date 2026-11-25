package com.mobility.ai.client;

/**
 * AI 서비스 {@code POST /api/v1/models/benchmark} 응답 배열의 1개 원소.
 *
 * @param modelName      모델 이름
 * @param task           측정 과제
 * @param detectRecall   검출 재현율
 * @param exactMatchRate 문자열 정확 일치율
 * @param latencyP50Ms   지연 p50(ms)
 * @param latencyP95Ms   지연 p95(ms)
 * @param memoryMb       메모리 사용량(MB)
 * @param modelSizeMb    모델 크기(MB)
 * @param quantized      양자화 여부
 */
public record AiBenchmarkResult(
        String modelName,
        String task,
        Double detectRecall,
        Double exactMatchRate,
        Double latencyP50Ms,
        Double latencyP95Ms,
        Double memoryMb,
        Double modelSizeMb,
        boolean quantized
) {
}
