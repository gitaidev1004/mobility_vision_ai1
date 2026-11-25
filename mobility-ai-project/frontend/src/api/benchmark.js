/**
 * 모델 벤치마크 API (계약서 4장)
 * GET  /api/v1/benchmarks       → model_benchmark 목록
 * POST /api/v1/benchmarks/run   → AI 벤치마크 실행 후 저장
 *
 * 레코드 스키마:
 *  {id, modelName, task, detectRecall, exactMatchRate,
 *   latencyP50Ms, latencyP95Ms, memoryMb, modelSizeMb, quantized, measuredAt}
 */
import http from './http'

/** 저장된 벤치마크 측정값 목록. */
export function listBenchmarks() {
  return http.get('/benchmarks').then((r) => r.data)
}

/**
 * 벤치마크 재측정 실행.
 * AI 서비스에서 모델별 지연시간/정확도를 다시 재고 DB 에 적재한다.
 * 측정에 시간이 걸릴 수 있으므로 타임아웃을 넉넉히 둔다.
 */
export function runBenchmark() {
  return http.post('/benchmarks/run', null, { timeout: 120000 }).then((r) => r.data)
}
