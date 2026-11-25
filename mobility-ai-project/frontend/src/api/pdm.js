/**
 * 예지보전(PdM) API (계약서 4장)
 * POST /api/v1/pdm/vehicles/{id}/predict   → AI 호출 후 저장
 * GET  /api/v1/pdm/vehicles/{id}/latest    → 최신 예측
 * GET  /api/v1/pdm/risky?level=HIGH&page=&size=
 *
 * 예측 응답 스키마(계약서 3장 AI 응답을 백엔드가 그대로 중계):
 *   {vehicleId, healthScore, anomalyScore, failureRisk, riskLevel, rulDays,
 *    topComponent, components:[{name,score,status,reason}],
 *    recommendations:[{priority,item,dueKm,dueDays}], modelVersion, computedAt}
 */
import http, { cleanParams } from './http'

/** 특정 차량의 예지보전 예측을 실행(AI 서비스 호출 후 DB 저장). */
export function predict(vehicleId) {
  return http.post(`/pdm/vehicles/${vehicleId}/predict`).then((r) => r.data)
}

/**
 * 최신 예측 결과 조회.
 * 아직 예측 이력이 없으면 404 가 올 수 있으므로 null 로 변환한다.
 */
export function getLatest(vehicleId) {
  return http
    .get(`/pdm/vehicles/${vehicleId}/latest`)
    .then((r) => r.data)
    .catch((e) => {
      if (e?.response?.status === 404) return null
      throw e
    })
}

/**
 * 위험 차량 목록.
 * @param {{level?:'LOW'|'MEDIUM'|'HIGH'|'CRITICAL', page?:number, size?:number}} params
 */
export function listRisky(params = {}) {
  const { level = 'HIGH', page = 0, size = 20 } = params
  return http
    .get('/pdm/risky', { params: cleanParams({ level, page, size }) })
    .then((r) => r.data)
}
