/**
 * 대시보드 API (계약서 4장)
 * GET /api/v1/dashboard/summary
 * 응답 스키마:
 *  {totalVehicles, todayEntries, todayExits, currentlyParked,
 *   recognitionRate, failureCount, unpaidCount, todayRevenue,
 *   riskyVehicles, avgLatencyMs,
 *   recentEvents:[{id, plateNumber, eventType, gateName, occurredAt, allowed}],
 *   hourlyTraffic:[{hour, entries, exits}],
 *   riskDistribution:{LOW, MEDIUM, HIGH, CRITICAL}}
 */
import http from './http'

/** 대시보드 요약 지표 조회(로그인 필요). */
export function getSummary() {
  return http.get('/dashboard/summary').then((r) => r.data)
}
