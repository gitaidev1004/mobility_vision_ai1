/**
 * 번호판 인식 API (계약서 4장)
 * POST /api/v1/recognitions                     multipart: file, gate, eventType
 *      → RecognitionResultResponse
 * GET  /api/v1/recognitions?from=&to=&status=&page=&size=
 * GET  /api/v1/recognitions/failures?page=&size=
 */
import http, { cleanParams } from './http'

/**
 * 이미지 업로드 → AI 인식 → 이벤트/주차세션 처리까지 백엔드가 한 번에 수행한다.
 * @param {{file: File, gate: string, eventType: 'ENTRY'|'EXIT', onProgress?: Function}} args
 * @returns {Promise<object>} RecognitionResultResponse
 *   {recognitionId, plateNumber, confidence, status, validFormat, matchedVehicle,
 *    eventId, eventType, allowed, rejectReason, parkingSessionId, feeAmount,
 *    parkedMinutes, detections, elapsedMs, quality}
 */
export function recognize({ file, gate, eventType, onProgress }) {
  const form = new FormData()
  form.append('file', file)
  form.append('gate', gate)
  form.append('eventType', eventType)

  return http
    .post('/recognitions', form, {
      // 업로드 진행률을 화면에 표시하기 위한 콜백
      onUploadProgress: (e) => {
        if (!onProgress || !e.total) return
        onProgress(Math.round((e.loaded / e.total) * 100))
      },
    })
    .then((r) => r.data)
}

/**
 * 인식 이력 조회.
 * @param {{from?:string, to?:string, status?:string, page?:number, size?:number}} params
 *        status ∈ SUCCESS|LOW_CONFIDENCE|NO_PLATE|FORMAT_INVALID
 */
export function listRecognitions(params = {}) {
  const { from, to, status, page = 0, size = 20 } = params
  return http
    .get('/recognitions', { params: cleanParams({ from, to, status, page, size }) })
    .then((r) => r.data)
}

/** 인식 실패 목록(상태가 SUCCESS 가 아닌 건). */
export function listFailures(params = {}) {
  const { page = 0, size = 20 } = params
  return http
    .get('/recognitions/failures', { params: cleanParams({ page, size }) })
    .then((r) => r.data)
}
