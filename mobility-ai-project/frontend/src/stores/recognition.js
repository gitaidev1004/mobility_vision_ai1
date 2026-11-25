/**
 * 번호판 인식 스토어 (계약서 5장)
 * - 업로드 → 인식 → 결과 표시 → 이력 갱신의 흐름을 관리한다.
 * - 검출 박스 오버레이를 그리려면 '원본 이미지 크기' 가 필요하므로 함께 보관한다.
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as recognitionApi from '@/api/recognition'
import { extractMessage } from '@/api/http'

export const useRecognitionStore = defineStore('recognition', () => {
  // ---------------- 상태: 업로드/인식 ----------------
  /** 선택한 파일 객체 */
  const file = ref(null)
  /** 미리보기용 Object URL */
  const previewUrl = ref('')
  /** 업로드한 이미지의 자연 크기 {width,height} — 박스 좌표 스케일 계산에 사용 */
  const imageSize = ref({ width: 0, height: 0 })
  /** 업로드 진행률 0~100 */
  const progress = ref(0)
  /** 게이트 선택값 */
  const gate = ref('정문')
  /** 이벤트 타입 ENTRY|EXIT */
  const eventType = ref('ENTRY')
  /** RecognitionResultResponse */
  const result = ref(null)
  const recognizing = ref(false)
  const error = ref('')

  // ---------------- 상태: 이력 ----------------
  const history = ref([])
  const historyTotal = ref(0)
  const historyTotalPages = ref(0)
  const historyPage = ref(0)
  const historySize = ref(10)
  const historyStatus = ref('')
  const historyLoading = ref(false)

  // ---------------- 계산 속성 ----------------
  /** 검출 박스 배열. detections:[{bbox:[x1,y1,x2,y2], score}] */
  const detections = computed(() => result.value?.detections ?? [])

  /**
   * 검출 박스를 0~100(%) 좌표로 정규화한다.
   * 이미지가 CSS로 축소되어도 오버레이가 어긋나지 않게 퍼센트로 그린다.
   */
  const detectionBoxes = computed(() => {
    const { width, height } = imageSize.value
    if (!width || !height) return []
    return detections.value.map((d, i) => {
      const [x1, y1, x2, y2] = d.bbox ?? [0, 0, 0, 0]
      return {
        index: i,
        label: d.label || 'plate',
        score: d.score ?? 0,
        left: (x1 / width) * 100,
        top: (y1 / height) * 100,
        width: ((x2 - x1) / width) * 100,
        height: ((y2 - y1) / height) * 100,
      }
    })
  })

  /** 인식 성공 여부 */
  const isSuccess = computed(() => result.value?.status === 'SUCCESS')

  /** 출입 허용 여부 */
  const isAllowed = computed(() => result.value?.allowed === true)

  /** 소요시간 요약 {detect, ocr, total} */
  const elapsed = computed(() => result.value?.elapsedMs ?? { detect: 0, ocr: 0, total: 0 })

  /** 이미지 품질 {score, readable} */
  const quality = computed(() => result.value?.quality ?? null)

  /** 매칭된 등록 차량 {id, plateNumber, ownerName, passType} */
  const matchedVehicle = computed(() => result.value?.matchedVehicle ?? null)

  /** 업로드 준비 완료 여부 */
  const canSubmit = computed(() => !!file.value && !recognizing.value)

  // ---------------- 액션 ----------------
  /**
   * 파일 선택. 이전 미리보기 URL 은 메모리 누수를 막기 위해 해제한다.
   * @param {File} f 이미지 파일
   */
  function selectFile(f) {
    revokePreview()
    file.value = f || null
    result.value = null
    error.value = ''
    progress.value = 0
    imageSize.value = { width: 0, height: 0 }
    if (f && typeof URL !== 'undefined') {
      previewUrl.value = URL.createObjectURL(f)
    } else {
      previewUrl.value = ''
    }
  }

  /** 미리보기 Object URL 해제. */
  function revokePreview() {
    if (previewUrl.value && typeof URL !== 'undefined') {
      URL.revokeObjectURL(previewUrl.value)
    }
    previewUrl.value = ''
  }

  /** <img> 의 load 이벤트에서 자연 크기를 기록한다. */
  function setImageSize(width, height) {
    imageSize.value = { width: Number(width) || 0, height: Number(height) || 0 }
  }

  /**
   * 인식 실행. 성공/실패 모두 이력을 다시 불러온다.
   * @returns {Promise<object|null>} RecognitionResultResponse
   */
  async function submit() {
    if (!file.value) {
      error.value = '인식할 이미지를 먼저 선택하세요.'
      return null
    }
    recognizing.value = true
    error.value = ''
    progress.value = 0
    try {
      const data = await recognitionApi.recognize({
        file: file.value,
        gate: gate.value,
        eventType: eventType.value,
        onProgress: (p) => {
          progress.value = p
        },
      })
      result.value = data
      // 새 인식 건이 맨 위에 보이도록 이력 첫 페이지를 다시 읽는다.
      await fetchHistory({ page: 0 })
      return data
    } catch (e) {
      error.value = extractMessage(e)
      return null
    } finally {
      recognizing.value = false
      progress.value = 100
    }
  }

  /**
   * 인식 이력 조회.
   * @param {{page?:number, size?:number, status?:string, from?:string, to?:string}} params
   */
  async function fetchHistory(params = {}) {
    if (params.page !== undefined) historyPage.value = params.page
    if (params.size !== undefined) historySize.value = params.size
    if (params.status !== undefined) historyStatus.value = params.status

    historyLoading.value = true
    try {
      const data = await recognitionApi.listRecognitions({
        page: historyPage.value,
        size: historySize.value,
        status: historyStatus.value || undefined,
        from: params.from,
        to: params.to,
      })
      history.value = data?.content ?? []
      historyTotal.value = data?.totalElements ?? 0
      historyTotalPages.value = data?.totalPages ?? 0
      historyPage.value = data?.page ?? historyPage.value
    } catch (e) {
      // 이력 조회 실패는 인식 자체를 막지 않으므로 메시지만 남긴다.
      error.value = error.value || extractMessage(e)
      history.value = []
    } finally {
      historyLoading.value = false
    }
  }

  /** 결과 패널 초기화(다시 업로드). */
  function clearResult() {
    revokePreview()
    file.value = null
    result.value = null
    error.value = ''
    progress.value = 0
    imageSize.value = { width: 0, height: 0 }
  }

  /** 스토어 초기화. */
  function reset() {
    clearResult()
    history.value = []
    historyTotal.value = 0
    historyTotalPages.value = 0
    historyPage.value = 0
    historyStatus.value = ''
  }

  return {
    file,
    previewUrl,
    imageSize,
    progress,
    gate,
    eventType,
    result,
    recognizing,
    error,
    history,
    historyTotal,
    historyTotalPages,
    historyPage,
    historySize,
    historyStatus,
    historyLoading,
    detections,
    detectionBoxes,
    isSuccess,
    isAllowed,
    elapsed,
    quality,
    matchedVehicle,
    canSubmit,
    selectFile,
    revokePreview,
    setImageSize,
    submit,
    fetchHistory,
    clearResult,
    reset,
  }
})
