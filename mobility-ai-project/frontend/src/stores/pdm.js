/**
 * 예지보전(PdM) 스토어 (계약서 5장)
 * - 차량 선택 → 최신 예측 + 센서 시계열 + 부품별 점수 + 권고를 한 번에 로드한다.
 * - 응답 필드명은 계약서 4장(백엔드 camelCase)을 그대로 따른다.
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as pdmApi from '@/api/pdm'
import * as sensorApi from '@/api/sensor'
import * as vehicleApi from '@/api/vehicle'
import { extractMessage } from '@/api/http'

/** 부품 코드 → 한국어 라벨 */
export const COMPONENT_LABEL = {
  engine: '엔진',
  battery: '배터리',
  brake: '브레이크',
  tire: '타이어',
  oil: '엔진오일',
  transmission: '변속기',
  cooling: '냉각계통',
  suspension: '서스펜션',
}

/** 부품 상태 enum → 배지 색상 톤 */
export const COMPONENT_STATUS_TONE = {
  NORMAL: 'success',
  WATCH: 'warn',
  WARNING: 'warn',
  CRITICAL: 'danger',
}

/** 부품 상태 enum → 한국어 라벨 */
export const COMPONENT_STATUS_LABEL = {
  NORMAL: '정상',
  WATCH: '관찰',
  WARNING: '주의',
  CRITICAL: '위험',
}

export const usePdmStore = defineStore('pdm', () => {
  // ---------------- 상태 ----------------
  /** 차량 선택 드롭다운용 목록 */
  const vehicles = ref([])
  /** 선택된 차량 ID */
  const selectedId = ref(null)
  /** 최신 예측 결과 */
  const prediction = ref(null)
  /** 센서 시계열(measured_at 오름차순) */
  const readings = ref([])
  /** 위험 차량 목록 */
  const risky = ref([])
  const riskyLevel = ref('HIGH')

  const loading = ref(false)
  const predicting = ref(false)
  const error = ref('')

  // ---------------- 계산 속성 ----------------
  /** 선택된 차량 객체 */
  const selectedVehicle = computed(
    () => vehicles.value.find((v) => v.id === selectedId.value) ?? null,
  )

  /** 건강 점수 0~100 (게이지) */
  const healthScore = computed(() => prediction.value?.healthScore ?? 0)

  /** 위험 등급 enum */
  const riskLevel = computed(() => prediction.value?.riskLevel ?? 'LOW')

  /** 고장 확률 0~1 */
  const failureRisk = computed(() => prediction.value?.failureRisk ?? 0)

  /** 이상치 점수 0~1 */
  const anomalyScore = computed(() => prediction.value?.anomalyScore ?? 0)

  /** 잔존수명(일) */
  const rulDays = computed(() => prediction.value?.rulDays ?? 0)

  /** 가장 위험한 부품 코드 */
  const topComponent = computed(() => prediction.value?.topComponent ?? null)

  /**
   * 부품별 점수. 낮은 점수(= 위험)부터 정렬하고 한국어 라벨을 붙인다.
   * components:[{name,score,status,reason}]
   */
  const components = computed(() => {
    const list = prediction.value?.components ?? []
    return [...list]
      .map((c) => ({
        ...c,
        label: COMPONENT_LABEL[c.name] || c.name,
        tone: COMPONENT_STATUS_TONE[c.status] || 'neutral',
        statusLabel: COMPONENT_STATUS_LABEL[c.status] || c.status,
        isTop: c.name === prediction.value?.topComponent,
      }))
      .sort((a, b) => (a.score ?? 0) - (b.score ?? 0))
  })

  /**
   * 정비 권고. priority 오름차순(1이 가장 급함).
   * recommendations:[{priority,item,dueKm,dueDays}]
   */
  const recommendations = computed(() => {
    const list = prediction.value?.recommendations ?? []
    return [...list].sort((a, b) => (a.priority ?? 99) - (b.priority ?? 99))
  })

  /** 게이지 색상 톤: 점수 구간별 */
  const healthTone = computed(() => {
    const s = healthScore.value
    if (s >= 80) return 'success'
    if (s >= 60) return 'warn'
    return 'danger'
  })

  /**
   * 센서 추이 차트용 시리즈.
   * 값의 스케일이 서로 크게 다르므로 LineChart 를 여러 개 나눠 그린다.
   */
  const sensorLabels = computed(() =>
    readings.value.map((r) => {
      const d = new Date(r.measuredAt ?? r.measured_at)
      if (Number.isNaN(d.getTime())) return ''
      return `${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')}`
    }),
  )

  /** 센서 필드명을 camelCase/snake_case 양쪽 모두 허용해서 읽는다. */
  function pick(row, camel, snake) {
    const v = row?.[camel] ?? row?.[snake]
    return v === null || v === undefined ? 0 : Number(v)
  }

  /** 엔진 온도 추이(°C) */
  const engineTempSeries = computed(() => [
    {
      name: '냉각수 온도(°C)',
      color: 'var(--c-chart-4)',
      values: readings.value.map((r) => pick(r, 'engineTempC', 'engine_temp_c')),
    },
  ])

  /** 배터리 추이(전압 + SOH%) */
  const batterySeries = computed(() => [
    {
      name: '배터리 전압(V)',
      color: 'var(--c-chart-1)',
      values: readings.value.map((r) => pick(r, 'batteryVoltage', 'battery_voltage')),
    },
  ])

  /** 배터리 SOH 추이(%) */
  const batterySohSeries = computed(() => [
    {
      name: '배터리 SOH(%)',
      color: 'var(--c-chart-5)',
      values: readings.value.map((r) => pick(r, 'batterySoh', 'battery_soh') * 100),
    },
  ])

  /** 브레이크 패드 두께 추이(mm) */
  const brakeSeries = computed(() => [
    {
      name: '브레이크 패드(mm)',
      color: 'var(--c-chart-3)',
      values: readings.value.map((r) => pick(r, 'brakePadMm', 'brake_pad_mm')),
    },
  ])

  /** 진동 RMS 추이 */
  const vibrationSeries = computed(() => [
    {
      name: '진동 RMS',
      color: 'var(--c-chart-6)',
      values: readings.value.map((r) => pick(r, 'vibrationRms', 'vibration_rms')),
    },
  ])

  /** 최신 센서 1건(현재 상태 표시용) */
  const latestReading = computed(() =>
    readings.value.length ? readings.value[readings.value.length - 1] : null,
  )

  /** 네 바퀴 타이어 공기압(psi) — 최신값 */
  const tirePressures = computed(() => {
    const r = latestReading.value
    if (!r) return []
    return [
      { pos: '앞 좌', value: pick(r, 'tirePressureFl', 'tire_pressure_fl') },
      { pos: '앞 우', value: pick(r, 'tirePressureFr', 'tire_pressure_fr') },
      { pos: '뒤 좌', value: pick(r, 'tirePressureRl', 'tire_pressure_rl') },
      { pos: '뒤 우', value: pick(r, 'tirePressureRr', 'tire_pressure_rr') },
    ]
  })

  // ---------------- 액션 ----------------
  /** 차량 선택 드롭다운 채우기(첫 페이지 100대). */
  async function fetchVehicles() {
    try {
      const data = await vehicleApi.listVehicles({ page: 0, size: 100 })
      vehicles.value = data?.content ?? []
      // 아무것도 선택되지 않았으면 첫 차량을 자동 선택한다.
      if (!selectedId.value && vehicles.value.length) {
        selectedId.value = vehicles.value[0].id
      }
    } catch (e) {
      error.value = extractMessage(e)
      vehicles.value = []
    }
  }

  /**
   * 차량 선택. 최신 예측과 센서 시계열을 함께 로드한다.
   * @param {number} id vehicle.id
   */
  async function selectVehicle(id) {
    selectedId.value = Number(id)
    await loadDetail()
  }

  /** 선택된 차량의 예측 + 센서를 병렬로 로드한다. */
  async function loadDetail() {
    if (!selectedId.value) return
    loading.value = true
    error.value = ''
    try {
      const [pred, rows] = await Promise.all([
        pdmApi.getLatest(selectedId.value),
        sensorApi.listSensors({ vehicleId: selectedId.value, limit: 60 }),
      ])
      prediction.value = pred
      // 서버가 최신순으로 줄 수도 있으므로 measured_at 오름차순으로 정렬한다.
      readings.value = (Array.isArray(rows) ? rows : rows?.content ?? [])
        .slice()
        .sort((a, b) => {
          const ta = new Date(a.measuredAt ?? a.measured_at).getTime()
          const tb = new Date(b.measuredAt ?? b.measured_at).getTime()
          return ta - tb
        })
    } catch (e) {
      error.value = extractMessage(e)
      prediction.value = null
      readings.value = []
    } finally {
      loading.value = false
    }
  }

  /** 예측 재실행(AI 서비스 호출). */
  async function runPredict() {
    if (!selectedId.value) return null
    predicting.value = true
    error.value = ''
    try {
      const data = await pdmApi.predict(selectedId.value)
      prediction.value = data
      return data
    } catch (e) {
      error.value = extractMessage(e)
      return null
    } finally {
      predicting.value = false
    }
  }

  /** 위험 차량 목록 조회. */
  async function fetchRisky(level) {
    if (level) riskyLevel.value = level
    try {
      const data = await pdmApi.listRisky({ level: riskyLevel.value, page: 0, size: 10 })
      risky.value = Array.isArray(data) ? data : data?.content ?? []
    } catch (e) {
      error.value = extractMessage(e)
      risky.value = []
    }
  }

  /** 스토어 초기화. */
  function reset() {
    vehicles.value = []
    selectedId.value = null
    prediction.value = null
    readings.value = []
    risky.value = []
    error.value = ''
  }

  return {
    vehicles,
    selectedId,
    prediction,
    readings,
    risky,
    riskyLevel,
    loading,
    predicting,
    error,
    selectedVehicle,
    healthScore,
    riskLevel,
    failureRisk,
    anomalyScore,
    rulDays,
    topComponent,
    components,
    recommendations,
    healthTone,
    sensorLabels,
    engineTempSeries,
    batterySeries,
    batterySohSeries,
    brakeSeries,
    vibrationSeries,
    latestReading,
    tirePressures,
    fetchVehicles,
    selectVehicle,
    loadDetail,
    runPredict,
    fetchRisky,
    reset,
  }
})
