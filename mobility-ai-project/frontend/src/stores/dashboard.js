/**
 * 대시보드 스토어 (계약서 5장)
 * - GET /api/v1/dashboard/summary 응답을 보관하고, 화면이 바로 쓰기 좋은 형태로 가공한다.
 * - 응답 필드명은 계약서와 1:1로 일치한다(camelCase).
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as dashboardApi from '@/api/dashboard'
import { extractMessage } from '@/api/http'

/** 위험도 등급 표시 순서(차트 범례/도넛 조각 순서) */
export const RISK_ORDER = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

export const useDashboardStore = defineStore('dashboard', () => {
  // ---------------- 상태 ----------------
  /** 서버 응답 원본 */
  const summary = ref(null)
  const loading = ref(false)
  const error = ref('')
  /** 마지막 갱신 시각 */
  const loadedAt = ref(null)

  // ---------------- 계산 속성 ----------------
  /**
   * 통계 카드 8개 묶음.
   * 각 항목은 StatCard 컴포넌트의 props 형태로 반환한다.
   */
  const stats = computed(() => {
    const s = summary.value
    return [
      { key: 'totalVehicles', label: '등록 차량', value: s?.totalVehicles ?? 0, unit: '대', tone: 'primary', icon: 'car' },
      { key: 'todayEntries', label: '금일 입차', value: s?.todayEntries ?? 0, unit: '건', tone: 'success', icon: 'in' },
      { key: 'todayExits', label: '금일 출차', value: s?.todayExits ?? 0, unit: '건', tone: 'info', icon: 'out' },
      { key: 'currentlyParked', label: '현재 주차', value: s?.currentlyParked ?? 0, unit: '대', tone: 'primary', icon: 'parking' },
      { key: 'recognitionRate', label: '인식 성공률', value: s?.recognitionRate ?? 0, format: 'percent', tone: 'success', icon: 'scan' },
      { key: 'failureCount', label: '인식 실패', value: s?.failureCount ?? 0, unit: '건', tone: 'danger', icon: 'alert' },
      { key: 'todayRevenue', label: '금일 정산액', value: s?.todayRevenue ?? 0, format: 'currency', tone: 'warn', icon: 'won' },
      { key: 'riskyVehicles', label: '위험 차량', value: s?.riskyVehicles ?? 0, unit: '대', tone: 'danger', icon: 'wrench' },
    ]
  })

  /** 미정산 건수 — 사이드바 배지에도 쓰인다. */
  const unpaidCount = computed(() => summary.value?.unpaidCount ?? 0)

  /** 평균 추론 지연시간(ms) */
  const avgLatencyMs = computed(() => summary.value?.avgLatencyMs ?? 0)

  /** 최근 출입 이벤트 배열 */
  const recentEvents = computed(() => summary.value?.recentEvents ?? [])

  /**
   * 시간대별 교통량을 0~23시 전체로 채운다.
   * 서버가 데이터가 있는 시간만 보내더라도 차트 x축이 끊기지 않게 한다.
   */
  const hourlyTraffic = computed(() => {
    const raw = summary.value?.hourlyTraffic ?? []
    const byHour = new Map(raw.map((h) => [Number(h.hour), h]))
    return Array.from({ length: 24 }, (_, hour) => {
      const found = byHour.get(hour)
      return {
        hour,
        entries: found?.entries ?? 0,
        exits: found?.exits ?? 0,
      }
    })
  })

  /** LineChart 용 두 개의 시리즈(입차/출차) */
  const trafficSeries = computed(() => [
    {
      name: '입차',
      color: 'var(--c-chart-1)',
      values: hourlyTraffic.value.map((h) => h.entries),
    },
    {
      name: '출차',
      color: 'var(--c-chart-2)',
      values: hourlyTraffic.value.map((h) => h.exits),
    },
  ])

  /** 시간대 x축 라벨(0,3,6,... 만 표기하도록 뷰에서 간격을 조절한다) */
  const trafficLabels = computed(() =>
    hourlyTraffic.value.map((h) => String(h.hour).padStart(2, '0')),
  )

  /**
   * 위험도 분포를 DonutChart 용 조각 배열로 변환한다.
   * riskDistribution: {LOW, MEDIUM, HIGH, CRITICAL}
   */
  const riskSlices = computed(() => {
    const dist = summary.value?.riskDistribution ?? {}
    const colorMap = {
      LOW: 'var(--c-chart-2)',
      MEDIUM: 'var(--c-chart-3)',
      HIGH: 'var(--c-chart-4)',
      CRITICAL: '#7f1d1d',
    }
    const labelMap = { LOW: '낮음', MEDIUM: '보통', HIGH: '높음', CRITICAL: '심각' }
    return RISK_ORDER.map((key) => ({
      key,
      name: labelMap[key],
      value: dist[key] ?? 0,
      color: colorMap[key],
    }))
  })

  /** 위험도 분포 합계(= 예측 대상 차량 수) */
  const riskTotal = computed(() => riskSlices.value.reduce((a, s) => a + s.value, 0))

  // ---------------- 액션 ----------------
  /** 요약 지표를 다시 불러온다. */
  async function fetchSummary() {
    loading.value = true
    error.value = ''
    try {
      summary.value = await dashboardApi.getSummary()
      loadedAt.value = new Date()
    } catch (e) {
      error.value = extractMessage(e)
    } finally {
      loading.value = false
    }
  }

  /** 스토어 초기화(로그아웃 시). */
  function reset() {
    summary.value = null
    error.value = ''
    loadedAt.value = null
  }

  return {
    summary,
    loading,
    error,
    loadedAt,
    stats,
    unpaidCount,
    avgLatencyMs,
    recentEvents,
    hourlyTraffic,
    trafficSeries,
    trafficLabels,
    riskSlices,
    riskTotal,
    fetchSummary,
    reset,
  }
})
