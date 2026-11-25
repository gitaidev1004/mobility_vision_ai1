<script setup>
/**
 * 대시보드 화면 (/)
 * - GET /api/v1/dashboard/summary 한 번으로 모든 위젯을 채운다.
 * - 구성: 통계 카드 8개 → 시간대별 교통량(LineChart) + 위험도 분포(DonutChart) → 최근 이벤트 표
 * - 자동 갱신 주기는 VITE_DASHBOARD_REFRESH_MS 로 조절한다(0 이면 끔).
 */
import { computed, onMounted, onBeforeUnmount } from 'vue'
import PageHeader from '@/components/common/PageHeader.vue'
import StatCard from '@/components/common/StatCard.vue'
import DataTable from '@/components/common/DataTable.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import AppIcon from '@/components/common/AppIcon.vue'
import LineChart from '@/components/charts/LineChart.vue'
import DonutChart from '@/components/charts/DonutChart.vue'
import BarChart from '@/components/charts/BarChart.vue'
import { useDashboardStore } from '@/stores/dashboard'
import { formatDateTime, formatDuration, formatPercent, formatNumber } from '@/utils/format'
import { toastError } from '@/utils/toast'

const dashboard = useDashboardStore()

/** 자동 갱신 타이머 */
let timer = null

/** 최근 이벤트 표 열 정의 */
const eventColumns = [
  { key: 'plateNumber', label: '번호판', width: '130px', cellClass: 'cell-plate' },
  { key: 'eventType', label: '구분', width: '80px', align: 'center' },
  { key: 'gateName', label: '게이트', width: '110px' },
  { key: 'occurredAt', label: '발생 시각', width: '150px', cellClass: 'cell-time' },
  { key: 'allowed', label: '처리', width: '90px', align: 'center' },
]

/** 시간대별 입차 합계(차트 요약 문구용) */
const totalEntries = computed(() =>
  dashboard.hourlyTraffic.reduce((a, h) => a + h.entries, 0),
)

/** 시간대별 출차 합계 */
const totalExits = computed(() => dashboard.hourlyTraffic.reduce((a, h) => a + h.exits, 0))

/** 가장 혼잡한 시간대 */
const peakHour = computed(() => {
  let best = null
  dashboard.hourlyTraffic.forEach((h) => {
    const sum = h.entries + h.exits
    if (!best || sum > best.sum) best = { hour: h.hour, sum }
  })
  return best
})

/**
 * 시간대별 통행량 막대그래프용 시리즈(입차/출차 합계).
 * 꺾은선과 함께 보여주면 지면에서 비교가 쉽다.
 */
const trafficBarSeries = computed(() => [
  {
    name: '통행량',
    color: 'var(--c-chart-1)',
    values: dashboard.hourlyTraffic.map((h) => h.entries + h.exits),
  },
])

/** 인식 품질 요약 카드용 수치 */
const recognitionRate = computed(() => dashboard.summary?.recognitionRate ?? 0)
const failureCount = computed(() => dashboard.summary?.failureCount ?? 0)

/** 요약 로드. 오류는 토스트로 알린다. */
async function load() {
  await dashboard.fetchSummary()
  if (dashboard.error) toastError(dashboard.error)
}

onMounted(() => {
  load()
  const interval = Number(import.meta.env?.VITE_DASHBOARD_REFRESH_MS ?? 30000)
  if (interval > 0) {
    timer = setInterval(() => dashboard.fetchSummary(), interval)
  }
})

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div class="dashboard">
    <PageHeader
      title="운영 현황"
      desc="번호판 인식 · 출입 · 주차 · 예지보전 지표를 한눈에 확인합니다."
    >
      <template #actions>
        <span v-if="dashboard.loadedAt" class="dashboard__updated">
          갱신 {{ formatDateTime(dashboard.loadedAt) }}
        </span>
        <button
          type="button"
          class="btn"
          aria-label="대시보드 새로고침"
          :disabled="dashboard.loading"
          @click="load"
        >
          <AppIcon name="refresh" :size="16" />
          새로고침
        </button>
      </template>
    </PageHeader>

    <!-- ① 통계 카드 -->
    <div class="grid grid--stats dashboard__stats">
      <StatCard
        v-for="s in dashboard.stats"
        :key="s.key"
        :label="s.label"
        :value="s.value"
        :unit="s.unit"
        :format="s.format || 'number'"
        :tone="s.tone"
        :icon="s.icon"
        :loading="dashboard.loading && !dashboard.summary"
      />
    </div>

    <!-- ② 시간대별 교통량 + 위험도 도넛 -->
    <div class="grid grid--main-side dashboard__charts">
      <section class="card">
        <div class="card__head">
          <h3 class="card__title">시간대별 교통량</h3>
          <span class="card__hint">
            입차 {{ formatNumber(totalEntries) }}건 · 출차 {{ formatNumber(totalExits) }}건
            <template v-if="peakHour">· 최다 {{ peakHour.hour }}시</template>
          </span>
        </div>
        <div class="card__body">
          <!-- 꺾은선: 입차/출차 두 시리즈 -->
          <LineChart
            :labels="dashboard.trafficLabels"
            :series="dashboard.trafficSeries"
            :height="230"
            :label-every="3"
            area
            dots
            unit="건"
          />

          <!-- 막대: 시간대별 총 통행량 -->
          <div class="dashboard__subchart">
            <div class="dashboard__subchart-title">시간대 총 통행량</div>
            <BarChart
              :labels="dashboard.trafficLabels"
              :series="trafficBarSeries"
              :height="130"
              :label-every="3"
              :ticks="3"
              :legend="false"
              unit="건"
            />
          </div>
        </div>
      </section>

      <section class="card">
        <div class="card__head">
          <h3 class="card__title">차량 위험도 분포</h3>
          <span class="card__hint">예지보전 기준</span>
        </div>
        <div class="card__body">
          <DonutChart
            :slices="dashboard.riskSlices"
            :size="200"
            :thickness="24"
            center-label="예측 차량"
          />

          <!-- 추론 성능 요약 -->
          <dl class="kv dashboard__kv">
            <dt>인식 성공률</dt>
            <dd>{{ formatPercent(recognitionRate) }}</dd>
            <dt>인식 실패</dt>
            <dd>{{ formatNumber(failureCount) }}건</dd>
            <dt>평균 추론 지연</dt>
            <dd>{{ formatDuration(dashboard.avgLatencyMs) }}</dd>
            <dt>미정산 건수</dt>
            <dd>{{ formatNumber(dashboard.unpaidCount) }}건</dd>
          </dl>
        </div>
      </section>
    </div>

    <!-- ③ 최근 출입 이벤트 -->
    <section class="card dashboard__events">
      <div class="card__head">
        <h3 class="card__title">최근 출입 이벤트</h3>
        <span class="card__hint">최신 {{ dashboard.recentEvents.length }}건</span>
        <div class="card__tools">
          <RouterLink to="/events" class="btn btn--sm">전체 보기</RouterLink>
        </div>
      </div>

      <div class="card__body card__body--flush">
        <DataTable
          caption="최근 출입 이벤트 목록. 번호판, 구분, 게이트, 발생 시각, 처리 결과를 보여줍니다."
          :columns="eventColumns"
          :rows="dashboard.recentEvents"
          :loading="dashboard.loading && !dashboard.summary"
          empty-title="출입 이벤트가 없습니다"
          empty-desc="번호판 인식을 수행하면 이벤트가 기록됩니다."
          empty-icon="gate"
          compact
        >
          <template #cell-eventType="{ row }">
            <StatusBadge kind="event" :value="row.eventType" />
          </template>
          <template #cell-occurredAt="{ row }">
            {{ formatDateTime(row.occurredAt) }}
          </template>
          <template #cell-allowed="{ row }">
            <StatusBadge kind="allowed" :value="row.allowed" />
          </template>
        </DataTable>
      </div>
    </section>
  </div>
</template>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: var(--sp-5);
}

.dashboard__updated {
  font-size: var(--fs-sm);
  color: var(--c-text-muted);
  font-variant-numeric: tabular-nums;
}

.dashboard__stats,
.dashboard__charts {
  /* .grid 가 이미 gap 을 주므로 추가 마진은 두지 않는다. */
  margin: 0;
}

.dashboard__subchart {
  margin-top: var(--sp-5);
  padding-top: var(--sp-4);
  border-top: 1px dashed var(--c-border);
}

.dashboard__subchart-title {
  font-size: var(--fs-sm);
  font-weight: 700;
  color: var(--c-text-sub);
  margin-bottom: var(--sp-2);
}

.dashboard__kv {
  margin-top: var(--sp-5);
  padding-top: var(--sp-4);
  border-top: 1px solid var(--c-border);
}
</style>
