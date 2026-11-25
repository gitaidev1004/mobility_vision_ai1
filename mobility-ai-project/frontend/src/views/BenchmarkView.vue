<script setup>
/**
 * 모델 성능 화면 (/benchmark)
 * - GET /api/v1/benchmarks 로 측정값을 불러와 비교표를 그린다.
 * - POST /api/v1/benchmarks/run 으로 재측정을 실행한다.
 * - 지연시간(p50/p95) 가로 막대그래프와 정확도 막대그래프를 함께 보여준다.
 * - 계약서에 benchmark 전용 스토어가 없으므로 이 화면에서 지역 상태로 관리한다.
 */
import { computed, onMounted, ref } from 'vue'
import PageHeader from '@/components/common/PageHeader.vue'
import StatCard from '@/components/common/StatCard.vue'
import DataTable from '@/components/common/DataTable.vue'
import AppIcon from '@/components/common/AppIcon.vue'
import BarChart from '@/components/charts/BarChart.vue'
import * as benchmarkApi from '@/api/benchmark'
import { extractMessage } from '@/api/http'
import { useAuthStore } from '@/stores/auth'
import { formatDateTime, formatDuration, formatPercent, formatNumber } from '@/utils/format'
import { toastSuccess, toastError } from '@/utils/toast'

const auth = useAuthStore()

/** 벤치마크 레코드 배열 */
const rows = ref([])
const loading = ref(false)
const running = ref(false)

/** 표 열 정의 (계약서 model_benchmark 컬럼) */
const columns = [
  { key: 'modelName', label: '모델', width: '210px' },
  { key: 'task', label: '태스크', width: '110px' },
  { key: 'detectRecall', label: '검출 재현율', width: '110px', align: 'right' },
  { key: 'exactMatchRate', label: '완전일치율', width: '110px', align: 'right' },
  { key: 'latencyP50Ms', label: 'p50 지연', width: '100px', align: 'right' },
  { key: 'latencyP95Ms', label: 'p95 지연', width: '100px', align: 'right' },
  { key: 'memoryMb', label: '메모리', width: '95px', align: 'right' },
  { key: 'modelSizeMb', label: '모델 크기', width: '100px', align: 'right' },
  { key: 'quantized', label: '양자화', width: '80px', align: 'center' },
  { key: 'measuredAt', label: '측정 시각', width: '140px', cellClass: 'cell-time' },
]

/**
 * 모델 이름이 길면 차트 라벨용으로 줄인다.
 * @param {string} name 원본 모델명
 * @param {number} max 최대 글자 수(차트 폭에 맞춰 호출부에서 조절)
 */
function shortName(name, max = 18) {
  if (!name) return '-'
  return name.length > max ? `${name.slice(0, max - 1)}…` : name
}

/** 지연시간 비교 막대그래프(p50 / p95 두 시리즈) */
const latencyChart = computed(() => ({
  labels: rows.value.map((r) => shortName(r.modelName)),
  series: [
    {
      name: 'p50 (ms)',
      color: 'var(--c-chart-1)',
      values: rows.value.map((r) => Number(r.latencyP50Ms) || 0),
    },
    {
      name: 'p95 (ms)',
      color: 'var(--c-chart-3)',
      values: rows.value.map((r) => Number(r.latencyP95Ms) || 0),
    },
  ],
}))

/** 정확도 비교 가로 막대그래프(재현율 / 완전일치율, % 단위) */
const accuracyChart = computed(() => ({
  labels: rows.value.map((r) => shortName(r.modelName, 26)),
  series: [
    {
      name: '검출 재현율 (%)',
      color: 'var(--c-chart-2)',
      values: rows.value.map((r) => (Number(r.detectRecall) || 0) * 100),
    },
    {
      name: '완전일치율 (%)',
      color: 'var(--c-chart-5)',
      values: rows.value.map((r) => (Number(r.exactMatchRate) || 0) * 100),
    },
  ],
}))

/** 모델 크기 비교(가로 막대). 라벨 영역 230px 에 맞춰 줄인다. */
const sizeChart = computed(() => ({
  labels: rows.value.map((r) => shortName(r.modelName, 26)),
  series: [
    {
      name: '모델 크기(MB)',
      color: 'var(--c-chart-6)',
      values: rows.value.map((r) => Number(r.modelSizeMb) || 0),
    },
  ],
}))

/** 완전일치율이 가장 높은 모델 */
const bestAccuracy = computed(() => {
  if (!rows.value.length) return null
  return rows.value.reduce((best, r) =>
    (Number(r.exactMatchRate) || 0) > (Number(best.exactMatchRate) || 0) ? r : best,
  )
})

/** p50 지연시간이 가장 낮은 모델 */
const fastest = computed(() => {
  if (!rows.value.length) return null
  return rows.value.reduce((best, r) =>
    (Number(r.latencyP50Ms) || Infinity) < (Number(best.latencyP50Ms) || Infinity) ? r : best,
  )
})

/** 모델 크기가 가장 작은 모델 */
const smallest = computed(() => {
  if (!rows.value.length) return null
  return rows.value.reduce((best, r) =>
    (Number(r.modelSizeMb) || Infinity) < (Number(best.modelSizeMb) || Infinity) ? r : best,
  )
})

/** 특정 행이 최고 성능인지 판단해 셀을 강조한다. */
function isBest(row, key) {
  if (key === 'exactMatchRate') return bestAccuracy.value?.id === row.id
  if (key === 'latencyP50Ms') return fastest.value?.id === row.id
  if (key === 'modelSizeMb') return smallest.value?.id === row.id
  return false
}

/** 목록 로드 */
async function load() {
  loading.value = true
  try {
    const data = await benchmarkApi.listBenchmarks()
    rows.value = Array.isArray(data) ? data : data?.content ?? []
  } catch (e) {
    toastError(extractMessage(e))
    rows.value = []
  } finally {
    loading.value = false
  }
}

/** 재측정 실행 */
async function onRun() {
  running.value = true
  try {
    const data = await benchmarkApi.runBenchmark()
    if (Array.isArray(data) && data.length) rows.value = data
    else await load()
    toastSuccess('벤치마크를 재측정했습니다.')
  } catch (e) {
    toastError(extractMessage(e))
  } finally {
    running.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="benchmark">
    <PageHeader
      title="모델 성능"
      desc="번호판 검출·판독 모델의 정확도와 추론 지연시간을 비교합니다."
    >
      <template #actions>
        <button
          type="button"
          class="btn"
          aria-label="목록 새로고침"
          :disabled="loading"
          @click="load"
        >
          <AppIcon name="refresh" :size="16" />
          새로고침
        </button>
        <button
          type="button"
          class="btn btn--primary"
          :disabled="running || !auth.canWrite"
          aria-label="벤치마크 재측정 실행"
          @click="onRun"
        >
          <span v-if="running" class="spinner" aria-hidden="true" />
          <AppIcon v-if="!running" name="chart" :size="16" />
          {{ running ? '측정 중…' : '재측정' }}
        </button>
      </template>
    </PageHeader>

    <!-- 최고 성능 요약 -->
    <div class="grid grid--3">
      <StatCard
        label="최고 정확도 모델"
        :value="bestAccuracy ? bestAccuracy.modelName : '-'"
        format="text"
        tone="success"
        icon="check"
        :loading="loading && !rows.length"
      />
      <StatCard
        label="최저 지연시간"
        :value="fastest?.latencyP50Ms ?? 0"
        format="duration"
        tone="primary"
        icon="clock"
        :loading="loading && !rows.length"
      />
      <StatCard
        label="최소 모델 크기"
        :value="smallest?.modelSizeMb ?? 0"
        unit="MB"
        tone="info"
        icon="download"
        :loading="loading && !rows.length"
      />
    </div>

    <!-- 비교표 -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">모델 성능 비교표</h3>
        <span class="card__hint">{{ formatNumber(rows.length) }}개 모델</span>
      </div>

      <div class="card__body card__body--flush">
        <DataTable
          caption="모델 성능 비교표. 모델명, 태스크, 검출 재현율, 완전일치율, p50·p95 지연시간, 메모리, 모델 크기, 양자화 여부, 측정 시각을 보여줍니다."
          :columns="columns"
          :rows="rows"
          :loading="loading"
          empty-title="측정된 벤치마크가 없습니다"
          empty-desc="재측정 버튼을 눌러 AI 서비스에서 성능을 측정하세요."
          empty-icon="chart"
          compact
        >
          <template #cell-modelName="{ row }">
            <span class="benchmark__model">{{ row.modelName }}</span>
          </template>
          <template #cell-task="{ row }">
            <span class="badge badge--neutral badge--plain">{{ row.task }}</span>
          </template>
          <template #cell-detectRecall="{ row }">
            {{ formatPercent(row.detectRecall) }}
          </template>
          <template #cell-exactMatchRate="{ row }">
            <span :class="{ 'is-best': isBest(row, 'exactMatchRate') }">
              {{ formatPercent(row.exactMatchRate) }}
            </span>
          </template>
          <template #cell-latencyP50Ms="{ row }">
            <span :class="{ 'is-best': isBest(row, 'latencyP50Ms') }">
              {{ formatDuration(row.latencyP50Ms) }}
            </span>
          </template>
          <template #cell-latencyP95Ms="{ row }">
            {{ formatDuration(row.latencyP95Ms) }}
          </template>
          <template #cell-memoryMb="{ row }">
            {{ formatNumber(row.memoryMb, 1) }}MB
          </template>
          <template #cell-modelSizeMb="{ row }">
            <span :class="{ 'is-best': isBest(row, 'modelSizeMb') }">
              {{ formatNumber(row.modelSizeMb, 1) }}MB
            </span>
          </template>
          <template #cell-quantized="{ row }">
            <span v-if="row.quantized" class="badge badge--info">INT8</span>
            <span v-else class="cell-muted">FP32</span>
          </template>
          <template #cell-measuredAt="{ row }">
            {{ formatDateTime(row.measuredAt) }}
          </template>
        </DataTable>
      </div>
    </section>

    <!-- 지연시간 막대그래프 -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">추론 지연시간 비교</h3>
        <span class="card__hint">p50 / p95 (ms, 낮을수록 좋음)</span>
      </div>
      <div class="card__body">
        <BarChart
          :labels="latencyChart.labels"
          :series="latencyChart.series"
          :width="900"
          :height="260"
          :ticks="4"
          show-values
          :value-format="(v) => v.toFixed(0)"
          unit="ms"
        />
      </div>
    </section>

    <!-- 정확도 (가로 막대: 모델명이 길어 세로축에 두는 편이 읽기 쉽다) -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">정확도 비교</h3>
        <span class="card__hint">검출 재현율 / 완전일치율 (%, 높을수록 좋음)</span>
      </div>
      <div class="card__body">
        <BarChart
          :labels="accuracyChart.labels"
          :series="accuracyChart.series"
          :width="1180"
          :height="280"
          :max-value="100"
          :ticks="4"
          :label-width="230"
          horizontal
          show-values
          :value-format="(v) => v.toFixed(1)"
          unit="%"
        />
      </div>
    </section>

    <!-- 모델 크기 -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">모델 크기</h3>
        <span class="card__hint">MB (작을수록 배포에 유리)</span>
      </div>
      <div class="card__body">
        <BarChart
          :labels="sizeChart.labels"
          :series="sizeChart.series"
          :width="1180"
          :height="230"
          :ticks="4"
          :label-width="230"
          horizontal
          show-values
          :value-format="(v) => v.toFixed(1)"
          :legend="false"
          unit="MB"
        />
      </div>
    </section>
  </div>
</template>

<style scoped>
.benchmark {
  display: flex;
  flex-direction: column;
  gap: var(--sp-5);
}

.benchmark__model {
  font-weight: 700;
}

/* 최고 성능 값 강조 */
.is-best {
  color: var(--c-success);
  font-weight: 800;
}

.is-best::after {
  content: ' ★';
  font-size: 10px;
}
</style>
