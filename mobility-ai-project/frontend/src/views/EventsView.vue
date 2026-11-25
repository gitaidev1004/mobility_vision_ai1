<script setup>
/**
 * 출입 이벤트 화면 (/events)
 * - GET /api/v1/events?type=&from=&to=&page=&size= 로 필터 조회한다.
 * - 날짜 입력은 <input type="date"> 를 쓰고, 백엔드로는 ISO LocalDateTime 으로 변환해 보낸다.
 * - 계약서에 events 전용 스토어가 없으므로 이 화면에서 지역 상태로 관리한다.
 */
import { computed, onMounted, reactive, ref } from 'vue'
import PageHeader from '@/components/common/PageHeader.vue'
import DataTable from '@/components/common/DataTable.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import PaginationBar from '@/components/common/PaginationBar.vue'
import AppIcon from '@/components/common/AppIcon.vue'
import BarChart from '@/components/charts/BarChart.vue'
import * as eventApi from '@/api/event'
import { extractMessage } from '@/api/http'
import { formatDateTime, formatPercent, formatNumber } from '@/utils/format'
import { toastError } from '@/utils/toast'

/** 필터 상태 */
const filters = reactive({
  type: '',
  from: '',
  to: '',
})

/** 목록 상태 */
const rows = ref([])
const page = ref(0)
const size = ref(20)
const totalElements = ref(0)
const totalPages = ref(0)
const loading = ref(false)

/** 표 열 정의 */
const columns = [
  { key: 'id', label: 'ID', width: '70px', align: 'right' },
  { key: 'plateNumber', label: '번호판', width: '130px', cellClass: 'cell-plate' },
  { key: 'eventType', label: '구분', width: '80px', align: 'center' },
  { key: 'gateName', label: '게이트', width: '110px' },
  { key: 'occurredAt', label: '발생 시각', width: '150px', cellClass: 'cell-time' },
  { key: 'confidence', label: '신뢰도', width: '90px', align: 'right' },
  { key: 'allowed', label: '처리', width: '90px', align: 'center' },
  { key: 'rejectReason', label: '거부 사유', cellClass: 'cell-muted' },
]

/** 현재 페이지 기준 게이트별 집계(간이 통계 막대그래프) */
const gateStats = computed(() => {
  const map = new Map()
  rows.value.forEach((r) => {
    const g = r.gateName || '미지정'
    if (!map.has(g)) map.set(g, { entries: 0, exits: 0 })
    const bucket = map.get(g)
    if (r.eventType === 'EXIT') bucket.exits += 1
    else bucket.entries += 1
  })
  const labels = [...map.keys()]
  return {
    labels,
    series: [
      { name: '입차', color: 'var(--c-chart-1)', values: labels.map((l) => map.get(l).entries) },
      { name: '출차', color: 'var(--c-chart-2)', values: labels.map((l) => map.get(l).exits) },
    ],
  }
})

/** 현재 페이지의 허용/거부 집계 */
const allowStats = computed(() => {
  const allowed = rows.value.filter((r) => r.allowed === true).length
  const rejected = rows.value.length - allowed
  return { allowed, rejected }
})

/**
 * 'YYYY-MM-DD' 문자열을 LocalDateTime 으로 바꾼다.
 * from 은 그날 00:00:00, to 는 그날 23:59:59 로 보정해 하루 전체를 포함시킨다.
 */
function toParam(dateStr, endOfDay = false) {
  if (!dateStr) return undefined
  return `${dateStr}T${endOfDay ? '23:59:59' : '00:00:00'}`
}

/** 목록 조회 */
async function load(nextPage = page.value) {
  loading.value = true
  try {
    const data = await eventApi.listEvents({
      type: filters.type || undefined,
      from: toParam(filters.from, false),
      to: toParam(filters.to, true),
      page: nextPage,
      size: size.value,
    })
    rows.value = data?.content ?? []
    totalElements.value = data?.totalElements ?? 0
    totalPages.value = data?.totalPages ?? 0
    page.value = data?.page ?? nextPage
  } catch (e) {
    toastError(extractMessage(e))
    rows.value = []
    totalElements.value = 0
    totalPages.value = 0
  } finally {
    loading.value = false
  }
}

/** 필터 적용(1페이지부터) */
function onApply() {
  load(0)
}

/** 필터 초기화 */
function onReset() {
  filters.type = ''
  filters.from = ''
  filters.to = ''
  load(0)
}

/** 오늘 하루로 범위 지정하는 단축 버튼 */
function setToday() {
  const d = new Date()
  const iso = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
  filters.from = iso
  filters.to = iso
  load(0)
}

/** 최근 7일 */
function setWeek() {
  const to = new Date()
  const from = new Date(to.getTime() - 6 * 24 * 3600 * 1000)
  const fmt = (d) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
  filters.from = fmt(from)
  filters.to = fmt(to)
  load(0)
}

onMounted(() => load(0))
</script>

<template>
  <div class="events">
    <PageHeader
      title="출입 이벤트"
      desc="게이트별 입·출차 이벤트와 허용/거부 처리 결과를 조회합니다."
    >
      <template #actions>
        <button type="button" class="btn btn--sm" @click="setToday">오늘</button>
        <button type="button" class="btn btn--sm" @click="setWeek">최근 7일</button>
      </template>
    </PageHeader>

    <!-- 필터 -->
    <section class="card">
      <div class="card__body">
        <form class="filter-bar" @submit.prevent="onApply">
          <div class="field">
            <label class="field__label" for="ev-type">구분</label>
            <select id="ev-type" v-model="filters.type" class="select">
              <option value="">전체</option>
              <option value="ENTRY">입차</option>
              <option value="EXIT">출차</option>
            </select>
          </div>

          <div class="field">
            <label class="field__label" for="ev-from">시작일</label>
            <input id="ev-from" v-model="filters.from" class="input" type="date" />
          </div>

          <div class="field">
            <label class="field__label" for="ev-to">종료일</label>
            <input id="ev-to" v-model="filters.to" class="input" type="date" />
          </div>

          <button type="submit" class="btn btn--primary" aria-label="필터 적용">
            <AppIcon name="search" :size="16" />
            조회
          </button>
          <button type="button" class="btn" aria-label="필터 초기화" @click="onReset">
            초기화
          </button>

          <!-- 현재 페이지 요약 -->
          <div class="events__summary">
            <span class="badge badge--success">허용 {{ allowStats.allowed }}</span>
            <span class="badge badge--danger">거부 {{ allowStats.rejected }}</span>
          </div>
        </form>
      </div>
    </section>

    <!-- 게이트별 집계 -->
    <section v-if="gateStats.labels.length" class="card">
      <div class="card__head">
        <h3 class="card__title">게이트별 입·출차 (현재 페이지)</h3>
        <span class="card__hint">{{ rows.length }}건 기준</span>
      </div>
      <div class="card__body">
        <BarChart
          :labels="gateStats.labels"
          :series="gateStats.series"
          :height="170"
          :ticks="3"
          show-values
          unit="건"
        />
      </div>
    </section>

    <!-- 목록 -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">이벤트 목록</h3>
        <span class="card__hint">총 {{ formatNumber(totalElements) }}건</span>
        <div class="card__tools">
          <button
            type="button"
            class="btn btn--sm"
            aria-label="목록 새로고침"
            @click="load()"
          >
            <AppIcon name="refresh" :size="15" />
          </button>
        </div>
      </div>

      <div class="card__body card__body--flush">
        <DataTable
          caption="출입 이벤트 목록. ID, 번호판, 구분, 게이트, 발생 시각, 신뢰도, 처리 결과, 거부 사유를 보여줍니다."
          :columns="columns"
          :rows="rows"
          :loading="loading"
          empty-title="조회된 이벤트가 없습니다"
          empty-desc="기간이나 구분 조건을 바꿔 다시 조회해 보세요."
          empty-icon="gate"
          compact
        >
          <template #cell-eventType="{ row }">
            <StatusBadge kind="event" :value="row.eventType" />
          </template>
          <template #cell-occurredAt="{ row }">
            {{ formatDateTime(row.occurredAt) }}
          </template>
          <template #cell-confidence="{ row }">
            {{ row.confidence !== null && row.confidence !== undefined ? formatPercent(row.confidence) : '-' }}
          </template>
          <template #cell-allowed="{ row }">
            <StatusBadge kind="allowed" :value="row.allowed" />
          </template>
          <template #cell-rejectReason="{ row }">
            {{ row.rejectReason || '-' }}
          </template>
        </DataTable>
      </div>

      <div v-if="totalPages > 1" class="card__foot">
        <PaginationBar
          :page="page"
          :total-pages="totalPages"
          :total-elements="totalElements"
          :size="size"
          @change="load"
        />
      </div>
    </section>
  </div>
</template>

<style scoped>
.events {
  display: flex;
  flex-direction: column;
  gap: var(--sp-5);
}

.events__summary {
  margin-left: auto;
  display: flex;
  gap: var(--sp-2);
  align-items: center;
}
</style>
