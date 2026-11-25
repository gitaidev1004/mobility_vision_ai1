<script setup>
/**
 * 주차 · 정산 화면 (/parking)
 * - 상단: 주차장 현황(수용/점유/가용/점유율 + 시간대별 입출차 막대)
 * - 하단: 주차 세션 목록 + 정산 처리 모달
 * - 엔드포인트: GET /parking/status, GET /parking/sessions, POST /parking/sessions/{id}/pay
 */
import { computed, onMounted, reactive, ref } from 'vue'
import PageHeader from '@/components/common/PageHeader.vue'
import StatCard from '@/components/common/StatCard.vue'
import DataTable from '@/components/common/DataTable.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import PaginationBar from '@/components/common/PaginationBar.vue'
import AppIcon from '@/components/common/AppIcon.vue'
import BarChart from '@/components/charts/BarChart.vue'
import DonutChart from '@/components/charts/DonutChart.vue'
import * as parkingApi from '@/api/parking'
import { extractMessage } from '@/api/http'
import { useAuthStore } from '@/stores/auth'
import { useDashboardStore } from '@/stores/dashboard'
import {
  formatDateTime,
  formatCurrency,
  formatPercent,
  formatMinutes,
  formatNumber,
  PAYMENT_METHOD_LABEL,
} from '@/utils/format'
import { toastSuccess, toastError } from '@/utils/toast'

const auth = useAuthStore()
const dashboard = useDashboardStore()

/** 주차장 현황 */
const status = ref(null)
const statusLoading = ref(false)

/** 세션 목록 */
const sessions = ref([])
const page = ref(0)
const size = ref(20)
const totalElements = ref(0)
const totalPages = ref(0)
const listLoading = ref(false)
/** 'all' | 'active' 필터 */
const scope = ref('active')

/** 정산 모달 */
const payTarget = ref(null)
const paying = ref(false)
const payForm = reactive({ paymentMethod: 'CARD', discountAmount: 0 })

/** 표 열 정의 */
const columns = [
  { key: 'plateNumber', label: '번호판', width: '125px', cellClass: 'cell-plate' },
  { key: 'entryAt', label: '입차 시각', width: '145px', cellClass: 'cell-time' },
  { key: 'exitAt', label: '출차 시각', width: '145px', cellClass: 'cell-time' },
  { key: 'parkedMinutes', label: '주차 시간', width: '110px', align: 'right' },
  { key: 'feeAmount', label: '요금', width: '110px', align: 'right' },
  { key: 'discountAmount', label: '할인', width: '95px', align: 'right' },
  { key: 'paid', label: '정산', width: '100px', align: 'center' },
  { key: 'paymentMethod', label: '수단', width: '80px', align: 'center' },
  { key: 'actions', label: '처리', width: '90px', align: 'center' },
]

/** 결제 수단 후보 */
const paymentMethods = ['CARD', 'CASH', 'MOBILE', 'PASS']

/** 점유율 0~1 */
const occupancyRate = computed(() => status.value?.occupancyRate ?? 0)

/** 점유/가용 도넛 조각 */
const occupancySlices = computed(() => [
  { name: '점유', value: status.value?.occupied ?? 0, color: 'var(--c-chart-1)' },
  { name: '가용', value: status.value?.available ?? 0, color: '#dbe3ee' },
])

/** 시간대별 입출차 막대 데이터 */
const byHour = computed(() => {
  const raw = status.value?.byHour ?? []
  const map = new Map(raw.map((h) => [Number(h.hour), h]))
  return Array.from({ length: 24 }, (_, hour) => ({
    hour,
    entries: map.get(hour)?.entries ?? 0,
    exits: map.get(hour)?.exits ?? 0,
  }))
})

const byHourLabels = computed(() => byHour.value.map((h) => String(h.hour).padStart(2, '0')))

const byHourSeries = computed(() => [
  { name: '입차', color: 'var(--c-chart-1)', values: byHour.value.map((h) => h.entries) },
  { name: '출차', color: 'var(--c-chart-2)', values: byHour.value.map((h) => h.exits) },
])

/** 현재 페이지 기준 미정산 합계 금액 */
const unpaidAmount = computed(() =>
  sessions.value
    .filter((s) => !s.paid)
    .reduce((a, s) => a + (Number(s.feeAmount) || 0), 0),
)

/** 정산 모달의 최종 결제 금액 */
const payFinalAmount = computed(() => {
  const fee = Number(payTarget.value?.feeAmount) || 0
  const discount = Number(payForm.discountAmount) || 0
  return Math.max(0, fee - discount)
})

/** 주차장 현황 조회 */
async function loadStatus() {
  statusLoading.value = true
  try {
    status.value = await parkingApi.getStatus()
  } catch (e) {
    toastError(extractMessage(e))
  } finally {
    statusLoading.value = false
  }
}

/** 세션 목록 조회 */
async function loadSessions(nextPage = page.value) {
  listLoading.value = true
  try {
    const data = await parkingApi.listSessions({
      // 'active' 범위일 때만 active=true 를 보낸다.
      active: scope.value === 'active' ? true : undefined,
      page: nextPage,
      size: size.value,
    })
    sessions.value = data?.content ?? []
    totalElements.value = data?.totalElements ?? 0
    totalPages.value = data?.totalPages ?? 0
    page.value = data?.page ?? nextPage
  } catch (e) {
    toastError(extractMessage(e))
    sessions.value = []
    totalElements.value = 0
    totalPages.value = 0
  } finally {
    listLoading.value = false
  }
}

/** 범위 전환 */
function onScopeChange(next) {
  scope.value = next
  loadSessions(0)
}

/** 정산 모달 열기 */
function openPay(row) {
  payTarget.value = row
  payForm.paymentMethod = 'CARD'
  payForm.discountAmount = Number(row.discountAmount) || 0
}

/** 정산 모달 닫기 */
function closePay() {
  payTarget.value = null
  paying.value = false
}

/** 정산 실행 */
async function onPay() {
  if (!payTarget.value) return
  paying.value = true
  try {
    await parkingApi.paySession(payTarget.value.id, {
      paymentMethod: payForm.paymentMethod,
      discountAmount: Number(payForm.discountAmount) || 0,
    })
    toastSuccess(
      `${payTarget.value.plateNumber} ${formatCurrency(payFinalAmount.value)} 정산 완료`,
    )
    closePay()
    // 현황/목록/대시보드 지표를 모두 갱신한다.
    await Promise.all([loadStatus(), loadSessions(), dashboard.fetchSummary()])
  } catch (e) {
    toastError(extractMessage(e))
    paying.value = false
  }
}

onMounted(() => {
  loadStatus()
  loadSessions(0)
})
</script>

<template>
  <div class="parking">
    <PageHeader
      title="주차 · 정산"
      desc="주차장 점유 현황을 확인하고 주차 세션 요금을 정산합니다."
    >
      <template #actions>
        <button
          type="button"
          class="btn"
          aria-label="주차 현황 새로고침"
          :disabled="statusLoading"
          @click="loadStatus(); loadSessions()"
        >
          <AppIcon name="refresh" :size="16" />
          새로고침
        </button>
      </template>
    </PageHeader>

    <!-- ① 현황 카드 -->
    <div class="grid grid--stats">
      <StatCard
        label="총 수용 대수"
        :value="status?.capacity ?? 0"
        unit="면"
        tone="info"
        icon="parking"
        :loading="statusLoading && !status"
      />
      <StatCard
        label="현재 점유"
        :value="status?.occupied ?? 0"
        unit="대"
        tone="primary"
        icon="car"
        :loading="statusLoading && !status"
      />
      <StatCard
        label="잔여 면수"
        :value="status?.available ?? 0"
        unit="면"
        tone="success"
        icon="check"
        :loading="statusLoading && !status"
      />
      <StatCard
        label="미정산"
        :value="status?.unpaidCount ?? 0"
        unit="건"
        tone="warn"
        icon="receipt"
        :loading="statusLoading && !status"
      />
    </div>

    <!-- ② 점유율 도넛 + 시간대별 입출차 -->
    <div class="grid grid--side-main">
      <section class="card">
        <div class="card__head">
          <h3 class="card__title">점유율</h3>
          <span class="card__hint">{{ formatPercent(occupancyRate) }}</span>
        </div>
        <div class="card__body">
          <DonutChart
            :slices="occupancySlices"
            :size="190"
            :thickness="24"
            :center-value="formatPercent(occupancyRate, 0)"
            center-label="점유율"
          />
          <!-- 점유율 진행 바(도넛을 보조) -->
          <div class="meter parking__meter">
            <div
              class="meter__fill"
              :class="{
                'meter__fill--warn': occupancyRate >= 0.7 && occupancyRate < 0.9,
                'meter__fill--danger': occupancyRate >= 0.9,
              }"
              :style="{ width: `${Math.min(100, occupancyRate * 100)}%` }"
            />
          </div>
          <p class="parking__meter-note">
            {{ formatNumber(status?.occupied ?? 0) }}면 사용 중 /
            총 {{ formatNumber(status?.capacity ?? 0) }}면
          </p>
        </div>
      </section>

      <section class="card">
        <div class="card__head">
          <h3 class="card__title">시간대별 입·출차</h3>
          <span class="card__hint">금일 00~23시</span>
        </div>
        <div class="card__body">
          <BarChart
            :labels="byHourLabels"
            :series="byHourSeries"
            :height="230"
            :label-every="2"
            unit="건"
          />
        </div>
      </section>
    </div>

    <!-- ③ 세션 목록 -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">주차 세션</h3>
        <span class="card__hint">
          총 {{ formatNumber(totalElements) }}건
          <template v-if="unpaidAmount > 0">
            · 미정산 {{ formatCurrency(unpaidAmount) }}
          </template>
        </span>
        <div class="card__tools">
          <!-- 범위 토글 -->
          <div class="parking__toggle" role="group" aria-label="세션 범위 선택">
            <button
              type="button"
              class="parking__toggle-btn"
              :class="{ 'is-active': scope === 'active' }"
              @click="onScopeChange('active')"
            >
              주차 중
            </button>
            <button
              type="button"
              class="parking__toggle-btn"
              :class="{ 'is-active': scope === 'all' }"
              @click="onScopeChange('all')"
            >
              전체
            </button>
          </div>
        </div>
      </div>

      <div class="card__body card__body--flush">
        <DataTable
          caption="주차 세션 목록. 번호판, 입·출차 시각, 주차 시간, 요금, 할인, 정산 상태, 결제 수단을 보여줍니다."
          :columns="columns"
          :rows="sessions"
          :loading="listLoading"
          :empty-title="scope === 'active' ? '주차 중인 차량이 없습니다' : '주차 세션이 없습니다'"
          empty-desc="입차 이벤트가 발생하면 세션이 생성됩니다."
          empty-icon="parking"
          compact
        >
          <template #cell-entryAt="{ row }">
            {{ formatDateTime(row.entryAt) }}
          </template>
          <template #cell-exitAt="{ row }">
            <span v-if="row.exitAt">{{ formatDateTime(row.exitAt) }}</span>
            <StatusBadge v-else kind="parked" :value="true" />
          </template>
          <template #cell-parkedMinutes="{ row }">
            {{ row.parkedMinutes !== null && row.parkedMinutes !== undefined ? formatMinutes(row.parkedMinutes) : '-' }}
          </template>
          <template #cell-feeAmount="{ row }">
            {{ row.feeAmount !== null && row.feeAmount !== undefined ? formatCurrency(row.feeAmount) : '-' }}
          </template>
          <template #cell-discountAmount="{ row }">
            {{ Number(row.discountAmount) > 0 ? `-${formatCurrency(row.discountAmount)}` : '-' }}
          </template>
          <template #cell-paid="{ row }">
            <StatusBadge kind="paid" :value="row.paid === true" />
          </template>
          <template #cell-paymentMethod="{ row }">
            {{ row.paymentMethod ? (PAYMENT_METHOD_LABEL[row.paymentMethod] || row.paymentMethod) : '-' }}
          </template>
          <template #cell-actions="{ row }">
            <button
              type="button"
              class="btn btn--sm btn--primary"
              :disabled="row.paid === true || !auth.canWrite || !row.exitAt"
              :aria-label="`${row.plateNumber} 정산 처리`"
              :title="row.paid ? '이미 정산되었습니다' : !row.exitAt ? '출차 후 정산할 수 있습니다' : '정산'"
              @click.stop="openPay(row)"
            >
              정산
            </button>
          </template>
        </DataTable>
      </div>

      <div v-if="totalPages > 1" class="card__foot">
        <PaginationBar
          :page="page"
          :total-pages="totalPages"
          :total-elements="totalElements"
          :size="size"
          @change="loadSessions"
        />
      </div>
    </section>

    <!-- 정산 모달 -->
    <Teleport to="body">
      <div
        v-if="payTarget"
        class="modal-backdrop"
        role="dialog"
        aria-modal="true"
        aria-label="주차 요금 정산"
        @click.self="closePay"
      >
        <div class="modal modal--sm">
          <div class="modal__head">
            <h3 class="modal__title">주차 요금 정산</h3>
            <button type="button" class="modal__close" aria-label="닫기" @click="closePay">
              <AppIcon name="close" :size="16" />
            </button>
          </div>

          <div class="modal__body">
            <!-- 정산 대상 요약 -->
            <dl class="kv pay__summary">
              <dt>번호판</dt>
              <dd>{{ payTarget.plateNumber }}</dd>
              <dt>입차</dt>
              <dd>{{ formatDateTime(payTarget.entryAt) }}</dd>
              <dt>출차</dt>
              <dd>{{ payTarget.exitAt ? formatDateTime(payTarget.exitAt) : '-' }}</dd>
              <dt>주차 시간</dt>
              <dd>{{ formatMinutes(payTarget.parkedMinutes) }}</dd>
              <dt>기본 요금</dt>
              <dd>{{ formatCurrency(payTarget.feeAmount) }}</dd>
            </dl>

            <div class="pay__form">
              <div class="field">
                <label class="field__label" for="pay-method">결제 수단</label>
                <select id="pay-method" v-model="payForm.paymentMethod" class="select">
                  <option v-for="m in paymentMethods" :key="m" :value="m">
                    {{ PAYMENT_METHOD_LABEL[m] }}
                  </option>
                </select>
              </div>

              <div class="field">
                <label class="field__label" for="pay-discount">할인 금액(원)</label>
                <input
                  id="pay-discount"
                  v-model="payForm.discountAmount"
                  class="input"
                  type="number"
                  min="0"
                  :max="payTarget.feeAmount || 0"
                  step="100"
                />
                <span class="field__hint">정기권·할인권 적용 시 입력합니다.</span>
              </div>
            </div>

            <!-- 최종 금액 -->
            <div class="pay__total">
              <span>최종 결제 금액</span>
              <strong>{{ formatCurrency(payFinalAmount) }}</strong>
            </div>
          </div>

          <div class="modal__foot">
            <button type="button" class="btn" :disabled="paying" @click="closePay">취소</button>
            <button
              type="button"
              class="btn btn--success"
              :disabled="paying"
              @click="onPay"
            >
              <span v-if="paying" class="spinner" aria-hidden="true" />
              정산 처리
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.parking {
  display: flex;
  flex-direction: column;
  gap: var(--sp-5);
}

.parking__meter {
  margin-top: var(--sp-4);
}

.parking__meter-note {
  margin-top: var(--sp-2);
  font-size: var(--fs-sm);
  color: var(--c-text-muted);
  text-align: center;
  font-variant-numeric: tabular-nums;
}

/* 범위 토글 */
.parking__toggle {
  display: inline-flex;
  border: 1px solid var(--c-border-strong);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.parking__toggle-btn {
  height: 28px;
  padding: 0 var(--sp-3);
  border: 0;
  background: var(--c-surface);
  font-size: var(--fs-sm);
  font-weight: 600;
  color: var(--c-text-sub);
}

.parking__toggle-btn + .parking__toggle-btn {
  border-left: 1px solid var(--c-border);
}

.parking__toggle-btn.is-active {
  background: var(--c-primary);
  color: #fff;
}

/* 정산 모달 */
.pay__summary {
  padding-bottom: var(--sp-4);
  border-bottom: 1px solid var(--c-border);
}

.pay__form {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--sp-4);
  margin-top: var(--sp-4);
}

.pay__total {
  margin-top: var(--sp-5);
  padding: var(--sp-4);
  background: var(--c-surface-alt);
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: var(--fs-md);
  font-weight: 600;
  color: var(--c-text-sub);
}

.pay__total strong {
  font-size: var(--fs-xl);
  font-weight: 800;
  color: var(--c-text);
  font-variant-numeric: tabular-nums;
}

@media (max-width: 640px) {
  .pay__form {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
