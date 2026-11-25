<script setup>
/**
 * 정비 이력 화면 (/maintenance)
 * - GET /api/v1/maintenance?vehicleId=&page=&size= 조회, POST /api/v1/maintenance 등록
 * - 차량 선택 드롭다운은 GET /vehicles 에서 채운다.
 * - 현재 페이지 기준 항목별 비용 집계를 가로 막대그래프로 보여준다.
 */
import { computed, onMounted, reactive, ref } from 'vue'
import PageHeader from '@/components/common/PageHeader.vue'
import StatCard from '@/components/common/StatCard.vue'
import DataTable from '@/components/common/DataTable.vue'
import PaginationBar from '@/components/common/PaginationBar.vue'
import AppIcon from '@/components/common/AppIcon.vue'
import BarChart from '@/components/charts/BarChart.vue'
import * as maintenanceApi from '@/api/maintenance'
import * as vehicleApi from '@/api/vehicle'
import { extractMessage } from '@/api/http'
import { useAuthStore } from '@/stores/auth'
import { formatDate, formatCurrency, formatKm, formatNumber } from '@/utils/format'
import { toastSuccess, toastError } from '@/utils/toast'

const auth = useAuthStore()

/** 차량 선택 목록 */
const vehicles = ref([])
/** 선택된 차량 ID('' 이면 전체) */
const vehicleId = ref('')

/** 목록 상태 */
const rows = ref([])
const page = ref(0)
const size = ref(20)
const totalElements = ref(0)
const totalPages = ref(0)
const loading = ref(false)

/** 등록 모달 */
const modalOpen = ref(false)
const saving = ref(false)

/** 정비 항목 후보 */
const items = [
  '엔진오일 교환',
  '브레이크 패드 교체',
  '타이어 교체',
  '배터리 교체',
  '에어컨 필터 교환',
  '변속기 오일 교환',
  '정기 점검',
  '냉각수 보충',
]

/** 등록 폼. maintenance_record 컬럼과 1:1 대응한다. */
const form = reactive({
  vehicleId: '',
  serviceDate: new Date().toISOString().slice(0, 10),
  odometerKm: '',
  item: '엔진오일 교환',
  partName: '',
  cost: '',
  shopName: '',
  memo: '',
})

/** 표 열 정의 */
const columns = [
  { key: 'serviceDate', label: '정비일', width: '110px', cellClass: 'cell-time' },
  { key: 'plateNumber', label: '번호판', width: '125px', cellClass: 'cell-plate' },
  { key: 'item', label: '정비 항목', width: '160px' },
  { key: 'partName', label: '부품', width: '140px' },
  { key: 'odometerKm', label: '주행거리', width: '110px', align: 'right' },
  { key: 'cost', label: '비용', width: '110px', align: 'right' },
  { key: 'shopName', label: '정비소', width: '140px' },
  { key: 'memo', label: '메모', cellClass: 'cell-muted' },
]

/** 현재 페이지 총 비용 */
const totalCost = computed(() => rows.value.reduce((a, r) => a + (Number(r.cost) || 0), 0))

/** 현재 페이지 평균 비용 */
const avgCost = computed(() => (rows.value.length ? totalCost.value / rows.value.length : 0))

/** 항목별 비용 집계(가로 막대그래프용, 상위 6개) */
const costByItem = computed(() => {
  const map = new Map()
  rows.value.forEach((r) => {
    const key = r.item || '기타'
    map.set(key, (map.get(key) || 0) + (Number(r.cost) || 0))
  })
  const sorted = [...map.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)
  return {
    labels: sorted.map((e) => e[0]),
    series: [
      { name: '비용(원)', color: 'var(--c-chart-3)', values: sorted.map((e) => e[1]) },
    ],
  }
})

/** 차량 목록 로드 */
async function loadVehicles() {
  try {
    const data = await vehicleApi.listVehicles({ page: 0, size: 100 })
    vehicles.value = data?.content ?? []
  } catch (e) {
    toastError(extractMessage(e))
  }
}

/** 정비 이력 로드 */
async function load(nextPage = page.value) {
  loading.value = true
  try {
    const data = await maintenanceApi.listMaintenance({
      vehicleId: vehicleId.value || undefined,
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

/** 차량 필터 변경 */
function onVehicleChange() {
  load(0)
}

/** 등록 모달 열기 */
function openCreate() {
  form.vehicleId = vehicleId.value || vehicles.value[0]?.id || ''
  form.serviceDate = new Date().toISOString().slice(0, 10)
  form.odometerKm = ''
  form.item = '엔진오일 교환'
  form.partName = ''
  form.cost = ''
  form.shopName = ''
  form.memo = ''
  modalOpen.value = true
}

/** 등록 저장 */
async function onSave() {
  if (!form.vehicleId) {
    toastError('차량을 선택하세요.')
    return
  }
  saving.value = true
  try {
    await maintenanceApi.createMaintenance({
      vehicleId: Number(form.vehicleId),
      serviceDate: form.serviceDate,
      odometerKm: form.odometerKm ? Number(form.odometerKm) : null,
      item: form.item,
      partName: form.partName.trim() || null,
      cost: form.cost ? Number(form.cost) : 0,
      shopName: form.shopName.trim() || null,
      memo: form.memo.trim() || null,
    })
    toastSuccess('정비 이력을 등록했습니다.')
    modalOpen.value = false
    await load(0)
  } catch (e) {
    toastError(extractMessage(e))
  } finally {
    saving.value = false
  }
}

onMounted(() => {
  loadVehicles()
  load(0)
})
</script>

<template>
  <div class="maintenance">
    <PageHeader
      title="정비 이력"
      desc="차량별 정비 내역과 비용을 기록하고 조회합니다."
    >
      <template #actions>
        <button
          type="button"
          class="btn btn--primary"
          :disabled="!auth.canWrite"
          aria-label="정비 이력 등록"
          @click="openCreate"
        >
          <AppIcon name="plus" :size="16" />
          정비 등록
        </button>
      </template>
    </PageHeader>

    <!-- 요약 카드 -->
    <div class="grid grid--3">
      <StatCard
        label="조회 건수"
        :value="totalElements"
        unit="건"
        tone="primary"
        icon="wrench"
        :loading="loading && !rows.length"
      />
      <StatCard
        label="현재 페이지 총 비용"
        :value="totalCost"
        format="currency"
        tone="warn"
        icon="won"
        :loading="loading && !rows.length"
      />
      <StatCard
        label="건당 평균 비용"
        :value="avgCost"
        format="currency"
        tone="info"
        icon="receipt"
        :loading="loading && !rows.length"
      />
    </div>

    <!-- 항목별 비용 -->
    <section v-if="costByItem.labels.length" class="card">
      <div class="card__head">
        <h3 class="card__title">정비 항목별 비용 (현재 페이지)</h3>
        <span class="card__hint">상위 {{ costByItem.labels.length }}개 항목</span>
      </div>
      <div class="card__body">
        <BarChart
          :labels="costByItem.labels"
          :series="costByItem.series"
          :width="720"
          :height="200"
          :ticks="4"
          horizontal
          show-values
          :value-format="(v) => formatCurrency(v, false)"
          :legend="false"
          unit="원"
        />
      </div>
    </section>

    <!-- 목록 -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">정비 내역</h3>
        <span class="card__hint">총 {{ formatNumber(totalElements) }}건</span>
        <div class="card__tools">
          <div class="field maintenance__filter">
            <label class="sr-only" for="mt-vehicle">차량 선택</label>
            <select
              id="mt-vehicle"
              v-model="vehicleId"
              class="select"
              @change="onVehicleChange"
            >
              <option value="">전체 차량</option>
              <option v-for="v in vehicles" :key="v.id" :value="v.id">
                {{ v.plateNumber }} · {{ v.ownerName }}
              </option>
            </select>
          </div>
          <button
            type="button"
            class="btn btn--sm"
            aria-label="정비 이력 새로고침"
            @click="load()"
          >
            <AppIcon name="refresh" :size="15" />
          </button>
        </div>
      </div>

      <div class="card__body card__body--flush">
        <DataTable
          caption="정비 이력 목록. 정비일, 번호판, 정비 항목, 부품, 주행거리, 비용, 정비소, 메모를 보여줍니다."
          :columns="columns"
          :rows="rows"
          :loading="loading"
          empty-title="정비 이력이 없습니다"
          empty-desc="정비 등록 버튼으로 첫 내역을 추가하세요."
          empty-icon="wrench"
          compact
        >
          <template #cell-serviceDate="{ row }">
            {{ formatDate(row.serviceDate) }}
          </template>
          <template #cell-odometerKm="{ row }">
            {{ row.odometerKm ? formatKm(row.odometerKm) : '-' }}
          </template>
          <template #cell-cost="{ row }">
            {{ formatCurrency(row.cost) }}
          </template>
          <template #cell-memo="{ row }">
            {{ row.memo || '-' }}
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

    <!-- 등록 모달 -->
    <Teleport to="body">
      <div
        v-if="modalOpen"
        class="modal-backdrop"
        role="dialog"
        aria-modal="true"
        aria-label="정비 이력 등록"
        @click.self="modalOpen = false"
      >
        <div class="modal">
          <div class="modal__head">
            <h3 class="modal__title">정비 이력 등록</h3>
            <button
              type="button"
              class="modal__close"
              aria-label="닫기"
              @click="modalOpen = false"
            >
              <AppIcon name="close" :size="16" />
            </button>
          </div>

          <form class="modal__body" @submit.prevent="onSave">
            <div class="form-grid">
              <div class="field">
                <label class="field__label" for="mf-vehicle">
                  차량<span class="req" aria-hidden="true">*</span>
                </label>
                <select id="mf-vehicle" v-model="form.vehicleId" class="select" required>
                  <option value="" disabled>차량을 선택하세요</option>
                  <option v-for="v in vehicles" :key="v.id" :value="v.id">
                    {{ v.plateNumber }} · {{ v.ownerName }}
                  </option>
                </select>
              </div>

              <div class="field">
                <label class="field__label" for="mf-date">
                  정비일<span class="req" aria-hidden="true">*</span>
                </label>
                <input
                  id="mf-date"
                  v-model="form.serviceDate"
                  class="input"
                  type="date"
                  required
                />
              </div>

              <div class="field">
                <label class="field__label" for="mf-item">정비 항목</label>
                <select id="mf-item" v-model="form.item" class="select">
                  <option v-for="i in items" :key="i" :value="i">{{ i }}</option>
                </select>
              </div>

              <div class="field">
                <label class="field__label" for="mf-part">부품명</label>
                <input
                  id="mf-part"
                  v-model="form.partName"
                  class="input"
                  type="text"
                  placeholder="0W-20 합성유 5L"
                />
              </div>

              <div class="field">
                <label class="field__label" for="mf-odo">주행거리(km)</label>
                <input
                  id="mf-odo"
                  v-model="form.odometerKm"
                  class="input"
                  type="number"
                  min="0"
                  placeholder="84210"
                />
              </div>

              <div class="field">
                <label class="field__label" for="mf-cost">비용(원)</label>
                <input
                  id="mf-cost"
                  v-model="form.cost"
                  class="input"
                  type="number"
                  min="0"
                  step="1000"
                  placeholder="85000"
                />
              </div>

              <div class="field form-grid__full">
                <label class="field__label" for="mf-shop">정비소</label>
                <input
                  id="mf-shop"
                  v-model="form.shopName"
                  class="input"
                  type="text"
                  placeholder="강남 오토센터"
                />
              </div>

              <div class="field form-grid__full">
                <label class="field__label" for="mf-memo">메모</label>
                <textarea
                  id="mf-memo"
                  v-model="form.memo"
                  class="textarea"
                  placeholder="다음 점검 예정일, 특이사항 등"
                />
              </div>
            </div>
          </form>

          <div class="modal__foot">
            <button type="button" class="btn" :disabled="saving" @click="modalOpen = false">
              취소
            </button>
            <button type="button" class="btn btn--primary" :disabled="saving" @click="onSave">
              <span v-if="saving" class="spinner" aria-hidden="true" />
              등록
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.maintenance {
  display: flex;
  flex-direction: column;
  gap: var(--sp-5);
}

.maintenance__filter {
  width: 200px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--sp-4);
}

.form-grid__full {
  grid-column: 1 / -1;
}

@media (max-width: 640px) {
  .form-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
