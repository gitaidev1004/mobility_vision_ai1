<script setup>
/**
 * 차량 관리 화면 (/vehicles)
 * - 검색(번호판/차주) + 페이지 목록 + 등록·수정 모달 + 삭제 확인
 * - 백엔드 엔드포인트: GET/POST /vehicles, GET/PUT/DELETE /vehicles/{id}
 * - VIEWER 역할은 쓰기 버튼이 비활성화된다(auth.canWrite).
 */
import { computed, onMounted, reactive, ref } from 'vue'
import PageHeader from '@/components/common/PageHeader.vue'
import DataTable from '@/components/common/DataTable.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import PaginationBar from '@/components/common/PaginationBar.vue'
import ConfirmDialog from '@/components/common/ConfirmDialog.vue'
import AppIcon from '@/components/common/AppIcon.vue'
import { useVehicleStore } from '@/stores/vehicle'
import { useAuthStore } from '@/stores/auth'
import { formatDate, formatNumber } from '@/utils/format'
import { toastSuccess, toastError } from '@/utils/toast'

const store = useVehicleStore()
const auth = useAuthStore()

/** 검색 입력(엔터 또는 버튼으로 실행) */
const keyword = ref('')

/** 등록/수정 모달 열림 상태 */
const modalOpen = ref(false)
/** 수정 중인 차량 ID(null 이면 신규 등록) */
const editingId = ref(null)
/** 삭제 확인 대상 */
const deleteTarget = ref(null)

/** 차종 후보 */
const vehicleTypes = ['승용', 'SUV', '승합', '화물', '이륜', '전기차']
/** 정기권 종류 후보 */
const passTypes = [
  { value: 'NONE', label: '없음(일반)' },
  { value: 'REGULAR', label: '정기권' },
  { value: 'MONTHLY', label: '월정액' },
  { value: 'VIP', label: 'VIP' },
  { value: 'STAFF', label: '임직원' },
]

/** 폼 상태. vehicle 테이블 컬럼과 1:1 대응한다. */
const form = reactive({
  plateNumber: '',
  ownerName: '',
  phone: '',
  vehicleType: '승용',
  modelName: '',
  modelYear: new Date().getFullYear(),
  vin: '',
  passType: 'NONE',
  passExpiresAt: '',
  active: true,
})

/** 폼 검증 오류 메시지 */
const formErrors = reactive({ plateNumber: '', ownerName: '' })

/** 표 열 정의 */
const columns = [
  { key: 'rowNo', label: '번호', width: '60px', align: 'right' },
  { key: 'plateNumber', label: '번호판', width: '130px', cellClass: 'cell-plate' },
  { key: 'ownerName', label: '차주', width: '90px' },
  { key: 'phone', label: '연락처', width: '130px' },
  { key: 'vehicleType', label: '차종', width: '80px' },
  { key: 'modelName', label: '모델', width: '140px' },
  { key: 'modelYear', label: '연식', width: '70px', align: 'right' },
  { key: 'passType', label: '정기권', width: '100px' },
  { key: 'passExpiresAt', label: '만료일', width: '110px', cellClass: 'cell-time' },
  { key: 'active', label: '상태', width: '80px', align: 'center' },
  { key: 'actions', label: '관리', width: '92px', align: 'center' },
]

/** 모달 제목 */
const modalTitle = computed(() => (editingId.value ? '차량 정보 수정' : '차량 등록'))

/** 한국 번호판 정규식: '12가3456' 또는 '서울 34나5678' */
const PLATE_RE = /^(?:[가-힣]{2}\s?)?\d{2,3}[가-힣]\d{4}$/

/** 폼을 초기화한다. */
function resetForm() {
  form.plateNumber = ''
  form.ownerName = ''
  form.phone = ''
  form.vehicleType = '승용'
  form.modelName = ''
  form.modelYear = new Date().getFullYear()
  form.vin = ''
  form.passType = 'NONE'
  form.passExpiresAt = ''
  form.active = true
  formErrors.plateNumber = ''
  formErrors.ownerName = ''
}

/** 등록 모달 열기 */
function openCreate() {
  editingId.value = null
  resetForm()
  modalOpen.value = true
}

/** 수정 모달 열기(행 데이터를 폼에 복사) */
function openEdit(row) {
  editingId.value = row.id
  resetForm()
  form.plateNumber = row.plateNumber ?? ''
  form.ownerName = row.ownerName ?? ''
  form.phone = row.phone ?? ''
  form.vehicleType = row.vehicleType ?? '승용'
  form.modelName = row.modelName ?? ''
  form.modelYear = row.modelYear ?? new Date().getFullYear()
  form.vin = row.vin ?? ''
  form.passType = row.passType ?? 'NONE'
  // <input type="date"> 는 'YYYY-MM-DD' 형식만 받는다.
  form.passExpiresAt = row.passExpiresAt ? String(row.passExpiresAt).slice(0, 10) : ''
  form.active = row.active !== false
  modalOpen.value = true
}

/** 모달 닫기 */
function closeModal() {
  modalOpen.value = false
  editingId.value = null
}

/** 폼 검증. 통과하면 true. */
function validate() {
  formErrors.plateNumber = ''
  formErrors.ownerName = ''
  const plate = form.plateNumber.trim()
  if (!plate) {
    formErrors.plateNumber = '번호판을 입력하세요.'
  } else if (!PLATE_RE.test(plate)) {
    formErrors.plateNumber = "형식이 올바르지 않습니다. 예) 12가3456 또는 '서울 34나5678'"
  }
  if (!form.ownerName.trim()) {
    formErrors.ownerName = '차주명을 입력하세요.'
  }
  return !formErrors.plateNumber && !formErrors.ownerName
}

/** 등록/수정 저장 */
async function onSave() {
  if (!validate()) return

  const payload = {
    plateNumber: form.plateNumber.trim(),
    ownerName: form.ownerName.trim(),
    phone: form.phone.trim() || null,
    vehicleType: form.vehicleType,
    modelName: form.modelName.trim() || null,
    modelYear: form.modelYear ? Number(form.modelYear) : null,
    vin: form.vin.trim() || null,
    passType: form.passType,
    // 'YYYY-MM-DD' → LocalDateTime 으로 보내기 위해 자정 시각을 붙인다.
    passExpiresAt: form.passExpiresAt ? `${form.passExpiresAt}T00:00:00` : null,
    active: form.active,
  }

  try {
    if (editingId.value) {
      await store.update(editingId.value, payload)
      toastSuccess(`${payload.plateNumber} 정보를 수정했습니다.`)
    } else {
      await store.create(payload)
      toastSuccess(`${payload.plateNumber} 차량을 등록했습니다.`)
    }
    closeModal()
  } catch {
    toastError(store.error || '저장에 실패했습니다.')
  }
}

/** 삭제 확인 */
function askDelete(row) {
  deleteTarget.value = row
}

/** 삭제 실행 */
async function onDelete() {
  if (!deleteTarget.value) return
  const plate = deleteTarget.value.plateNumber
  try {
    await store.remove(deleteTarget.value.id)
    toastSuccess(`${plate} 차량을 삭제했습니다.`)
  } catch {
    toastError(store.error || '삭제에 실패했습니다.')
  } finally {
    deleteTarget.value = null
  }
}

/** 검색 실행 */
function onSearch() {
  store.search(keyword.value.trim())
}

/** 검색 초기화 */
function onClearSearch() {
  keyword.value = ''
  store.search('')
}

onMounted(() => {
  store.fetchList({ page: 0 })
})
</script>

<template>
  <div class="vehicles">
    <PageHeader
      title="차량 관리"
      desc="등록 차량의 번호판 · 차주 · 정기권 정보를 관리합니다."
    >
      <template #actions>
        <button
          type="button"
          class="btn btn--primary"
          :disabled="!auth.canWrite"
          :title="auth.canWrite ? '' : '조회 권한만 있습니다.'"
          aria-label="차량 등록"
          @click="openCreate"
        >
          <AppIcon name="plus" :size="16" />
          차량 등록
        </button>
      </template>
    </PageHeader>

    <section class="card">
      <!-- 검색 영역 -->
      <div class="card__head">
        <form class="filter-bar vehicles__search" @submit.prevent="onSearch">
          <div class="field field--grow">
            <label class="sr-only" for="veh-q">번호판 또는 차주 검색</label>
            <input
              id="veh-q"
              v-model="keyword"
              class="input"
              type="search"
              placeholder="번호판 또는 차주명으로 검색 (예: 12가3456, 홍길동)"
            />
          </div>
          <button type="submit" class="btn btn--primary" aria-label="검색 실행">
            <AppIcon name="search" :size="16" />
            검색
          </button>
          <button
            v-if="store.query"
            type="button"
            class="btn"
            aria-label="검색 초기화"
            @click="onClearSearch"
          >
            초기화
          </button>
        </form>

        <div class="card__tools">
          <span class="card__hint">총 {{ formatNumber(store.totalElements) }}대</span>
        </div>
      </div>

      <!-- 목록 -->
      <div class="card__body card__body--flush">
        <DataTable
          caption="등록 차량 목록. 번호판, 차주, 연락처, 차종, 모델, 연식, 정기권, 만료일, 상태를 보여줍니다."
          :columns="columns"
          :rows="store.items"
          :loading="store.loading"
          :empty-title="store.query ? '검색 결과가 없습니다' : '등록된 차량이 없습니다'"
          :empty-desc="store.query ? '다른 검색어로 시도해 보세요.' : '차량 등록 버튼으로 첫 차량을 추가하세요.'"
          empty-icon="car"
        >
          <!-- 행 번호: 페이지 오프셋을 더해 전체 기준 번호로 표시 -->
          <template #cell-rowNo="{ index }">
            {{ store.rowOffset + index + 1 }}
          </template>

          <template #cell-passType="{ row }">
            <span
              v-if="row.passType && row.passType !== 'NONE'"
              class="badge badge--primary"
            >{{ row.passType }}</span>
            <span v-else class="cell-muted">일반</span>
          </template>

          <template #cell-passExpiresAt="{ row }">
            {{ row.passExpiresAt ? formatDate(row.passExpiresAt) : '-' }}
          </template>

          <template #cell-active="{ row }">
            <StatusBadge kind="active" :value="row.active !== false" />
          </template>

          <!-- 관리 버튼 -->
          <template #cell-actions="{ row }">
            <div class="vehicles__row-actions">
              <button
                type="button"
                class="btn btn--sm btn--icon"
                :disabled="!auth.canWrite"
                :aria-label="`${row.plateNumber} 수정`"
                title="수정"
                @click.stop="openEdit(row)"
              >
                <AppIcon name="edit" :size="14" />
              </button>
              <button
                type="button"
                class="btn btn--sm btn--icon"
                :disabled="!auth.canWrite"
                :aria-label="`${row.plateNumber} 삭제`"
                title="삭제"
                @click.stop="askDelete(row)"
              >
                <AppIcon name="trash" :size="14" />
              </button>
            </div>
          </template>

          <template #empty-action>
            <button
              type="button"
              class="btn btn--primary"
              :disabled="!auth.canWrite"
              @click="openCreate"
            >
              차량 등록
            </button>
          </template>
        </DataTable>
      </div>

      <!-- 페이지네이션 -->
      <div v-if="store.totalPages > 1" class="card__foot">
        <PaginationBar
          :page="store.page"
          :total-pages="store.totalPages"
          :total-elements="store.totalElements"
          :size="store.size"
          @change="store.goPage"
        />
      </div>
    </section>

    <!-- 등록/수정 모달 -->
    <Teleport to="body">
      <div
        v-if="modalOpen"
        class="modal-backdrop"
        role="dialog"
        aria-modal="true"
        :aria-label="modalTitle"
        @click.self="closeModal"
      >
        <div class="modal">
          <div class="modal__head">
            <h3 class="modal__title">{{ modalTitle }}</h3>
            <button type="button" class="modal__close" aria-label="닫기" @click="closeModal">
              <AppIcon name="close" :size="16" />
            </button>
          </div>

          <form class="modal__body" @submit.prevent="onSave">
            <div class="form-grid">
              <div class="field">
                <label class="field__label" for="f-plate">
                  번호판<span class="req" aria-hidden="true">*</span>
                </label>
                <input
                  id="f-plate"
                  v-model="form.plateNumber"
                  class="input"
                  type="text"
                  placeholder="12가3456"
                  required
                />
                <span v-if="formErrors.plateNumber" class="field__error">
                  {{ formErrors.plateNumber }}
                </span>
              </div>

              <div class="field">
                <label class="field__label" for="f-owner">
                  차주<span class="req" aria-hidden="true">*</span>
                </label>
                <input
                  id="f-owner"
                  v-model="form.ownerName"
                  class="input"
                  type="text"
                  placeholder="홍길동"
                  required
                />
                <span v-if="formErrors.ownerName" class="field__error">
                  {{ formErrors.ownerName }}
                </span>
              </div>

              <div class="field">
                <label class="field__label" for="f-phone">연락처</label>
                <input
                  id="f-phone"
                  v-model="form.phone"
                  class="input"
                  type="tel"
                  placeholder="010-1234-5678"
                />
              </div>

              <div class="field">
                <label class="field__label" for="f-type">차종</label>
                <select id="f-type" v-model="form.vehicleType" class="select">
                  <option v-for="t in vehicleTypes" :key="t" :value="t">{{ t }}</option>
                </select>
              </div>

              <div class="field">
                <label class="field__label" for="f-model">모델명</label>
                <input
                  id="f-model"
                  v-model="form.modelName"
                  class="input"
                  type="text"
                  placeholder="아이오닉 5"
                />
              </div>

              <div class="field">
                <label class="field__label" for="f-year">연식</label>
                <input
                  id="f-year"
                  v-model="form.modelYear"
                  class="input"
                  type="number"
                  min="1980"
                  max="2100"
                />
              </div>

              <div class="field form-grid__full">
                <label class="field__label" for="f-vin">차대번호(VIN)</label>
                <input
                  id="f-vin"
                  v-model="form.vin"
                  class="input"
                  type="text"
                  placeholder="KMHL14JA1MA123456"
                  maxlength="32"
                />
                <span class="field__hint">17자리 차대번호. 모르면 비워 두세요.</span>
              </div>

              <div class="field">
                <label class="field__label" for="f-pass">정기권</label>
                <select id="f-pass" v-model="form.passType" class="select">
                  <option v-for="p in passTypes" :key="p.value" :value="p.value">
                    {{ p.label }}
                  </option>
                </select>
              </div>

              <div class="field">
                <label class="field__label" for="f-expire">정기권 만료일</label>
                <input
                  id="f-expire"
                  v-model="form.passExpiresAt"
                  class="input"
                  type="date"
                  :disabled="form.passType === 'NONE'"
                />
              </div>

              <div class="field form-grid__full">
                <label class="vehicles__checkbox">
                  <input v-model="form.active" type="checkbox" />
                  <span>사용 중인 차량 (체크를 해제하면 출입이 거부됩니다)</span>
                </label>
              </div>
            </div>
          </form>

          <div class="modal__foot">
            <button type="button" class="btn" :disabled="store.saving" @click="closeModal">
              취소
            </button>
            <button
              type="button"
              class="btn btn--primary"
              :disabled="store.saving"
              @click="onSave"
            >
              <span v-if="store.saving" class="spinner" aria-hidden="true" />
              {{ editingId ? '수정' : '등록' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- 삭제 확인 -->
    <ConfirmDialog
      :open="!!deleteTarget"
      title="차량 삭제"
      :message="`${deleteTarget?.plateNumber ?? ''} 차량을 삭제하시겠습니까? 출입·주차 이력은 남지만 차량 정보는 복구할 수 없습니다.`"
      confirm-text="삭제"
      tone="danger"
      :busy="store.saving"
      @confirm="onDelete"
      @cancel="deleteTarget = null"
    />
  </div>
</template>

<style scoped>
.vehicles {
  display: flex;
  flex-direction: column;
  gap: var(--sp-5);
}

.vehicles__search {
  flex: 1 1 auto;
  min-width: 0;
}

.vehicles__row-actions {
  display: flex;
  gap: 4px;
  justify-content: center;
}

/* 모달 폼: 2열 그리드 */
.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--sp-4);
}

.form-grid__full {
  grid-column: 1 / -1;
}

.vehicles__checkbox {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  font-size: var(--fs-base);
  color: var(--c-text-sub);
  cursor: pointer;
}

@media (max-width: 640px) {
  .form-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
