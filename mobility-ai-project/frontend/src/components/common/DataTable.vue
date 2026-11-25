<script setup>
/**
 * 범용 데이터 표
 * - columns 로 열을 선언하고, 행 셀은 기본 렌더링 또는 이름 슬롯으로 커스터마이즈한다.
 *   예) <template #cell-plateNumber="{ row }"> ... </template>
 * - 접근성: <caption> 과 <th scope="col"> 을 반드시 출력한다.
 * - 로딩/빈 상태를 표 안에서 직접 처리해 각 화면의 코드를 줄인다.
 *
 * columns 항목 형식:
 *   { key:'plateNumber', label:'번호판', align:'left'|'right'|'center',
 *     width:'120px', cellClass:'cell-plate' }
 */
import { computed } from 'vue'
import EmptyState from './EmptyState.vue'

const props = defineProps({
  /** 열 정의 배열 */
  columns: { type: Array, required: true },
  /** 행 데이터 배열 */
  rows: { type: Array, default: () => [] },
  /** 접근성용 표 설명(필수) */
  caption: { type: String, required: true },
  /** 행 고유 키 필드명 */
  rowKey: { type: String, default: 'id' },
  /** 로딩 중이면 스켈레톤 행을 보여준다. */
  loading: { type: Boolean, default: false },
  /** 로딩 스켈레톤 행 수 */
  skeletonRows: { type: Number, default: 5 },
  /** 행 높이를 줄인 조밀한 표 */
  compact: { type: Boolean, default: false },
  /** 빈 상태 문구 */
  emptyTitle: { type: String, default: '표시할 데이터가 없습니다' },
  emptyDesc: { type: String, default: '' },
  emptyIcon: { type: String, default: 'inbox' },
})

const emit = defineEmits(['row-click'])

/** 열 정렬 클래스를 계산한다. */
function alignClass(col) {
  if (col.align === 'right') return 'is-num'
  if (col.align === 'center') return 'is-center'
  return ''
}

/** 셀 값을 안전하게 읽는다(중첩 키 'matchedVehicle.ownerName' 도 지원). */
function cellValue(row, key) {
  if (!key) return ''
  if (!key.includes('.')) {
    const v = row?.[key]
    return v === null || v === undefined || v === '' ? '-' : v
  }
  const v = key.split('.').reduce((acc, k) => (acc == null ? acc : acc[k]), row)
  return v === null || v === undefined || v === '' ? '-' : v
}

/** 행 클릭 가능 여부(리스너가 붙어 있을 때만 커서를 바꾼다). */
const clickable = computed(() => !!props.rows.length)

const isEmpty = computed(() => !props.loading && props.rows.length === 0)
</script>

<template>
  <div class="table-wrap">
    <table class="table" :class="{ 'table--compact': compact }">
      <!-- 접근성: 표의 목적을 설명하는 caption (시각적으로는 감춤) -->
      <caption>{{ caption }}</caption>

      <thead>
        <tr>
          <th
            v-for="col in columns"
            :key="col.key || col.label"
            scope="col"
            :class="alignClass(col)"
            :style="col.width ? { width: col.width } : undefined"
          >
            {{ col.label }}
          </th>
        </tr>
      </thead>

      <tbody>
        <!-- 로딩 스켈레톤 -->
        <template v-if="loading">
          <tr v-for="n in skeletonRows" :key="`sk-${n}`">
            <td v-for="col in columns" :key="`sk-${n}-${col.key || col.label}`">
              <span class="skeleton table__skeleton" aria-hidden="true" />
            </td>
          </tr>
        </template>

        <!-- 실제 데이터 -->
        <template v-else>
          <tr
            v-for="(row, i) in rows"
            :key="row?.[rowKey] ?? i"
            :class="{ 'is-clickable': clickable }"
            @click="emit('row-click', row)"
          >
            <td
              v-for="col in columns"
              :key="col.key || col.label"
              :class="[alignClass(col), col.cellClass]"
            >
              <!-- 열별 커스텀 슬롯이 있으면 그것을 쓰고, 없으면 값을 그대로 출력 -->
              <slot :name="`cell-${col.key}`" :row="row" :value="row?.[col.key]" :index="i">
                {{ cellValue(row, col.key) }}
              </slot>
            </td>
          </tr>
        </template>
      </tbody>
    </table>

    <!-- 빈 상태: 표 아래에 안내를 띄운다. -->
    <EmptyState
      v-if="isEmpty"
      :title="emptyTitle"
      :desc="emptyDesc"
      :icon="emptyIcon"
    >
      <slot name="empty-action" />
    </EmptyState>
  </div>
</template>

<style scoped>
.table__skeleton {
  display: block;
  height: 14px;
  width: 80%;
}

.table tbody tr.is-clickable {
  cursor: default;
}
</style>
