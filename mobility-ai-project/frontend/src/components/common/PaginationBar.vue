<script setup>
/**
 * 페이지네이션
 * - 백엔드 페이지 응답의 page(0-base)/totalPages/totalElements 를 그대로 받는다.
 * - 페이지 수가 많으면 현재 페이지 주변 5개만 숫자로 노출한다.
 */
import { computed } from 'vue'

const props = defineProps({
  /** 현재 페이지(0-base) */
  page: { type: Number, default: 0 },
  /** 전체 페이지 수 */
  totalPages: { type: Number, default: 0 },
  /** 전체 행 수 */
  totalElements: { type: Number, default: 0 },
  /** 페이지 크기 */
  size: { type: Number, default: 20 },
  /** 숫자 버튼 최대 개수 */
  window: { type: Number, default: 5 },
})

const emit = defineEmits(['change'])

/** 표시할 페이지 번호 배열(0-base) */
const pages = computed(() => {
  const total = props.totalPages
  if (total <= 0) return []
  const half = Math.floor(props.window / 2)
  let start = Math.max(0, props.page - half)
  const end = Math.min(total - 1, start + props.window - 1)
  // 뒤쪽에서 창이 줄어들면 앞으로 당겨 항상 window 개를 유지한다.
  start = Math.max(0, end - props.window + 1)
  const out = []
  for (let i = start; i <= end; i += 1) out.push(i)
  return out
})

/** '1-20 / 총 137건' 안내 문구 */
const info = computed(() => {
  if (props.totalElements === 0) return '총 0건'
  const from = props.page * props.size + 1
  const to = Math.min(props.totalElements, (props.page + 1) * props.size)
  return `${from.toLocaleString('ko-KR')}–${to.toLocaleString('ko-KR')} / 총 ${props.totalElements.toLocaleString('ko-KR')}건`
})

const isFirst = computed(() => props.page <= 0)
const isLast = computed(() => props.page >= props.totalPages - 1)

/** 유효 범위 안에서만 change 이벤트를 올린다. */
function go(p) {
  if (p < 0 || p > props.totalPages - 1 || p === props.page) return
  emit('change', p)
}
</script>

<template>
  <nav class="pagination" aria-label="페이지 이동">
    <span class="pagination__info">{{ info }}</span>

    <button
      type="button"
      class="pagination__btn"
      :disabled="isFirst"
      aria-label="첫 페이지"
      @click="go(0)"
    >
      «
    </button>
    <button
      type="button"
      class="pagination__btn"
      :disabled="isFirst"
      aria-label="이전 페이지"
      @click="go(page - 1)"
    >
      ‹
    </button>

    <button
      v-for="p in pages"
      :key="p"
      type="button"
      class="pagination__btn"
      :class="{ 'is-active': p === page }"
      :aria-current="p === page ? 'page' : undefined"
      :aria-label="`${p + 1}페이지`"
      @click="go(p)"
    >
      {{ p + 1 }}
    </button>

    <button
      type="button"
      class="pagination__btn"
      :disabled="isLast"
      aria-label="다음 페이지"
      @click="go(page + 1)"
    >
      ›
    </button>
    <button
      type="button"
      class="pagination__btn"
      :disabled="isLast"
      aria-label="마지막 페이지"
      @click="go(totalPages - 1)"
    >
      »
    </button>
  </nav>
</template>
