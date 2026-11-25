<script setup>
/**
 * 통계 카드
 * - 라벨 + 큰 숫자 + 단위 + 증감 표기 + 좌측 아이콘으로 구성한다.
 * - format prop 으로 숫자/퍼센트/금액 포맷을 선택한다.
 */
import { computed } from 'vue'
import AppIcon from './AppIcon.vue'
import { formatNumber, formatCurrency, formatPercent, formatDuration } from '@/utils/format'

const props = defineProps({
  /** 지표 이름 */
  label: { type: String, required: true },
  /** 지표 값 */
  value: { type: [Number, String], default: 0 },
  /** 값 뒤에 붙는 단위(대, 건 등). format 이 currency/percent 면 무시된다. */
  unit: { type: String, default: '' },
  /** 'number' | 'currency' | 'percent' | 'duration' | 'text' */
  format: { type: String, default: 'number' },
  /** 아이콘 색상 톤: primary | success | warn | danger | info */
  tone: { type: String, default: 'primary' },
  /** utils/icons.js 아이콘 이름 */
  icon: { type: String, default: 'dashboard' },
  /** 전일/전주 대비 증감(숫자). null 이면 표시하지 않는다. */
  delta: { type: Number, default: null },
  /** 증감 설명 문구(예: '어제 대비') */
  deltaLabel: { type: String, default: '' },
  /** 로딩 중이면 스켈레톤을 보여준다. */
  loading: { type: Boolean, default: false },
})

/** format 에 맞춰 값 문자열을 만든다. */
const display = computed(() => {
  if (props.format === 'currency') return formatCurrency(props.value, false)
  if (props.format === 'percent') return formatPercent(props.value, 1)
  if (props.format === 'duration') return formatDuration(props.value)
  // 'text' 는 숫자가 아닌 값(부품명 등)을 그대로 보여준다.
  if (props.format === 'text') return props.value === null || props.value === '' ? '-' : String(props.value)
  return formatNumber(props.value)
})

/** currency 포맷일 때는 단위를 '원' 으로 강제한다. */
const displayUnit = computed(() => {
  if (props.format === 'currency') return '원'
  if (props.format === 'percent' || props.format === 'duration' || props.format === 'text') return ''
  return props.unit
})

/** 증감 방향에 따른 클래스 */
const deltaClass = computed(() => {
  if (props.delta === null) return ''
  if (props.delta > 0) return 'stat-card__delta--up'
  if (props.delta < 0) return 'stat-card__delta--down'
  return 'stat-card__delta--flat'
})

/** 증감 표시 문자열 */
const deltaText = computed(() => {
  if (props.delta === null) return ''
  const sign = props.delta > 0 ? '▲' : props.delta < 0 ? '▼' : '—'
  return `${sign} ${formatNumber(Math.abs(props.delta))}`
})
</script>

<template>
  <div class="stat-card">
    <span class="stat-card__icon" :class="`stat-card__icon--${tone}`">
      <AppIcon :name="icon" :size="20" />
    </span>

    <div class="stat-card__main">
      <div class="stat-card__label">{{ label }}</div>

      <!-- 로딩 중에는 숫자 자리에 스켈레톤을 보여준다. -->
      <div v-if="loading" class="skeleton stat-card__skeleton" aria-hidden="true" />
      <div v-else class="stat-card__value">
        {{ display }}<span v-if="displayUnit" class="stat-card__unit">{{ displayUnit }}</span>
      </div>

      <div v-if="delta !== null && !loading" class="stat-card__delta" :class="deltaClass">
        <span>{{ deltaText }}</span>
        <span v-if="deltaLabel" class="stat-card__delta-label">{{ deltaLabel }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.stat-card__skeleton {
  height: 32px;
  width: 70%;
  margin-top: 4px;
}

.stat-card__delta-label {
  color: var(--c-text-muted);
  font-weight: 500;
}
</style>
