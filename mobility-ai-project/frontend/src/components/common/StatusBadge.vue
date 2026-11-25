<script setup>
/**
 * 상태 배지
 * - 계약서 2장의 enum 값을 받아 한국어 라벨 + 색상 톤으로 표시한다.
 * - kind prop 으로 어느 enum 인지 알려준다.
 *   kind='recognition' → SUCCESS|LOW_CONFIDENCE|NO_PLATE|FORMAT_INVALID
 *   kind='risk'        → LOW|MEDIUM|HIGH|CRITICAL
 *   kind='event'       → ENTRY|EXIT
 *   kind='allowed'     → true|false (불리언)
 *   kind='paid'        → true|false (불리언)
 *   kind='active'      → true|false (불리언)
 *   kind='component'   → NORMAL|WATCH|WARNING|CRITICAL
 */
import { computed } from 'vue'

const props = defineProps({
  /** enum 값 또는 불리언 */
  value: { type: [String, Boolean, null], default: null },
  /** enum 종류 */
  kind: { type: String, default: 'recognition' },
  /** 점 표식 없이 평평하게 표시 */
  plain: { type: Boolean, default: false },
})

/** 각 enum 별 { 라벨, 톤 } 매핑 테이블 */
const MAP = {
  recognition: {
    SUCCESS: ['성공', 'success'],
    LOW_CONFIDENCE: ['신뢰도 낮음', 'warn'],
    NO_PLATE: ['번호판 미검출', 'danger'],
    FORMAT_INVALID: ['형식 오류', 'danger'],
  },
  risk: {
    LOW: ['낮음', 'success'],
    MEDIUM: ['보통', 'warn'],
    HIGH: ['높음', 'danger'],
    CRITICAL: ['심각', 'danger'],
  },
  event: {
    ENTRY: ['입차', 'primary'],
    EXIT: ['출차', 'info'],
  },
  component: {
    NORMAL: ['정상', 'success'],
    WATCH: ['관찰', 'warn'],
    WARNING: ['주의', 'warn'],
    CRITICAL: ['위험', 'danger'],
  },
  allowed: {
    true: ['허용', 'success'],
    false: ['거부', 'danger'],
  },
  paid: {
    true: ['정산 완료', 'success'],
    false: ['미정산', 'warn'],
  },
  active: {
    true: ['사용', 'success'],
    false: ['중지', 'neutral'],
  },
  parked: {
    true: ['주차 중', 'primary'],
    false: ['출차', 'neutral'],
  },
}

/** [라벨, 톤] 쌍을 찾는다. 모르는 값은 원본 문자열 + neutral. */
const pair = computed(() => {
  const table = MAP[props.kind] || {}
  const key = String(props.value)
  return table[key] || [props.value === null ? '-' : key, 'neutral']
})

const label = computed(() => pair.value[0])
const tone = computed(() => pair.value[1])
</script>

<template>
  <span class="badge" :class="[`badge--${tone}`, { 'badge--plain': plain }]">{{ label }}</span>
</template>
