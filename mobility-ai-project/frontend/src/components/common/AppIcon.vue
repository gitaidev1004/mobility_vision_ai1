<script setup>
/**
 * 인라인 SVG 아이콘
 * - utils/icons.js 의 path 문자열을 stroke 방식으로 그린다.
 * - 장식용이므로 기본적으로 aria-hidden 이며, 단독으로 의미를 가질 때만 label 을 준다.
 */
import { computed } from 'vue'
import { iconPath } from '@/utils/icons'

const props = defineProps({
  /** utils/icons.js 의 키 이름 */
  name: { type: String, required: true },
  /** 픽셀 크기(정사각형) */
  size: { type: [Number, String], default: 18 },
  /** 선 두께 */
  width: { type: [Number, String], default: 1.8 },
  /** 스크린리더용 설명. 주면 role="img" 로 노출된다. */
  label: { type: String, default: '' },
})

const d = computed(() => iconPath(props.name))
</script>

<template>
  <svg
    class="app-icon"
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    :stroke-width="width"
    stroke-linecap="round"
    stroke-linejoin="round"
    :role="label ? 'img' : undefined"
    :aria-label="label || undefined"
    :aria-hidden="label ? undefined : 'true'"
  >
    <path :d="d" />
  </svg>
</template>

<style scoped>
.app-icon {
  display: block;
  flex: 0 0 auto;
}
</style>
