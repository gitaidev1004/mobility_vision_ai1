<script setup>
/**
 * 빈 상태 안내
 * - 조회 결과가 없거나 아직 작업을 시작하지 않았을 때 표의 자리를 채운다.
 * - 액션 슬롯에 '등록' 같은 버튼을 넣을 수 있다.
 */
import AppIcon from './AppIcon.vue'

defineProps({
  /** 굵은 한 줄 */
  title: { type: String, default: '표시할 데이터가 없습니다' },
  /** 설명 한 줄 */
  desc: { type: String, default: '' },
  /** utils/icons.js 아이콘 이름 */
  icon: { type: String, default: 'inbox' },
})
</script>

<template>
  <div class="empty-state">
    <AppIcon :name="icon" :size="44" :width="1.4" class="empty-state__icon" />
    <div class="empty-state__title">{{ title }}</div>
    <div v-if="desc" class="empty-state__desc">{{ desc }}</div>

    <div v-if="$slots.default" class="empty-state__action">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.empty-state__action {
  margin-top: var(--sp-4);
  display: flex;
  justify-content: center;
}
</style>
