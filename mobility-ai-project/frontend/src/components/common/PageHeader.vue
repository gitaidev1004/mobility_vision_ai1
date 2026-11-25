<script setup>
/**
 * 화면 상단 제목 영역
 * - 제목 + 설명 + 우측 액션 슬롯으로 구성한다.
 * - 모든 업무 화면이 동일한 리듬을 갖도록 강제하는 역할을 한다.
 */
defineProps({
  /** 화면 제목 */
  title: { type: String, required: true },
  /** 제목 아래 보조 설명(한 줄) */
  desc: { type: String, default: '' },
})
</script>

<template>
  <div class="page-header">
    <div class="page-header__text">
      <h2 class="page-header__title">{{ title }}</h2>
      <p v-if="desc" class="page-header__desc">{{ desc }}</p>
    </div>

    <!-- 우측 버튼/필터 영역 -->
    <div v-if="$slots.actions" class="page-header__actions">
      <slot name="actions" />
    </div>
  </div>
</template>
