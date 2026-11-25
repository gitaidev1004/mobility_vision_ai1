<script setup>
/**
 * 토스트 표시 영역
 * - utils/toast.js 의 전역 reactive 배열을 읽어 우측 하단에 쌓는다.
 * - aria-live="polite" 로 스크린리더에도 알림이 전달되게 한다.
 */
import AppIcon from './AppIcon.vue'
import { toasts, removeToast } from '@/utils/toast'

/** 톤별 아이콘 이름 */
const ICON_BY_TONE = {
  success: 'check',
  warn: 'alert',
  danger: 'alert',
  info: 'clock',
}
</script>

<template>
  <Teleport to="body">
    <div class="toast-host" role="status" aria-live="polite">
      <TransitionGroup name="toast">
        <div
          v-for="t in toasts"
          :key="t.id"
          class="toast"
          :class="`toast--${t.tone}`"
        >
          <AppIcon :name="ICON_BY_TONE[t.tone] || 'clock'" :size="16" class="toast__icon" />
          <span class="toast__text">{{ t.text }}</span>
          <button
            type="button"
            class="toast__close"
            aria-label="알림 닫기"
            @click="removeToast(t.id)"
          >
            <AppIcon name="close" :size="14" />
          </button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<style scoped>
.toast__icon {
  margin-top: 2px;
  color: var(--c-text-sub);
}

.toast__close {
  border: 0;
  background: transparent;
  color: var(--c-text-muted);
  padding: 2px;
  border-radius: 4px;
  display: grid;
  place-items: center;
}

.toast__close:hover {
  background: var(--c-surface-alt);
  color: var(--c-text);
}

/* 등장/퇴장 애니메이션 */
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateY(8px);
}

.toast-enter-active,
.toast-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
</style>
