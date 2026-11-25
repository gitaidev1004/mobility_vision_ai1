<script setup>
/**
 * 확인 다이얼로그
 * - 삭제/정산처럼 되돌릴 수 없는 동작 앞에 띄운다.
 * - ESC 로 닫히고, 열릴 때 확인 버튼에 포커스를 준다(키보드 접근성).
 * - v-model:open 으로 열림 상태를 양방향 바인딩한다.
 */
import { ref, watch, onBeforeUnmount } from 'vue'
import AppIcon from './AppIcon.vue'

const props = defineProps({
  /** 열림 여부 */
  open: { type: Boolean, default: false },
  /** 제목 */
  title: { type: String, default: '확인' },
  /** 본문 메시지 */
  message: { type: String, default: '이 작업을 진행하시겠습니까?' },
  /** 확인 버튼 문구 */
  confirmText: { type: String, default: '확인' },
  /** 취소 버튼 문구 */
  cancelText: { type: String, default: '취소' },
  /** 확인 버튼 톤: primary | danger | success */
  tone: { type: String, default: 'primary' },
  /** 처리 중이면 버튼을 잠근다. */
  busy: { type: Boolean, default: false },
})

const emit = defineEmits(['update:open', 'confirm', 'cancel'])

/** 확인 버튼 DOM 참조(포커스 이동용) */
const confirmBtn = ref(null)

/** 닫기 */
function close() {
  emit('update:open', false)
  emit('cancel')
}

/** 확인 */
function confirm() {
  if (props.busy) return
  emit('confirm')
}

/** ESC 키로 닫는다. */
function onKeydown(e) {
  if (e.key === 'Escape') close()
}

// 열림 상태가 바뀔 때 키 리스너를 붙이거나 떼고, 포커스를 이동한다.
watch(
  () => props.open,
  async (isOpen) => {
    if (isOpen) {
      window.addEventListener('keydown', onKeydown)
      // DOM 이 그려진 뒤 포커스를 준다.
      requestAnimationFrame(() => confirmBtn.value?.focus())
    } else {
      window.removeEventListener('keydown', onKeydown)
    }
  },
)

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="modal-backdrop"
      role="dialog"
      aria-modal="true"
      :aria-label="title"
      @click.self="close"
    >
      <div class="modal modal--sm">
        <div class="modal__head">
          <span class="confirm__icon" :class="`confirm__icon--${tone}`">
            <AppIcon :name="tone === 'danger' ? 'alert' : 'check'" :size="18" />
          </span>
          <h3 class="modal__title">{{ title }}</h3>
          <button
            type="button"
            class="modal__close"
            aria-label="닫기"
            @click="close"
          >
            <AppIcon name="close" :size="16" />
          </button>
        </div>

        <div class="modal__body">
          <p class="confirm__message">{{ message }}</p>
          <!-- 추가 설명이 필요하면 기본 슬롯으로 넣는다. -->
          <slot />
        </div>

        <div class="modal__foot">
          <button type="button" class="btn" :disabled="busy" @click="close">
            {{ cancelText }}
          </button>
          <button
            ref="confirmBtn"
            type="button"
            class="btn"
            :class="`btn--${tone}`"
            :disabled="busy"
            @click="confirm"
          >
            <span v-if="busy" class="spinner" aria-hidden="true" />
            {{ confirmText }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.confirm__icon {
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  background: var(--c-primary-soft);
  color: var(--c-primary);
}

.confirm__icon--danger {
  background: var(--c-danger-soft);
  color: var(--c-danger);
}

.confirm__icon--success {
  background: var(--c-success-soft);
  color: var(--c-success);
}

.confirm__message {
  font-size: var(--fs-md);
  color: var(--c-text-sub);
  line-height: 1.65;
}
</style>
