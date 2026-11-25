/**
 * 아주 작은 토스트(알림) 허브
 * - pinia 스토어 목록은 계약서에 고정되어 있으므로, 토스트는 전역 reactive 배열로 관리한다.
 * - ToastHost.vue 가 이 배열을 렌더링한다.
 */
import { reactive } from 'vue'

/** 표시 중인 토스트 목록 */
export const toasts = reactive([])

let seq = 0

/**
 * 토스트를 띄운다.
 * @param {string} text 표시할 문구
 * @param {'info'|'success'|'warn'|'danger'} tone 색상 톤
 * @param {number} duration 자동 닫힘 시간(ms). 0 이면 수동으로만 닫힌다.
 * @returns {number} 토스트 ID
 */
export function pushToast(text, tone = 'info', duration = 3200) {
  const id = ++seq
  toasts.push({ id, text, tone })
  if (duration > 0) {
    setTimeout(() => removeToast(id), duration)
  }
  return id
}

/** 성공 토스트 단축 함수. */
export function toastSuccess(text) {
  return pushToast(text, 'success')
}

/** 경고 토스트 단축 함수. */
export function toastWarn(text) {
  return pushToast(text, 'warn')
}

/** 오류 토스트 단축 함수(조금 더 길게 보여준다). */
export function toastError(text) {
  return pushToast(text, 'danger', 5000)
}

/** 특정 토스트를 닫는다. */
export function removeToast(id) {
  const i = toasts.findIndex((t) => t.id === id)
  if (i >= 0) toasts.splice(i, 1)
}

/** 모든 토스트를 닫는다(라우트 전환 시). */
export function clearToasts() {
  toasts.splice(0, toasts.length)
}
