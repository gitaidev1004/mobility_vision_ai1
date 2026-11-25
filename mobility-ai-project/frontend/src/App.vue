<script setup>
/**
 * 루트 컴포넌트
 * - 로그인 화면은 레이아웃(사이드바/상단바) 없이 단독으로 렌더링한다.
 * - 그 외 라우트는 AppLayout 안에서 렌더링한다.
 * - http.js 가 401 에서 발생시키는 커스텀 이벤트를 받아 강제 로그아웃 처리한다.
 */
import { computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppLayout from '@/components/layout/AppLayout.vue'
import ToastHost from '@/components/common/ToastHost.vue'
import { useAuthStore } from '@/stores/auth'
import { UNAUTHORIZED_EVENT } from '@/api/http'
import { toastWarn } from '@/utils/toast'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

/** 공개 라우트(= 로그인)는 레이아웃 없이 전체 화면을 쓴다. */
const isPublic = computed(() => route.meta?.public === true)

/** 401 응답 시: 세션을 비우고 로그인 화면으로 보낸다. */
function onUnauthorized() {
  auth.clearSession()
  toastWarn('세션이 만료되었습니다. 다시 로그인해 주세요.')
  if (route.name !== 'login') {
    router.replace({ name: 'login', query: { redirect: route.fullPath } })
  }
}

onMounted(() => {
  window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized)
})

onBeforeUnmount(() => {
  window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized)
})
</script>

<template>
  <!-- 로그인 화면: 레이아웃 없음 -->
  <RouterView v-if="isPublic" />

  <!-- 업무 화면: 사이드바 + 상단바 레이아웃 -->
  <AppLayout v-else>
    <RouterView />
  </AppLayout>

  <!-- 토스트는 레이아웃과 무관하게 항상 최상위에 둔다. -->
  <ToastHost />
</template>
