<script setup>
/**
 * 애플리케이션 셸 레이아웃
 * - 좌측 사이드바(고정) + 우측 { 상단바 + 본문 } 구조.
 * - 본문은 기본 슬롯으로 받아 RouterView 를 그린다.
 * - 대시보드 요약은 사이드바 '미정산' 배지에도 쓰이므로 레이아웃 진입 시 한 번 로드한다.
 */
import { onMounted } from 'vue'
import AppSidebar from './AppSidebar.vue'
import AppTopbar from './AppTopbar.vue'
import { useDashboardStore } from '@/stores/dashboard'
import { useAuthStore } from '@/stores/auth'

const dashboard = useDashboardStore()
const auth = useAuthStore()

onMounted(() => {
  // 이미 받아둔 요약이 있으면 재요청하지 않는다(화면 전환 시 중복 호출 방지).
  if (auth.isAuthenticated && !dashboard.summary) {
    dashboard.fetchSummary()
  }
})
</script>

<template>
  <div class="app-shell">
    <AppSidebar />

    <div class="app-main">
      <AppTopbar />

      <!-- 각 화면의 내용 -->
      <main class="app-content">
        <slot />
      </main>
    </div>
  </div>
</template>
