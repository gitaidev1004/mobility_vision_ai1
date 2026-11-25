<script setup>
/**
 * 상단 바
 * - 현재 라우트 제목, 실시간 시계, 로그인 사용자, 로그아웃 버튼을 담는다.
 * - 시계는 1초마다 갱신하고, 컴포넌트가 사라질 때 타이머를 반드시 정리한다.
 */
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppIcon from '@/components/common/AppIcon.vue'
import { useAuthStore } from '@/stores/auth'
import { formatDateTime } from '@/utils/format'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

/** 현재 시각(1초 주기 갱신) */
const now = ref(new Date())
let timer = null

/** 상단바 제목 = 라우트 meta.title */
const title = computed(() => route.meta?.title || '대시보드')

/** 'YYYY-MM-DD HH:mm' 포맷 시계 */
const clock = computed(() => formatDateTime(now.value))

/** 로그아웃 후 로그인 화면으로 이동한다. */
async function onLogout() {
  await auth.logout()
  router.replace({ name: 'login' })
}

onMounted(() => {
  timer = setInterval(() => {
    now.value = new Date()
  }, 1000)
})

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <header class="topbar">
    <h1 class="topbar__title">{{ title }}</h1>
    <span class="topbar__crumb">모빌리티 비전 AI · 관리자</span>

    <div class="topbar__spacer" />

    <!-- 실시간 시계 -->
    <time class="topbar__clock" :datetime="now.toISOString()">{{ clock }}</time>

    <!-- 로그인 사용자 -->
    <div class="topbar__user">
      <span class="avatar" aria-hidden="true">{{ auth.initial }}</span>
      <span>
        <span class="topbar__user-name">{{ auth.displayName }}</span>
        <span class="topbar__user-role">{{ auth.username }} · {{ auth.roleLabel }}</span>
      </span>
    </div>

    <button
      type="button"
      class="btn btn--ghost btn--icon"
      aria-label="로그아웃"
      title="로그아웃"
      @click="onLogout"
    >
      <AppIcon name="logout" :size="18" />
    </button>
  </header>
</template>

<style scoped>
/* 사용자 이름과 역할을 2줄로 쌓기 위한 래퍼 */
.topbar__user > span:last-child {
  display: flex;
  flex-direction: column;
}
</style>
