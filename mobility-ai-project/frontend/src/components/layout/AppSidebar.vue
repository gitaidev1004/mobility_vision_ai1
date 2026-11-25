<script setup>
/**
 * 좌측 사이드바 (계약서 5장)
 * - 라우터 메뉴를 두 그룹(운영 / 분석)으로 나눠 표시한다.
 * - 현재 라우트와 일치하는 메뉴에 .is-active 를 준다.
 * - 1280px 이하에서는 components.css 의 미디어쿼리가 아이콘만 남기고 축소한다.
 */
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '@/components/common/AppIcon.vue'
import { useDashboardStore } from '@/stores/dashboard'

const route = useRoute()
const dashboard = useDashboardStore()

/** 운영 업무 메뉴 */
const operationMenu = [
  { name: 'dashboard', to: '/', label: '대시보드', icon: 'dashboard' },
  { name: 'recognition', to: '/recognition', label: '번호판 인식', icon: 'scan' },
  { name: 'events', to: '/events', label: '출입 이벤트', icon: 'gate' },
  { name: 'parking', to: '/parking', label: '주차 · 정산', icon: 'parking', badge: 'unpaid' },
]

/** 자산/분석 메뉴 */
const assetMenu = [
  { name: 'vehicles', to: '/vehicles', label: '차량 관리', icon: 'car' },
  { name: 'maintenance', to: '/maintenance', label: '정비 이력', icon: 'wrench' },
  { name: 'pdm', to: '/pdm', label: '예지보전', icon: 'pulse' },
  { name: 'benchmark', to: '/benchmark', label: '모델 성능', icon: 'chart' },
]

/** 미정산 건수 배지(0 이면 숨긴다). */
const unpaid = computed(() => dashboard.unpaidCount)

/** 현재 라우트 이름과 비교해 활성 여부를 판단한다. */
function isActive(name) {
  return route.name === name
}
</script>

<template>
  <aside class="sidebar" aria-label="주 메뉴">
    <!-- 브랜드 -->
    <div class="sidebar__brand">
      <svg class="sidebar__logo" viewBox="0 0 64 64" aria-hidden="true">
        <rect width="64" height="64" rx="14" fill="#1d4ed8" />
        <path
          d="M12 38h40l-4-11a6 6 0 0 0-5.6-4H21.6A6 6 0 0 0 16 27z"
          fill="#bfdbfe"
        />
        <circle cx="21" cy="42" r="4.5" fill="#f8fafc" />
        <circle cx="43" cy="42" r="4.5" fill="#f8fafc" />
        <rect
          x="24"
          y="16"
          width="16"
          height="8"
          rx="2"
          fill="none"
          stroke="#4ade80"
          stroke-width="2.5"
        />
      </svg>
      <div>
        <div class="sidebar__title">모빌리티 비전 AI</div>
        <div class="sidebar__subtitle">관리자 콘솔 v1.0</div>
      </div>
    </div>

    <!-- 메뉴 -->
    <nav class="sidebar__nav">
      <div class="sidebar__group-label">운영</div>
      <RouterLink
        v-for="m in operationMenu"
        :key="m.name"
        :to="m.to"
        class="nav-item"
        :class="{ 'is-active': isActive(m.name) }"
        :aria-current="isActive(m.name) ? 'page' : undefined"
      >
        <AppIcon :name="m.icon" :size="18" class="nav-item__icon" />
        <span class="nav-item__label">{{ m.label }}</span>
        <span
          v-if="m.badge === 'unpaid' && unpaid > 0"
          class="nav-item__badge"
          :aria-label="`미정산 ${unpaid}건`"
        >{{ unpaid }}</span>
      </RouterLink>

      <div class="sidebar__group-label">자산 · 분석</div>
      <RouterLink
        v-for="m in assetMenu"
        :key="m.name"
        :to="m.to"
        class="nav-item"
        :class="{ 'is-active': isActive(m.name) }"
        :aria-current="isActive(m.name) ? 'page' : undefined"
      >
        <AppIcon :name="m.icon" :size="18" class="nav-item__icon" />
        <span class="nav-item__label">{{ m.label }}</span>
      </RouterLink>
    </nav>

    <!-- 하단 정보 -->
    <div class="sidebar__footer">
      <div>Backend :8080 · AI :8000</div>
      <div>교재 부록 예제 프로젝트</div>
    </div>
  </aside>
</template>
