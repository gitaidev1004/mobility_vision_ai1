/**
 * 라우터 설정 (계약서 5장)
 * 라우트: /login, /(Dashboard), /recognition, /vehicles, /events,
 *        /parking, /maintenance, /pdm, /benchmark
 * - meta.requiresAuth 가 true 인 라우트는 토큰이 없으면 /login 으로 리다이렉트한다.
 * - meta.title 은 상단바 제목과 document.title 에 함께 쓴다.
 * - 화면 컴포넌트는 동적 import 로 코드 분할한다.
 */
import { createRouter, createWebHistory } from 'vue-router'
import { readToken } from '@/api/http'

const routes = [
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/LoginView.vue'),
    meta: { title: '로그인', public: true },
  },
  {
    path: '/',
    name: 'dashboard',
    component: () => import('@/views/DashboardView.vue'),
    meta: { requiresAuth: true, title: '대시보드', icon: 'dashboard' },
  },
  {
    path: '/recognition',
    name: 'recognition',
    component: () => import('@/views/RecognitionView.vue'),
    meta: { requiresAuth: true, title: '번호판 인식', icon: 'scan' },
  },
  {
    path: '/vehicles',
    name: 'vehicles',
    component: () => import('@/views/VehiclesView.vue'),
    meta: { requiresAuth: true, title: '차량 관리', icon: 'car' },
  },
  {
    path: '/events',
    name: 'events',
    component: () => import('@/views/EventsView.vue'),
    meta: { requiresAuth: true, title: '출입 이벤트', icon: 'gate' },
  },
  {
    path: '/parking',
    name: 'parking',
    component: () => import('@/views/ParkingView.vue'),
    meta: { requiresAuth: true, title: '주차 · 정산', icon: 'parking' },
  },
  {
    path: '/maintenance',
    name: 'maintenance',
    component: () => import('@/views/MaintenanceView.vue'),
    meta: { requiresAuth: true, title: '정비 이력', icon: 'wrench' },
  },
  {
    path: '/pdm',
    name: 'pdm',
    component: () => import('@/views/PdmView.vue'),
    meta: { requiresAuth: true, title: '예지보전', icon: 'pulse' },
  },
  {
    path: '/benchmark',
    name: 'benchmark',
    component: () => import('@/views/BenchmarkView.vue'),
    meta: { requiresAuth: true, title: '모델 성능', icon: 'chart' },
  },
  {
    // 정의되지 않은 경로는 대시보드로 보낸다.
    path: '/:pathMatch(.*)*',
    redirect: '/',
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  // 라우트 전환 시 항상 스크롤을 최상단으로 올린다.
  scrollBehavior: () => ({ top: 0 }),
})

/**
 * 전역 내비게이션 가드
 * - 토큰이 없고 보호된 라우트라면 /login 으로 보내고, 원래 목적지를 redirect 쿼리에 남긴다.
 * - 이미 로그인한 사용자가 /login 에 접근하면 대시보드로 되돌린다.
 */
router.beforeEach((to) => {
  const authenticated = !!readToken()

  if (to.meta?.requiresAuth && !authenticated) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }

  if (to.name === 'login' && authenticated) {
    return { path: '/' }
  }

  return true
})

/** 브라우저 탭 제목 갱신 */
router.afterEach((to) => {
  const base = '모빌리티 비전 AI 관리자'
  document.title = to.meta?.title ? `${to.meta.title} · ${base}` : base
})

export default router
