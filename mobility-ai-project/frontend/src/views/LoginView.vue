<script setup>
/**
 * 로그인 화면 (/login)
 * - POST /api/v1/auth/login 으로 인증하고, 성공하면 redirect 쿼리 목적지로 이동한다.
 * - 교재 실습 편의를 위해 시드 계정 3개를 클릭 한 번으로 채울 수 있게 했다.
 */
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppIcon from '@/components/common/AppIcon.vue'
import { useAuthStore } from '@/stores/auth'
import { toastSuccess } from '@/utils/toast'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

const username = ref('admin')
const password = ref('')

/** 계약서 2장의 시드 계정 */
const seedAccounts = [
  { username: 'admin', password: 'admin123!', role: 'ADMIN', desc: '전체 권한' },
  { username: 'operator', password: 'operator123!', role: 'OPERATOR', desc: '운영 · 정산' },
  { username: 'viewer', password: 'viewer123!', role: 'VIEWER', desc: '조회 전용' },
]

/** 시드 계정 버튼을 누르면 입력란을 채운다. */
function fill(acc) {
  username.value = acc.username
  password.value = acc.password
  auth.clearError()
}

/** 로그인 제출. */
async function onSubmit() {
  const ok = await auth.login(username.value.trim(), password.value)
  if (!ok) return
  toastSuccess(`${auth.displayName}님, 환영합니다.`)
  // 가드가 남긴 원래 목적지로 복귀한다.
  const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/'
  router.replace(redirect)
}

onMounted(() => {
  auth.clearError()
})
</script>

<template>
  <div class="login">
    <!-- 좌측 브랜드 패널 -->
    <section class="login__brand" aria-hidden="true">
      <div class="login__brand-inner">
        <svg class="login__logo" viewBox="0 0 64 64">
          <rect width="64" height="64" rx="16" fill="#2563eb" />
          <path d="M12 38h40l-4-11a6 6 0 0 0-5.6-4H21.6A6 6 0 0 0 16 27z" fill="#bfdbfe" />
          <circle cx="21" cy="42" r="4.5" fill="#f8fafc" />
          <circle cx="43" cy="42" r="4.5" fill="#f8fafc" />
          <rect x="24" y="16" width="16" height="8" rx="2" fill="none" stroke="#4ade80" stroke-width="2.5" />
        </svg>
        <h1 class="login__brand-title">모빌리티 비전 AI</h1>
        <p class="login__brand-desc">
          번호판 인식 · 출입 관리 · 주차 정산 · 예지보전을<br />
          하나의 콘솔에서 운영합니다.
        </p>

        <ul class="login__features">
          <li><AppIcon name="scan" :size="16" /> YOLO + CRNN/CTC 번호판 인식</li>
          <li><AppIcon name="gate" :size="16" /> 실시간 입·출차 이벤트 처리</li>
          <li><AppIcon name="won" :size="16" /> 정책 기반 주차 요금 자동 정산</li>
          <li><AppIcon name="pulse" :size="16" /> 센서 기반 고장 예지보전</li>
        </ul>
      </div>
    </section>

    <!-- 우측 로그인 폼 -->
    <section class="login__panel">
      <div class="login__card">
        <h2 class="login__title">관리자 로그인</h2>
        <p class="login__subtitle">계정 정보를 입력해 주세요.</p>

        <form class="login__form" @submit.prevent="onSubmit">
          <div class="field">
            <label class="field__label" for="login-username">
              아이디<span class="req" aria-hidden="true">*</span>
            </label>
            <input
              id="login-username"
              v-model="username"
              class="input"
              type="text"
              autocomplete="username"
              placeholder="admin"
              required
              @input="auth.clearError()"
            />
          </div>

          <div class="field">
            <label class="field__label" for="login-password">
              비밀번호<span class="req" aria-hidden="true">*</span>
            </label>
            <input
              id="login-password"
              v-model="password"
              class="input"
              type="password"
              autocomplete="current-password"
              placeholder="••••••••"
              required
              @input="auth.clearError()"
            />
          </div>

          <!-- 오류 메시지 (role=alert 로 스크린리더에 즉시 전달) -->
          <p v-if="auth.error" class="login__error" role="alert">
            <AppIcon name="alert" :size="15" />
            {{ auth.error }}
          </p>

          <button
            type="submit"
            class="btn btn--primary btn--lg btn--block"
            :disabled="auth.loading || !username || !password"
          >
            <span v-if="auth.loading" class="spinner" aria-hidden="true" />
            {{ auth.loading ? '인증 중…' : '로그인' }}
          </button>
        </form>

        <!-- 교재 실습용 시드 계정 -->
        <div class="login__seed">
          <div class="login__seed-title">실습용 시드 계정</div>
          <div class="login__seed-list">
            <button
              v-for="acc in seedAccounts"
              :key="acc.username"
              type="button"
              class="login__seed-btn"
              :aria-label="`${acc.username} 계정으로 입력란 채우기`"
              @click="fill(acc)"
            >
              <strong>{{ acc.username }}</strong>
              <span class="badge badge--neutral badge--plain">{{ acc.role }}</span>
              <span class="login__seed-desc">{{ acc.desc }}</span>
            </button>
          </div>
        </div>

        <p class="login__hint">
          백엔드가 기동되어 있어야 로그인됩니다 (<code>http://localhost:8080</code>)
        </p>
      </div>
    </section>
  </div>
</template>

<style scoped>
.login {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 480px;
}

/* ---- 좌측 브랜드 패널 ---- */
.login__brand {
  background: var(--c-bg);
  color: var(--c-text-invert);
  display: grid;
  place-items: center;
  padding: var(--sp-8);
  /* 라디얼 그라디언트로 깊이를 만든다(이미지 파일 없이). */
  background-image: radial-gradient(circle at 25% 20%, #1e3a8a 0%, transparent 55%),
    radial-gradient(circle at 80% 85%, #0e7490 0%, transparent 50%);
  background-color: var(--c-bg);
}

.login__brand-inner {
  max-width: 420px;
}

.login__logo {
  width: 56px;
  height: 56px;
  margin-bottom: var(--sp-5);
}

.login__brand-title {
  font-size: var(--fs-3xl);
  font-weight: 800;
  letter-spacing: -0.035em;
}

.login__brand-desc {
  font-size: var(--fs-lg);
  color: #a8b6cc;
  margin-top: var(--sp-3);
  line-height: 1.7;
}

.login__features {
  margin-top: var(--sp-8);
  display: flex;
  flex-direction: column;
  gap: var(--sp-3);
}

.login__features li {
  display: flex;
  align-items: center;
  gap: var(--sp-3);
  font-size: var(--fs-md);
  color: #cbd5e1;
  font-weight: 500;
}

/* ---- 우측 폼 ---- */
.login__panel {
  background: var(--c-surface);
  display: grid;
  place-items: center;
  padding: var(--sp-8) var(--sp-6);
}

.login__card {
  width: 100%;
  max-width: 352px;
}

.login__title {
  font-size: var(--fs-xl);
  font-weight: 800;
  letter-spacing: -0.03em;
}

.login__subtitle {
  font-size: var(--fs-md);
  color: var(--c-text-sub);
  margin-top: var(--sp-1);
}

.login__form {
  display: flex;
  flex-direction: column;
  gap: var(--sp-4);
  margin-top: var(--sp-6);
}

.login__form .input {
  height: 40px;
}

.login__error {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  font-size: var(--fs-base);
  font-weight: 600;
  color: var(--c-danger);
  background: var(--c-danger-soft);
  border-radius: var(--radius-sm);
  padding: var(--sp-3);
}

/* ---- 시드 계정 ---- */
.login__seed {
  margin-top: var(--sp-6);
  padding-top: var(--sp-5);
  border-top: 1px solid var(--c-border);
}

.login__seed-title {
  font-size: var(--fs-sm);
  font-weight: 700;
  color: var(--c-text-sub);
  margin-bottom: var(--sp-3);
}

.login__seed-list {
  display: flex;
  flex-direction: column;
  gap: var(--sp-2);
}

.login__seed-btn {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  width: 100%;
  padding: 8px var(--sp-3);
  border: 1px solid var(--c-border);
  border-radius: var(--radius-sm);
  background: var(--c-surface-alt);
  font-size: var(--fs-sm);
  text-align: left;
}

.login__seed-btn:hover {
  border-color: var(--c-primary);
  background: var(--c-primary-soft);
}

.login__seed-desc {
  margin-left: auto;
  color: var(--c-text-muted);
}

.login__hint {
  margin-top: var(--sp-5);
  font-size: var(--fs-xs);
  color: var(--c-text-muted);
  text-align: center;
}

.login__hint code {
  font-family: var(--font-mono);
  background: var(--c-surface-alt);
  padding: 1px 5px;
  border-radius: 4px;
}

/* ---- 반응형: 좁은 화면에서는 브랜드 패널을 숨긴다. ---- */
@media (max-width: 960px) {
  .login {
    grid-template-columns: minmax(0, 1fr);
  }

  .login__brand {
    display: none;
  }
}
</style>
