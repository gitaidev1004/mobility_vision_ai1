/**
 * 인증 스토어 (계약서 5장)
 * - 액세스 토큰과 로그인 사용자 정보를 보관한다.
 * - 새로고침에도 세션이 유지되도록 localStorage 와 동기화한다.
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as authApi from '@/api/auth'
import { readToken, writeToken, readUser, writeUser, extractMessage } from '@/api/http'
import { ROLE_LABEL } from '@/utils/format'

export const useAuthStore = defineStore('auth', () => {
  // ---------------- 상태 ----------------
  /** 액세스 토큰(JWT). 초기값은 localStorage 에서 복구한다. */
  const token = ref(readToken())
  /** {username, role, displayName} */
  const user = ref(readUser())
  /** 토큰 만료까지 남은 초(로그인 응답의 expiresIn) */
  const expiresIn = ref(0)
  const loading = ref(false)
  const error = ref('')

  // ---------------- 계산 속성 ----------------
  /** 토큰 보유 여부 = 인증 상태 */
  const isAuthenticated = computed(() => !!token.value)
  /** 화면 표시용 사용자명(없으면 'guest') */
  const username = computed(() => user.value?.username || 'guest')
  /** 역할 enum (ADMIN|OPERATOR|VIEWER) */
  const role = computed(() => user.value?.role || 'VIEWER')
  /** 역할 한국어 라벨 */
  const roleLabel = computed(() => ROLE_LABEL[role.value] || role.value)
  /** 표시 이름(없으면 username) */
  const displayName = computed(() => user.value?.displayName || username.value)
  /** 아바타용 머리글자 1자 */
  const initial = computed(() => (displayName.value || 'U').trim().charAt(0).toUpperCase())
  /** 쓰기 권한: VIEWER 는 조회만 가능하다. */
  const canWrite = computed(() => role.value === 'ADMIN' || role.value === 'OPERATOR')
  /** 관리자 전용 기능(사용자/모델 관리) 권한 */
  const isAdmin = computed(() => role.value === 'ADMIN')

  // ---------------- 액션 ----------------
  /**
   * 로그인. 성공하면 토큰/사용자 정보를 저장하고 true 를 반환한다.
   * @param {string} id 사용자명
   * @param {string} pw 비밀번호
   */
  async function login(id, pw) {
    loading.value = true
    error.value = ''
    try {
      const data = await authApi.login({ username: id, password: pw })
      token.value = data.accessToken
      expiresIn.value = data.expiresIn ?? 3600
      user.value = {
        username: data.username,
        role: data.role,
        displayName: data.displayName,
      }
      writeToken(token.value)
      writeUser(user.value)
      return true
    } catch (e) {
      // 401 은 자격 증명 오류이므로 친절한 문구로 바꿔준다.
      error.value =
        e?.response?.status === 401
          ? '아이디 또는 비밀번호가 올바르지 않습니다.'
          : extractMessage(e)
      return false
    } finally {
      loading.value = false
    }
  }

  /**
   * 로그아웃. 서버 호출은 실패해도 무시하고 로컬 세션을 반드시 비운다.
   */
  async function logout() {
    try {
      if (token.value) await authApi.logout()
    } finally {
      clearSession()
    }
  }

  /** 로컬 세션만 비운다(401 인터셉터에서 호출). */
  function clearSession() {
    token.value = null
    user.value = null
    expiresIn.value = 0
    writeToken(null)
    writeUser(null)
  }

  /** 오류 메시지 초기화(폼 재입력 시). */
  function clearError() {
    error.value = ''
  }

  return {
    token,
    user,
    expiresIn,
    loading,
    error,
    isAuthenticated,
    username,
    role,
    roleLabel,
    displayName,
    initial,
    canWrite,
    isAdmin,
    login,
    logout,
    clearSession,
    clearError,
  }
})
