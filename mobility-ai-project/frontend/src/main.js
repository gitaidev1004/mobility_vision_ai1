/**
 * 애플리케이션 진입점
 * - 전역 CSS 는 토큰 → 기본 → 컴포넌트 순서로 읽어야 변수 참조가 깨지지 않는다.
 * - pinia, vue-router 를 설치하고 #app 에 마운트한다.
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

import './assets/styles/tokens.css'
import './assets/styles/base.css'
import './assets/styles/components.css'

const app = createApp(App)

app.use(createPinia())
app.use(router)

// 개발 중 처리되지 않은 렌더 오류를 콘솔에 남긴다(교재 디버깅용).
app.config.errorHandler = (err, instance, info) => {
  console.error('[Vue 전역 오류]', info, err)
}

app.mount('#app')
