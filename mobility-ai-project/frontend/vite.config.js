// Vite 설정 파일
// - 개발 서버는 5173 포트로 고정한다(계약서 1장).
// - '/api' 로 시작하는 모든 요청은 Spring Boot 백엔드(8080)로 프록시한다.
//   이렇게 하면 프론트엔드 코드에서는 CORS 설정 없이 상대 경로만 사용할 수 있다.
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      // '@' 를 src 디렉터리의 별칭으로 등록해 import 경로를 짧게 유지한다.
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // 백엔드가 이미 /api/v1 prefix 를 사용하므로 rewrite 는 하지 않는다.
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    // 차트/UI 라이브러리를 사용하지 않으므로 번들이 매우 작다.
    chunkSizeWarningLimit: 600,
  },
})
