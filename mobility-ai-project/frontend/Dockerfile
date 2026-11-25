# Vue 3 프런트엔드 (빌드 후 nginx 로 정적 서빙)
# 빌드:  docker build -t mobility-frontend ./frontend
FROM node:20-alpine AS build
WORKDIR /src

COPY package.json package-lock.json* ./
# package-lock.json 이 있으면 ci, 없으면 install 로 받는다.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build

# ---------------------------------------------------------------- 런타임
FROM nginx:1.27-alpine
COPY --from=build /src/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1
CMD ["nginx", "-g", "daemon off;"]
