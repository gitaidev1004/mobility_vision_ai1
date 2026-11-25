# Spring Boot 4 백엔드 (멀티스테이지 빌드)
# 빌드:  docker build -t mobility-backend ./backend
FROM eclipse-temurin:21-jdk AS build
WORKDIR /src

# Gradle 래퍼와 빌드 스크립트를 먼저 복사해 의존성 캐시 레이어를 만든다.
COPY gradlew settings.gradle build.gradle gradle.properties ./
COPY gradle ./gradle
RUN chmod +x gradlew && ./gradlew --no-daemon dependencies || true

COPY src ./src
RUN ./gradlew --no-daemon clean bootJar -x test

# ---------------------------------------------------------------- 런타임
FROM eclipse-temurin:21-jre
WORKDIR /app

# 루트가 아닌 사용자로 실행한다.
RUN useradd -r -u 1001 -m appuser \
    && mkdir -p /app/data/uploads \
    && chown -R appuser:appuser /app

COPY --from=build /src/build/libs/*.jar app.jar
RUN chown appuser:appuser /app/app.jar
USER appuser

ENV TZ=Asia/Seoul \
    SPRING_PROFILES_ACTIVE=prod \
    JAVA_OPTS="-XX:MaxRAMPercentage=70"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["sh", "-c", "wget -qO- http://localhost:8080/actuator/health | grep -q UP"]

ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar /app/app.jar"]
