package com.mobility.ai.config;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * {@code app.*} 설정을 타입 세이프하게 묶은 프로퍼티 클래스.
 * record 가 아니라 가변 클래스로 두어 YAML 의 중첩 구조를 그대로 받는다.
 *
 * <p>{@code MobilityAiApplication} 의 {@code @ConfigurationPropertiesScan} 으로 등록된다.
 */
@ConfigurationProperties(prefix = "app")
public class AppProperties {

    private final Jwt jwt = new Jwt();
    private final Ai ai = new Ai();
    private final Storage storage = new Storage();
    private final Parking parking = new Parking();
    private final Cors cors = new Cors();
    private final Recognition recognition = new Recognition();
    private final Pdm pdm = new Pdm();

    public Jwt getJwt() {
        return jwt;
    }

    public Ai getAi() {
        return ai;
    }

    public Storage getStorage() {
        return storage;
    }

    public Parking getParking() {
        return parking;
    }

    public Cors getCors() {
        return cors;
    }

    public Recognition getRecognition() {
        return recognition;
    }

    public Pdm getPdm() {
        return pdm;
    }

    /** {@code app.jwt.*} */
    public static class Jwt {
        /** HMAC-SHA256 대칭키(32바이트 이상) */
        private String secret = "mobility-ai-dev-secret-key-change-me-32bytes-min!";
        /** 토큰 만료(초) */
        private long expiresSeconds = 3600L;
        /** 발급자 클레임 */
        private String issuer = "mobility-ai";

        public String getSecret() {
            return secret;
        }

        public void setSecret(String secret) {
            this.secret = secret;
        }

        public long getExpiresSeconds() {
            return expiresSeconds;
        }

        public void setExpiresSeconds(long expiresSeconds) {
            this.expiresSeconds = expiresSeconds;
        }

        public String getIssuer() {
            return issuer;
        }

        public void setIssuer(String issuer) {
            this.issuer = issuer;
        }
    }

    /** {@code app.ai.*} */
    public static class Ai {
        private String baseUrl = "http://localhost:8000";
        private int connectTimeoutMs = 2000;
        private int readTimeoutMs = 15000;

        public String getBaseUrl() {
            return baseUrl;
        }

        public void setBaseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
        }

        public int getConnectTimeoutMs() {
            return connectTimeoutMs;
        }

        public void setConnectTimeoutMs(int connectTimeoutMs) {
            this.connectTimeoutMs = connectTimeoutMs;
        }

        public int getReadTimeoutMs() {
            return readTimeoutMs;
        }

        public void setReadTimeoutMs(int readTimeoutMs) {
            this.readTimeoutMs = readTimeoutMs;
        }
    }

    /** {@code app.storage.*} */
    public static class Storage {
        private String uploadDir = "./data/uploads";

        public String getUploadDir() {
            return uploadDir;
        }

        public void setUploadDir(String uploadDir) {
            this.uploadDir = uploadDir;
        }
    }

    /** {@code app.parking.*} */
    public static class Parking {
        /** 주차장 총 수용 대수 */
        private int capacity = 120;

        public int getCapacity() {
            return capacity;
        }

        public void setCapacity(int capacity) {
            this.capacity = capacity;
        }
    }

    /** {@code app.cors.*} */
    public static class Cors {
        private List<String> allowedOrigins =
                List.of("http://localhost:5173", "http://127.0.0.1:5173");

        public List<String> getAllowedOrigins() {
            return allowedOrigins;
        }

        public void setAllowedOrigins(List<String> allowedOrigins) {
            this.allowedOrigins = allowedOrigins;
        }
    }

    /** {@code app.recognition.*} */
    public static class Recognition {
        /** 통행 허용 최소 신뢰도 */
        private double minConfidence = 0.60;
        /** 중복 이벤트 억제 창(초) */
        private int dedupWindowSeconds = 30;

        public double getMinConfidence() {
            return minConfidence;
        }

        public void setMinConfidence(double minConfidence) {
            this.minConfidence = minConfidence;
        }

        public int getDedupWindowSeconds() {
            return dedupWindowSeconds;
        }

        public void setDedupWindowSeconds(int dedupWindowSeconds) {
            this.dedupWindowSeconds = dedupWindowSeconds;
        }
    }

    /** {@code app.pdm.*} */
    public static class Pdm {
        /** 예측 1회에 사용할 최근 센서 건수 */
        private int readingsWindow = 50;

        public int getReadingsWindow() {
            return readingsWindow;
        }

        public void setReadingsWindow(int readingsWindow) {
            this.readingsWindow = readingsWindow;
        }
    }
}
