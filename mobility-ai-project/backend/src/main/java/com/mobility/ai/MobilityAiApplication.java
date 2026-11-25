package com.mobility.ai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/**
 * 모빌리티 비전 AI 백엔드 진입점.
 *
 * <p>{@code @ConfigurationPropertiesScan} 으로 {@code com.mobility.ai.config.AppProperties}
 * 를 자동 등록한다({@code @EnableConfigurationProperties} 를 따로 쓰지 않는다).
 *
 * <p>실행: {@code ./gradlew bootRun} (기본 프로파일 dev, 포트 8080)
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class MobilityAiApplication {

    /**
     * 애플리케이션을 기동한다.
     *
     * @param args 커맨드라인 인자
     */
    public static void main(String[] args) {
        SpringApplication.run(MobilityAiApplication.class, args);
    }
}
