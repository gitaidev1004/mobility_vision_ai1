package com.mobility.ai.config;

import java.net.http.HttpClient;
import java.time.Duration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

/**
 * AI 추론 서비스 호출용 {@link RestClient} 구성.
 *
 * <p>Spring Boot 4 에서 권장되는 동기 HTTP 클라이언트는 {@code RestClient} 다.
 * JDK 내장 {@link HttpClient} 를 감싼 {@link JdkClientHttpRequestFactory} 를 써
 * 추가 의존성(Apache HttpClient 등) 없이 연결/읽기 타임아웃을 지정한다.
 *
 * <ul>
 *   <li>연결 타임아웃: {@code app.ai.connect-timeout-ms} (기본 2초) — AI 서비스가
 *       내려가 있을 때 요청이 길게 매달리지 않도록 짧게 잡는다.</li>
 *   <li>읽기 타임아웃: {@code app.ai.read-timeout-ms} (기본 15초) — CPU 추론이
 *       느릴 수 있으므로 넉넉히 준다.</li>
 * </ul>
 *
 * <p>오류 상태(4xx/5xx)는 기본 동작대로 예외가 되고, {@code AiInferenceClient} 가
 * 이를 잡아 폴백 결과로 변환한다. 즉 <b>AI 장애가 백엔드 500 으로 번지지 않는다.</b>
 */
@Configuration
public class AiClientConfig {

    private final AppProperties appProperties;

    /**
     * @param appProperties {@code app.*} 설정
     */
    public AiClientConfig(AppProperties appProperties) {
        this.appProperties = appProperties;
    }

    /**
     * AI 서비스 전용 RestClient.
     *
     * @return baseUrl 과 타임아웃이 설정된 RestClient
     */
    @Bean
    public RestClient aiRestClient() {
        AppProperties.Ai ai = appProperties.getAi();

        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(ai.getConnectTimeoutMs()))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();

        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(Duration.ofMillis(ai.getReadTimeoutMs()));

        return RestClient.builder()
                .baseUrl(ai.getBaseUrl())
                .requestFactory(requestFactory)
                .build();
    }
}
