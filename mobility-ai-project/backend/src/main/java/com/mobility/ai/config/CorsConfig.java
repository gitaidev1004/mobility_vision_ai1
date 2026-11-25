package com.mobility.ai.config;

import java.util.List;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

/**
 * CORS 설정. 프런트엔드(Vite dev server, 5173)에서 직접 호출하는 경우를 허용한다.
 *
 * <p>계약서 1장대로 운영 구성에서는 Vite 프록시({@code /api} → 8080)를 쓰므로
 * CORS 가 필요 없지만, 교재 실습 중 브라우저에서 바로 호출하는 상황을 위해
 * 허용 출처를 {@code app.cors.allowed-origins} 로 열어 둔다.
 */
@Configuration
public class CorsConfig {

    private final AppProperties appProperties;

    /**
     * @param appProperties {@code app.*} 설정
     */
    public CorsConfig(AppProperties appProperties) {
        this.appProperties = appProperties;
    }

    /**
     * CORS 설정 소스. {@code SecurityConfig} 가 이 빈을 주입받아 사용한다.
     *
     * @return CORS 설정 소스
     */
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();
        config.setAllowedOrigins(appProperties.getCors().getAllowedOrigins());
        config.setAllowedMethods(List.of("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
        config.setAllowedHeaders(List.of("*"));
        config.setExposedHeaders(List.of("Location", "Content-Disposition"));
        config.setAllowCredentials(true);
        config.setMaxAge(3600L);

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return source;
    }
}
