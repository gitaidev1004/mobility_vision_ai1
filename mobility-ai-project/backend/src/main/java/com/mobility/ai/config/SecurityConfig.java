package com.mobility.ai.config;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.convert.converter.Converter;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AbstractAuthenticationToken;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.web.cors.CorsConfigurationSource;

/**
 * Spring Security 7 설정. 람다 DSL 만 사용한다(계약서 0장 규칙).
 *
 * <h2>정책</h2>
 * <ul>
 *   <li>세션을 만들지 않는 <b>stateless</b> 리소스 서버</li>
 *   <li>CSRF 비활성화(토큰 기반 API)</li>
 *   <li>{@code /api/v1/auth/**}, {@code /actuator/health}, CORS 프리플라이트는 permitAll</li>
 *   <li>나머지는 모두 authenticated — 계약서대로 {@code /api/v1/dashboard/summary} 도 로그인 필요</li>
 *   <li>권한은 JWT 의 {@code roles} 또는 {@code scope} 클레임을 읽어 {@code ROLE_} 접두사를 붙인다</li>
 * </ul>
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    /** 권한 접두사 */
    private static final String ROLE_PREFIX = "ROLE_";

    /**
     * 보안 필터 체인.
     *
     * @param http                    HttpSecurity 빌더
     * @param corsConfigurationSource CORS 설정(CorsConfig 가 제공)
     * @return 필터 체인
     * @throws Exception 구성 실패 시
     */
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http,
                                                   CorsConfigurationSource corsConfigurationSource)
            throws Exception {
        http
                .csrf(csrf -> csrf.disable())
                .cors(cors -> cors.configurationSource(corsConfigurationSource))
                .sessionManagement(session ->
                        session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(auth -> auth
                        // CORS 프리플라이트는 인증 없이 통과시킨다.
                        .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                        // 로그인/토큰 발급
                        .requestMatchers("/api/v1/auth/**").permitAll()
                        // 컨테이너 헬스체크
                        .requestMatchers("/actuator/health", "/actuator/health/**").permitAll()
                        // 그 외 전부 인증 필요 (대시보드 요약 포함)
                        .anyRequest().authenticated()
                )
                .oauth2ResourceServer(oauth2 -> oauth2
                        .jwt(jwt -> jwt.jwtAuthenticationConverter(jwtAuthenticationConverter()))
                )
                .httpBasic(basic -> basic.disable())
                .formLogin(form -> form.disable());

        return http.build();
    }

    /**
     * BCrypt 비밀번호 인코더. 시드 해시({@code $2a$10$...})와 호환된다.
     *
     * @return 인코더
     */
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    /**
     * JWT → 인증 객체 변환기.
     * {@code roles}(문자열 또는 배열)와 {@code scope}(공백 구분 문자열) 양쪽을 읽어
     * {@code ROLE_ADMIN} 같은 권한으로 변환한다.
     *
     * <p>이 메서드를 {@code @Bean} 으로 노출하지 않는 이유: Spring Boot 는
     * 컨텍스트의 {@link Converter} 빈을 MVC 의 {@code FormattingConversionService}
     * 에 자동 등록한다. 요청 파라미터 변환과 무관한 이 변환기가 거기 섞이면
     * 혼란스럽기만 하므로, 필터 체인 안에서만 쓰도록 private 로 둔다.
     *
     * @return 변환기
     */
    private Converter<Jwt, AbstractAuthenticationToken> jwtAuthenticationConverter() {
        return jwt -> new JwtAuthenticationToken(jwt, extractAuthorities(jwt), jwt.getSubject());
    }

    /**
     * {@code roles} / {@code scope} 클레임에서 권한을 뽑는다.
     *
     * @param jwt 디코딩된 토큰
     * @return 권한 컬렉션(중복 없음)
     */
    private Collection<GrantedAuthority> extractAuthorities(Jwt jwt) {
        List<String> names = new ArrayList<>();

        Object roles = jwt.getClaim("roles");
        if (roles instanceof String s) {
            for (String part : s.split("[\\s,]+")) {
                if (!part.isBlank()) {
                    names.add(part.trim());
                }
            }
        } else if (roles instanceof Collection<?> collection) {
            for (Object item : collection) {
                if (item != null && !String.valueOf(item).isBlank()) {
                    names.add(String.valueOf(item).trim());
                }
            }
        }

        Object scope = jwt.getClaim("scope");
        if (scope instanceof String s) {
            for (String part : s.split("[\\s,]+")) {
                if (!part.isBlank()) {
                    names.add(part.trim());
                }
            }
        } else if (scope instanceof Collection<?> collection) {
            for (Object item : collection) {
                if (item != null && !String.valueOf(item).isBlank()) {
                    names.add(String.valueOf(item).trim());
                }
            }
        }

        List<GrantedAuthority> authorities = new ArrayList<>();
        List<String> seen = new ArrayList<>();
        for (String name : names) {
            String authority = name.startsWith(ROLE_PREFIX) ? name : ROLE_PREFIX + name;
            if (!seen.contains(authority)) {
                seen.add(authority);
                authorities.add(new SimpleGrantedAuthority(authority));
            }
        }
        return authorities;
    }
}
