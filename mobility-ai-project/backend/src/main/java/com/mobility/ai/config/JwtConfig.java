package com.mobility.ai.config;

import com.nimbusds.jose.JWSAlgorithm;
import com.nimbusds.jose.jwk.OctetSequenceKey;
import com.nimbusds.jose.jwk.source.ImmutableJWKSet;
import com.nimbusds.jose.jwk.source.JWKSource;
import com.nimbusds.jose.jwk.JWKSet;
import com.nimbusds.jose.proc.SecurityContext;
import java.nio.charset.StandardCharsets;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder;
import org.springframework.security.oauth2.jwt.NimbusJwtEncoder;

/**
 * JWT 인코더/디코더 구성.
 *
 * <p>교재 목적상 외부 인증 서버 없이 <b>HMAC-SHA256 대칭키</b> 하나로 토큰을
 * 발급하고 검증한다. {@code spring-boot-starter-oauth2-resource-server} 가 제공하는
 * Nimbus 구현({@link NimbusJwtEncoder} / {@link NimbusJwtDecoder})을 그대로 쓰므로
 * 별도 JWT 라이브러리를 추가하지 않는다.
 *
 * <p>비밀키는 {@code app.jwt.secret} 이며 HS256 규격상 <b>32바이트(256비트) 이상</b>
 * 이어야 한다. 짧으면 기동 시점에 바로 실패시켜 운영 사고를 막는다.
 */
@Configuration
public class JwtConfig {

    /** HS256 최소 키 길이(바이트) */
    private static final int MIN_SECRET_BYTES = 32;

    private final AppProperties appProperties;

    /**
     * @param appProperties {@code app.*} 설정
     */
    public JwtConfig(AppProperties appProperties) {
        this.appProperties = appProperties;
    }

    /**
     * 설정의 비밀키로 HMAC 키를 만든다.
     *
     * @return HS256 서명용 대칭키
     * @throws IllegalStateException 키가 32바이트 미만일 때
     */
    @Bean
    public SecretKeySpec jwtSecretKey() {
        String secret = appProperties.getJwt().getSecret();
        byte[] raw = secret == null ? new byte[0] : secret.getBytes(StandardCharsets.UTF_8);
        if (raw.length < MIN_SECRET_BYTES) {
            throw new IllegalStateException(
                    "app.jwt.secret 은 32바이트 이상이어야 합니다. 현재 길이: " + raw.length);
        }
        return new SecretKeySpec(raw, "HmacSHA256");
    }

    /**
     * 토큰 발급기. 대칭키 1개를 담은 JWK 소스를 쓴다.
     *
     * @param secretKey HMAC 키
     * @return JWT 인코더
     */
    @Bean
    public JwtEncoder jwtEncoder(SecretKeySpec secretKey) {
        OctetSequenceKey jwk = new OctetSequenceKey.Builder(secretKey.getEncoded())
                .algorithm(JWSAlgorithm.HS256)
                .build();
        JWKSource<SecurityContext> jwkSource = new ImmutableJWKSet<>(new JWKSet(jwk));
        return new NimbusJwtEncoder(jwkSource);
    }

    /**
     * 토큰 검증기. HS256 만 허용한다.
     *
     * @param secretKey HMAC 키
     * @return JWT 디코더
     */
    @Bean
    public JwtDecoder jwtDecoder(SecretKeySpec secretKey) {
        return NimbusJwtDecoder.withSecretKey(secretKey)
                .macAlgorithm(org.springframework.security.oauth2.jose.jws.MacAlgorithm.HS256)
                .build();
    }
}
