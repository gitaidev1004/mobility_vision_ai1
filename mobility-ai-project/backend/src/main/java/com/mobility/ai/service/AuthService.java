package com.mobility.ai.service;

import com.mobility.ai.config.AppProperties;
import com.mobility.ai.domain.AppUser;
import com.mobility.ai.dto.AuthDtos.LoginRequest;
import com.mobility.ai.dto.AuthDtos.LoginResponse;
import com.mobility.ai.dto.AuthDtos.MeResponse;
import com.mobility.ai.repository.AppUserRepository;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.DisabledException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.JwsHeader;
import org.springframework.security.oauth2.jwt.JwtClaimsSet;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtEncoderParameters;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 인증 서비스. 아이디/비밀번호를 검증하고 HS256 JWT 를 발급한다
 * (교재 6장 "JWT 인증" 절 인용 대상).
 *
 * <p>발급 클레임
 * <ul>
 *   <li>{@code iss} — {@code app.jwt.issuer}</li>
 *   <li>{@code sub} — 로그인 아이디</li>
 *   <li>{@code iat}/{@code exp} — 발급/만료(초 단위 절사)</li>
 *   <li>{@code roles} — 권한 1개 문자열(예: {@code ADMIN}).
 *       {@code SecurityConfig} 가 {@code ROLE_} 접두사를 붙인다.</li>
 *   <li>{@code scope} — 동일 값을 공백 구분 문자열로도 넣어, 표준 스코프 기반
 *       권한 매핑과도 호환시킨다.</li>
 *   <li>{@code displayName} — 화면 표시용 이름</li>
 * </ul>
 */
@Service
public class AuthService {

    private static final Logger log = LoggerFactory.getLogger(AuthService.class);

    private final AppUserRepository appUserRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtEncoder jwtEncoder;
    private final AppProperties appProperties;

    /**
     * @param appUserRepository 계정 리포지토리
     * @param passwordEncoder   BCrypt 인코더
     * @param jwtEncoder        JWT 발급기
     * @param appProperties     {@code app.*} 설정
     */
    public AuthService(AppUserRepository appUserRepository,
                       PasswordEncoder passwordEncoder,
                       JwtEncoder jwtEncoder,
                       AppProperties appProperties) {
        this.appUserRepository = appUserRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtEncoder = jwtEncoder;
        this.appProperties = appProperties;
    }

    /**
     * 로그인 처리.
     *
     * @param request 아이디/비밀번호
     * @return 토큰과 사용자 정보
     * @throws BadCredentialsException 아이디가 없거나 비밀번호가 틀릴 때
     * @throws DisabledException       비활성 계정일 때
     */
    @Transactional(readOnly = true)
    public LoginResponse login(LoginRequest request) {
        AppUser user = appUserRepository.findByUsername(request.username())
                .orElseThrow(() -> new BadCredentialsException("아이디 또는 비밀번호가 올바르지 않습니다."));

        if (!user.isEnabled()) {
            throw new DisabledException("비활성화된 계정입니다.");
        }

        if (!passwordEncoder.matches(request.password(), user.getPasswordHash())) {
            log.info("로그인 실패(비밀번호 불일치): {}", request.username());
            throw new BadCredentialsException("아이디 또는 비밀번호가 올바르지 않습니다.");
        }

        long expiresSeconds = appProperties.getJwt().getExpiresSeconds();
        String token = issueToken(user, expiresSeconds);

        log.info("로그인 성공: {} ({})", user.getUsername(), user.getRole());
        return new LoginResponse(
                token,
                "Bearer",
                expiresSeconds,
                user.getUsername(),
                user.getRole().name(),
                user.getDisplayName()
        );
    }

    /**
     * 현재 로그인 사용자 정보를 조회한다.
     *
     * @param username 토큰의 subject
     * @return 사용자 정보
     * @throws BadCredentialsException 계정을 찾을 수 없을 때
     */
    @Transactional(readOnly = true)
    public MeResponse me(String username) {
        AppUser user = appUserRepository.findByUsername(username)
                .orElseThrow(() -> new BadCredentialsException("계정을 찾을 수 없습니다: " + username));
        return new MeResponse(user.getUsername(), user.getRole().name(), user.getDisplayName());
    }

    /**
     * 로그아웃을 기록한다.
     *
     * <p>JWT 는 상태를 서버에 두지 않으므로 토큰을 무효화할 방법이 없다. 따라서 이 메서드는
     * 감사 로그만 남긴다. 토큰을 즉시 폐기해야 하는 요구가 생기면 여기서 토큰 식별자(jti)를
     * 블랙리스트 테이블에 넣고, 디코더 단계에서 조회하도록 확장하면 된다.
     *
     * @param username 토큰의 subject. 비인증 상태로 호출되면 null 일 수 있다.
     */
    public void logout(String username) {
        if (username == null) {
            log.debug("비인증 상태의 로그아웃 요청을 무시한다");
            return;
        }
        log.info("로그아웃: username={}", username);
    }

    /**
     * JWT 를 발급한다.
     *
     * @param user           대상 계정
     * @param expiresSeconds 만료까지 초
     * @return 직렬화된 JWT 문자열
     */
    private String issueToken(AppUser user, long expiresSeconds) {
        Instant now = Instant.now().truncatedTo(ChronoUnit.SECONDS);
        String role = user.getRole().name();

        JwtClaimsSet claims = JwtClaimsSet.builder()
                .issuer(appProperties.getJwt().getIssuer())
                .issuedAt(now)
                .expiresAt(now.plusSeconds(expiresSeconds))
                .subject(user.getUsername())
                .claim("roles", role)
                .claim("scope", role)
                .claim("displayName", user.getDisplayName())
                .build();

        JwsHeader header = JwsHeader.with(MacAlgorithm.HS256).build();
        return jwtEncoder.encode(JwtEncoderParameters.from(header, claims)).getTokenValue();
    }
}
