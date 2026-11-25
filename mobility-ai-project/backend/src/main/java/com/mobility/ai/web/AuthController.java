package com.mobility.ai.web;

import com.mobility.ai.dto.AuthDtos.LoginRequest;
import com.mobility.ai.dto.AuthDtos.LoginResponse;
import com.mobility.ai.dto.AuthDtos.MeResponse;
import com.mobility.ai.service.AuthService;
import jakarta.validation.Valid;
import java.security.Principal;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 인증 컨트롤러. 계약서 4장 {@code POST /api/v1/auth/login}.
 *
 * <p>{@code /api/v1/auth/**} 는 {@code SecurityConfig} 에서 permitAll 이지만,
 * {@code /auth/me} 는 토큰이 있어야 의미가 있으므로 Principal 이 없으면 401 을 돌려준다.
 */
@RestController
@RequestMapping("/api/v1/auth")
public class AuthController {

    private final AuthService authService;

    /**
     * @param authService 인증 서비스
     */
    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    /**
     * 로그인하고 JWT 를 발급받는다.
     *
     * @param request 아이디/비밀번호
     * @return 토큰 + 사용자 정보
     */
    @PostMapping("/login")
    public LoginResponse login(@Valid @RequestBody LoginRequest request) {
        return authService.login(request);
    }

    /**
     * 현재 로그인 사용자 정보.
     *
     * @param principal 인증 주체(토큰 subject)
     * @return 사용자 정보, 비인증이면 401
     */
    @GetMapping("/me")
    public ResponseEntity<MeResponse> me(Principal principal) {
        if (principal == null) {
            return ResponseEntity.status(401).build();
        }
        return ResponseEntity.ok(authService.me(principal.getName()));
    }

    /**
     * 로그아웃.
     *
     * <p>JWT 는 서버에 세션을 두지 않으므로 서버가 할 일은 없다. 토큰 폐기는 클라이언트가
     * 저장소에서 토큰을 지우는 것으로 끝난다. 그럼에도 이 엔드포인트를 두는 이유는 두 가지다.
     * 첫째, 프런트엔드가 로그아웃 시점을 서버에 알려 감사 로그를 남길 수 있다.
     * 둘째, 나중에 토큰 블랙리스트나 리프레시 토큰 폐기를 도입할 때 호출 지점을 바꾸지
     * 않아도 된다. 지금은 204 No Content 만 돌려준다.
     *
     * @param principal 인증 주체(없을 수도 있다)
     * @return 204 No Content
     */
    @PostMapping("/logout")
    public ResponseEntity<Void> logout(Principal principal) {
        authService.logout(principal == null ? null : principal.getName());
        return ResponseEntity.noContent().build();
    }
}
