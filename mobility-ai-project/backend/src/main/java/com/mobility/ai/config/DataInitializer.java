package com.mobility.ai.config;

import com.mobility.ai.domain.AppUser;
import com.mobility.ai.domain.ParkingPolicy;
import com.mobility.ai.domain.Role;
import com.mobility.ai.repository.AppUserRepository;
import com.mobility.ai.repository.ParkingPolicyRepository;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.data.domain.PageRequest;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

/**
 * 기동 시 1회 실행되는 초기화 작업.
 *
 * <h2>하는 일</h2>
 * <ol>
 *   <li><b>비밀번호 해시 안전장치</b> — 계약서 2장 요구사항. 시드 SQL 의
 *       {@code password_hash} 가 비어 있거나 {@code {noop}} 로 시작하면
 *       (즉 BCrypt 가 아니면) 규약된 기본 비밀번호로 재인코딩한다. 교재 독자가
 *       해시를 잘못 붙여 넣어도 로그인이 되게 한다.</li>
 *   <li><b>시드 계정 보강</b> — 계정이 아예 없으면 3종 계정을 만든다
 *       (Flyway 를 끈 환경에서도 로그인 가능).</li>
 *   <li><b>요금 정책 보강</b> — 활성 정책이 없으면 기본 요금제를 만든다.</li>
 *   <li><b>업로드 디렉터리 생성</b> — {@code app.storage.upload-dir}</li>
 * </ol>
 */
@Component
public class DataInitializer implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(DataInitializer.class);

    /** BCrypt 해시 접두사 */
    private static final List<String> BCRYPT_PREFIXES = List.of("$2a$", "$2b$", "$2y$");

    private final AppUserRepository appUserRepository;
    private final ParkingPolicyRepository parkingPolicyRepository;
    private final PasswordEncoder passwordEncoder;
    private final AppProperties appProperties;

    /**
     * @param appUserRepository       계정 리포지토리
     * @param parkingPolicyRepository 요금 정책 리포지토리
     * @param passwordEncoder         BCrypt 인코더
     * @param appProperties           {@code app.*} 설정
     */
    public DataInitializer(AppUserRepository appUserRepository,
                           ParkingPolicyRepository parkingPolicyRepository,
                           PasswordEncoder passwordEncoder,
                           AppProperties appProperties) {
        this.appUserRepository = appUserRepository;
        this.parkingPolicyRepository = parkingPolicyRepository;
        this.passwordEncoder = passwordEncoder;
        this.appProperties = appProperties;
    }

    /**
     * 기동 시 호출된다.
     *
     * @param args 애플리케이션 인자(사용하지 않음)
     */
    @Override
    @Transactional
    public void run(ApplicationArguments args) {
        ensureUploadDirectory();
        ensureSeedUsers();
        repairPasswordHashes();
        ensureParkingPolicy();
    }

    /**
     * 업로드 디렉터리를 만든다(이미 있으면 그대로 둔다).
     */
    private void ensureUploadDirectory() {
        String dir = appProperties.getStorage().getUploadDir();
        try {
            Path path = Paths.get(dir).toAbsolutePath().normalize();
            Files.createDirectories(path);
            log.info("업로드 디렉터리 준비 완료: {}", path);
        } catch (IOException e) {
            log.warn("업로드 디렉터리 생성 실패({}): {}", dir, e.getMessage());
        }
    }

    /**
     * 시드 계정 3종이 없으면 만든다. Flyway 가 비활성인 환경에서도 로그인 가능하게 한다.
     */
    private void ensureSeedUsers() {
        createIfAbsent("admin", "admin123!", "관리자", Role.ADMIN);
        createIfAbsent("operator", "operator123!", "운영자", Role.OPERATOR);
        createIfAbsent("viewer", "viewer123!", "조회자", Role.VIEWER);
    }

    /**
     * 계정이 없을 때만 생성한다.
     *
     * @param username    아이디
     * @param rawPassword 평문 비밀번호
     * @param displayName 표시 이름
     * @param role        권한
     */
    private void createIfAbsent(String username, String rawPassword, String displayName, Role role) {
        if (appUserRepository.existsByUsername(username)) {
            return;
        }
        AppUser user = new AppUser(username, passwordEncoder.encode(rawPassword), displayName, role, true);
        appUserRepository.save(user);
        log.info("시드 계정 생성: {} ({})", username, role);
    }

    /**
     * 비밀번호 해시 안전장치. BCrypt 형태가 아닌 계정을 찾아
     * 규약된 기본 비밀번호로 다시 인코딩한다.
     *
     * <p>규약 비밀번호를 알 수 없는 계정(관리자가 추가한 계정 등)은
     * 아이디와 동일한 값 + {@code 123!} 를 쓰지 않고 건너뛰어, 의도치 않은
     * 비밀번호 변경이 생기지 않게 한다.
     */
    private void repairPasswordHashes() {
        for (AppUser user : appUserRepository.findAll()) {
            if (isBcrypt(user.getPasswordHash())) {
                continue;
            }
            String defaultPassword = defaultPasswordFor(user.getUsername());
            if (defaultPassword == null) {
                log.warn("계정 '{}' 의 password_hash 가 BCrypt 형식이 아니지만 기본 비밀번호를 알 수 없어 "
                        + "그대로 둡니다. 관리자가 직접 재설정해야 합니다.", user.getUsername());
                continue;
            }
            user.setPasswordHash(passwordEncoder.encode(defaultPassword));
            appUserRepository.save(user);
            log.warn("계정 '{}' 의 비밀번호 해시를 재인코딩했습니다(기본 비밀번호 적용).", user.getUsername());
        }
    }

    /**
     * BCrypt 해시인지 형태로 판정한다.
     *
     * @param hash 저장된 해시(null 허용)
     * @return BCrypt 형태면 true
     */
    private boolean isBcrypt(String hash) {
        if (hash == null || hash.isBlank()) {
            return false;
        }
        if (hash.startsWith("{noop}")) {
            return false;
        }
        for (String prefix : BCRYPT_PREFIXES) {
            if (hash.startsWith(prefix)) {
                // BCrypt 해시 전체 길이는 60자다. 잘려 들어온 값도 비정상으로 본다.
                return hash.length() >= 59;
            }
        }
        return false;
    }

    /**
     * 시드 계정의 규약 비밀번호를 돌려준다.
     *
     * @param username 아이디
     * @return 평문 비밀번호, 시드 계정이 아니면 null
     */
    private String defaultPasswordFor(String username) {
        if (username == null) {
            return null;
        }
        return switch (username) {
            case "admin" -> "admin123!";
            case "operator" -> "operator123!";
            case "viewer" -> "viewer123!";
            default -> null;
        };
    }

    /**
     * 활성 요금 정책이 없으면 계약서 기준 기본 요금제를 만든다.
     */
    private void ensureParkingPolicy() {
        List<ParkingPolicy> active = parkingPolicyRepository.findActive(PageRequest.of(0, 1));
        if (!active.isEmpty()) {
            return;
        }
        ParkingPolicy policy = new ParkingPolicy(
                "기본 요금제", 30, 60, 2000L, 10, 500L, 20000L, true);
        parkingPolicyRepository.save(policy);
        log.info("기본 주차 요금 정책을 생성했습니다(무료 30분 / 기본 60분 2,000원 / 10분당 500원 / 일 상한 20,000원).");
    }
}
