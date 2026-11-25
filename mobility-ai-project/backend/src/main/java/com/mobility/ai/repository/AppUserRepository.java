package com.mobility.ai.repository;

import com.mobility.ai.domain.AppUser;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

/**
 * 계정 리포지토리.
 */
public interface AppUserRepository extends JpaRepository<AppUser, Long> {

    /**
     * 로그인 아이디로 계정을 찾는다.
     *
     * @param username 로그인 아이디
     * @return 계정(없으면 empty)
     */
    Optional<AppUser> findByUsername(String username);

    /**
     * 아이디 존재 여부.
     *
     * @param username 로그인 아이디
     * @return 있으면 true
     */
    boolean existsByUsername(String username);
}
