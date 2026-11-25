package com.mobility.ai.domain;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.LocalDateTime;

/**
 * 로그인 계정 엔티티. 테이블 {@code app_user}.
 * Lombok 을 쓰지 않고 명시적 생성자/게터/세터로 작성한다(교재 가독성).
 */
@Entity
@Table(name = "app_user")
public class AppUser {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    /** 로그인 아이디(유니크) */
    @Column(name = "username", nullable = false, unique = true, length = 50)
    private String username;

    /** BCrypt 해시. 시드 SQL 의 값이 비정상이면 DataInitializer 가 재인코딩한다. */
    @Column(name = "password_hash", nullable = false, length = 200)
    private String passwordHash;

    /** 화면에 표시할 이름 */
    @Column(name = "display_name", nullable = false, length = 100)
    private String displayName;

    /** ADMIN|OPERATOR|VIEWER */
    @Enumerated(EnumType.STRING)
    @Column(name = "role", nullable = false, length = 20)
    private Role role;

    @Column(name = "enabled", nullable = false)
    private boolean enabled = true;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    /** JPA 가 요구하는 기본 생성자 */
    protected AppUser() {
    }

    public AppUser(String username, String passwordHash, String displayName, Role role, boolean enabled) {
        this.username = username;
        this.passwordHash = passwordHash;
        this.displayName = displayName;
        this.role = role;
        this.enabled = enabled;
    }

    /** INSERT 직전에 생성시각을 채운다. */
    @PrePersist
    void onCreate() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
        }
    }

    public Long getId() {
        return id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPasswordHash() {
        return passwordHash;
    }

    public void setPasswordHash(String passwordHash) {
        this.passwordHash = passwordHash;
    }

    public String getDisplayName() {
        return displayName;
    }

    public void setDisplayName(String displayName) {
        this.displayName = displayName;
    }

    public Role getRole() {
        return role;
    }

    public void setRole(Role role) {
        this.role = role;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
}
