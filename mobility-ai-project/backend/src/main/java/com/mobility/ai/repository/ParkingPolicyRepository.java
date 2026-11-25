package com.mobility.ai.repository;

import com.mobility.ai.domain.ParkingPolicy;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

/**
 * 주차 요금 정책 리포지토리.
 */
public interface ParkingPolicyRepository extends JpaRepository<ParkingPolicy, Long> {

    /**
     * 활성 정책을 id 오름차순으로 찾는다. 서비스는 첫 번째 것을 사용한다.
     *
     * @param pageable 크기 제한용 페이지 정보
     * @return 활성 정책 목록
     */
    @Query("SELECT p FROM ParkingPolicy p WHERE p.active = true ORDER BY p.id ASC")
    List<ParkingPolicy> findActive(Pageable pageable);
}
