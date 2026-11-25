package com.mobility.ai.repository;

import com.mobility.ai.domain.ModelBenchmark;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

/**
 * 모델 벤치마크 리포지토리.
 */
public interface ModelBenchmarkRepository extends JpaRepository<ModelBenchmark, Long> {

    /**
     * 측정 시각 내림차순 전체 목록(최신 측정이 위로 온다).
     *
     * @return 벤치마크 목록
     */
    @Query("SELECT b FROM ModelBenchmark b ORDER BY b.measuredAt DESC, b.id DESC")
    List<ModelBenchmark> findAllOrderByMeasuredAtDesc();
}
