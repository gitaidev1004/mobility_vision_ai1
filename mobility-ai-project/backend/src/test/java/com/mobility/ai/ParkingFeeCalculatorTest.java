package com.mobility.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.mobility.ai.domain.ParkingPolicy;
import com.mobility.ai.domain.PassType;
import com.mobility.ai.service.ParkingFeeCalculator;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * 주차 요금 계산 엔진 단위 테스트 — 스프링 컨텍스트 없이 순수 로직만 검증한다.
 *
 * <p>기준 정책: 무료 30분 / 기본 60분 2,000원 / 추가 10분당 500원 / 일 상한 20,000원
 */
@DisplayName("주차 요금 계산 엔진")
class ParkingFeeCalculatorTest {

    /**
     * 테스트용 기본 정책을 만든다.
     *
     * @return 무료 30분 / 기본 60분 2,000원 / 10분당 500원 / 일 상한 20,000원
     */
    private ParkingPolicy defaultPolicy() {
        return new ParkingPolicy("테스트 요금제", 30, 60, 2000L, 10, 500L, 20000L, true);
    }

    @Nested
    @DisplayName("무료 시간 경계값")
    class FreeMinutesBoundary {

        @Test
        @DisplayName("0분 주차는 무료다")
        void zeroMinutesIsFree() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 0, PassType.NONE, false, false);

            assertEquals(0L, result.payableFee());
            assertEquals("FREE_MINUTES", result.freeReason());
        }

        @Test
        @DisplayName("무료 시간과 정확히 같은 30분은 무료다 (경계 포함)")
        void exactlyFreeMinutesIsFree() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 30, PassType.NONE, false, false);

            assertEquals(0L, result.payableFee());
            assertEquals(0L, result.grossFee());
            assertEquals("FREE_MINUTES", result.freeReason());
        }

        @Test
        @DisplayName("무료 시간을 1분 넘기면 기본 요금이 부과된다")
        void oneMinuteOverFreeChargesBaseFee() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 31, PassType.NONE, false, false);

            assertEquals(2000L, result.payableFee());
            assertNull(result.freeReason());
        }
    }

    @Nested
    @DisplayName("기본 시간 구간")
    class BaseMinutes {

        @Test
        @DisplayName("기본 시간 미만(45분)은 기본 요금만 부과된다")
        void underBaseMinutesChargesBaseFeeOnly() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 45, PassType.NONE, false, false);

            assertEquals(2000L, result.payableFee());
        }

        @Test
        @DisplayName("기본 시간과 정확히 같은 60분은 기본 요금만 부과된다 (경계 포함)")
        void exactlyBaseMinutesChargesBaseFeeOnly() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 60, PassType.NONE, false, false);

            assertEquals(2000L, result.payableFee());
        }

        @Test
        @DisplayName("기본 시간을 1분 넘기면 단위 요금 1단위가 올림 가산된다")
        void oneMinuteOverBaseAddsOneUnit() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 61, PassType.NONE, false, false);

            // 2000 + 1단위(10분) * 500 = 2500
            assertEquals(2500L, result.payableFee());
        }
    }

    @Nested
    @DisplayName("단위 요금 올림 처리")
    class UnitRounding {

        @ParameterizedTest(name = "{0}분 → {1}원")
        @CsvSource({
                "61, 2500",     // 초과 1분  → 1단위
                "70, 2500",     // 초과 10분 → 1단위 (경계)
                "71, 3000",     // 초과 11분 → 2단위
                "90, 3500",     // 초과 30분 → 3단위
                "120, 5000",    // 초과 60분 → 6단위
                "180, 8000"     // 초과 120분 → 12단위
        })
        @DisplayName("초과분은 단위 시간으로 올림하여 가산한다")
        void roundsUpToUnit(long minutes, long expectedFee) {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), minutes, PassType.NONE, false, false);

            assertEquals(expectedFee, result.payableFee());
        }
    }

    @Nested
    @DisplayName("일일 상한")
    class DailyCap {

        @Test
        @DisplayName("상한에 도달하지 않는 구간은 계산값 그대로다")
        void underCapUsesComputedFee() {
            // 300분: 2000 + (240/10)*500 = 2000 + 12000 = 14000 (< 상한 20000)
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 300, PassType.NONE, false, false);

            assertEquals(14000L, result.payableFee());
        }

        @Test
        @DisplayName("상한을 처음 넘기는 구간은 상한값이 된다")
        void justOverCapIsClamped() {
            // 600분: 2000 + (540/10)*500 = 29000 → 1일 상한 20000
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 600, PassType.NONE, false, false);

            assertEquals(20000L, result.payableFee());
        }

        @Test
        @DisplayName("일일 상한을 초과하는 요금은 상한으로 잘린다")
        void overCapIsClamped() {
            // 1440분(24시간): 2000 + (1380/10)*500 = 71000 → 1일 상한 20000
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 1440, PassType.NONE, false, false);

            assertEquals(20000L, result.payableFee());
        }

        @Test
        @DisplayName("24시간을 넘기면 하루치 상한이 추가된다")
        void multipleDaysMultiplyCap() {
            // 1800분(30시간) → 2일분 상한 = 40000
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 1800, PassType.NONE, false, false);

            assertEquals(40000L, result.payableFee());
        }

        @Test
        @DisplayName("정확히 48시간은 2일분 상한이다 (경계 포함)")
        void exactlyTwoDaysUsesTwoDayCap() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 2880, PassType.NONE, false, false);

            assertEquals(40000L, result.payableFee());
        }
    }

    @Nested
    @DisplayName("이용권 처리")
    class PassHandling {

        @Test
        @DisplayName("유효한 정기권(REGULAR)은 전액 무료다")
        void validRegularPassIsFree() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 600, PassType.REGULAR, true, false);

            assertEquals(0L, result.payableFee());
            assertEquals("FREE_PASS", result.freeReason());
        }

        @Test
        @DisplayName("유효한 임직원권(STAFF)은 전액 무료다")
        void validStaffPassIsFree() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 600, PassType.STAFF, true, false);

            assertEquals(0L, result.payableFee());
            assertEquals("FREE_PASS", result.freeReason());
        }

        @Test
        @DisplayName("만료된 정기권은 일반 차량과 같이 과금된다")
        void expiredRegularPassIsCharged() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 90, PassType.REGULAR, false, false);

            assertEquals(3500L, result.payableFee());
            assertNull(result.freeReason());
        }

        @Test
        @DisplayName("유효한 방문권은 20% 할인된다")
        void validVisitorPassGetsDiscount() {
            // 90분 → 3500원, 20% 할인 700원 → 2800원
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 90, PassType.VISITOR, false, true);

            assertEquals(3500L, result.grossFee());
            assertEquals(700L, result.discount());
            assertEquals(2800L, result.payableFee());
        }

        @Test
        @DisplayName("만료된 방문권은 할인되지 않는다")
        void expiredVisitorPassGetsNoDiscount() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 90, PassType.VISITOR, false, false);

            assertEquals(0L, result.discount());
            assertEquals(3500L, result.payableFee());
        }

        @Test
        @DisplayName("방문권 할인은 일일 상한 적용 후의 금액에 대해 계산된다")
        void visitorDiscountAppliesAfterCap() {
            // 1440분 → 상한 20000 → 20% 할인 4000 → 16000
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 1440, PassType.VISITOR, false, true);

            assertEquals(20000L, result.grossFee());
            assertEquals(4000L, result.discount());
            assertEquals(16000L, result.payableFee());
        }

        @Test
        @DisplayName("무료 시간 이내라면 이용권 종류와 무관하게 0원이다")
        void withinFreeMinutesAlwaysZero() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 20, PassType.VISITOR, false, true);

            assertEquals(0L, result.payableFee());
            assertEquals("FREE_MINUTES", result.freeReason());
        }
    }

    @Nested
    @DisplayName("방어 로직")
    class Defensive {

        @Test
        @DisplayName("정책이 없으면 0원으로 처리한다")
        void nullPolicyIsFree() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    null, 600, PassType.NONE, false, false);

            assertEquals(0L, result.payableFee());
            assertEquals("NO_POLICY", result.freeReason());
        }

        @Test
        @DisplayName("음수 주차 시간은 0분으로 본다")
        void negativeMinutesTreatedAsZero() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), -10, PassType.NONE, false, false);

            assertEquals(0L, result.payableFee());
        }

        @Test
        @DisplayName("passType 이 null 이어도 일반 차량으로 계산된다")
        void nullPassTypeTreatedAsNone() {
            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    defaultPolicy(), 90, null, false, false);

            assertEquals(3500L, result.payableFee());
            assertNotNull(result);
        }

        @Test
        @DisplayName("단위 시간이 0인 잘못된 정책은 기본 요금만 부과한다")
        void zeroUnitMinutesChargesBaseFeeOnly() {
            ParkingPolicy broken =
                    new ParkingPolicy("잘못된 정책", 30, 60, 2000L, 0, 500L, 20000L, true);

            ParkingFeeCalculator.FeeResult result = ParkingFeeCalculator.calculate(
                    broken, 600, PassType.NONE, false, false);

            assertEquals(2000L, result.payableFee());
        }
    }
}
