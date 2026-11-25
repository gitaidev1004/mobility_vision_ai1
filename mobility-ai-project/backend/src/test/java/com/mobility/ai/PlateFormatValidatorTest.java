package com.mobility.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.mobility.ai.service.PlateFormatValidator;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * 한국 번호판 형식 검증기 단위 테스트.
 */
@DisplayName("번호판 형식 검증기")
class PlateFormatValidatorTest {

    @ParameterizedTest(name = "유효: {0}")
    @ValueSource(strings = {
            "12가3456",      // 신형 2자리
            "123나4567",     // 신형 3자리
            "01바1234",      // 앞자리 0 허용
            "서울12가3456",  // 구형(지역 + 2자리)
            "경기7다1234"    // 구형(지역 + 1자리)
    })
    @DisplayName("신형/구형 형식을 모두 유효로 판정한다")
    void validPlates(String plate) {
        assertTrue(PlateFormatValidator.isValid(plate), "유효해야 함: " + plate);
    }

    @ParameterizedTest(name = "무효: {0}")
    @ValueSource(strings = {
            "XY-1234",       // 영문 번호판
            "1가3456",       // 앞 숫자 1자리
            "1234가5678",    // 앞 숫자 4자리
            "12가345",       // 뒤 숫자 3자리
            "12가34567",     // 뒤 숫자 5자리
            "123456",        // 한글 없음
            "가나다라",      // 숫자 없음
            "12AB3456",      // 한글 자리에 영문
            "서1가3456"      // 지역 1글자
    })
    @DisplayName("규격을 벗어난 문자열은 무효로 판정한다")
    void invalidPlates(String plate) {
        assertFalse(PlateFormatValidator.isValid(plate), "무효해야 함: " + plate);
    }

    @ParameterizedTest
    @NullAndEmptySource
    @DisplayName("null 과 빈 문자열은 무효다")
    void nullOrEmptyIsInvalid(String plate) {
        assertFalse(PlateFormatValidator.isValid(plate));
    }

    @Test
    @DisplayName("공백과 하이픈은 제거한 뒤 판정한다")
    void normalizesWhitespaceAndHyphen() {
        assertTrue(PlateFormatValidator.isValid("12 가 3456"));
        assertTrue(PlateFormatValidator.isValid("12-가-3456"));
        assertTrue(PlateFormatValidator.isValid(" 123나4567 "));
    }

    @Test
    @DisplayName("normalize 는 공백/하이픈/점을 제거한 문자열을 돌려준다")
    void normalizeStripsNoise() {
        assertEquals("12가3456", PlateFormatValidator.normalize(" 12-가 3456 "));
        assertEquals("123나4567", PlateFormatValidator.normalize("123.나.4567"));
    }

    @Test
    @DisplayName("normalize 는 null 또는 공백만 있는 입력에 null 을 돌려준다")
    void normalizeReturnsNullForBlank() {
        assertNull(PlateFormatValidator.normalize(null));
        assertNull(PlateFormatValidator.normalize("   "));
        assertNull(PlateFormatValidator.normalize("---"));
    }
}
