package com.mobility.ai.service;

import java.util.regex.Pattern;

/**
 * 한국 자동차 번호판 형식 검증기 — 순수 정적 메서드만 두어 단위 테스트가 쉽다
 * (교재 6장 "인식 결과 후처리" 절에서 인용).
 *
 * <h2>허용 형식</h2>
 * <ul>
 *   <li><b>신형(2006년 이후)</b> : 숫자 2~3 + 한글 1 + 숫자 4 — 예 {@code 12가3456}, {@code 123나4567}</li>
 *   <li><b>구형(지역명 포함)</b> : 지역 2글자 + 숫자 1~2 + 한글 1 + 숫자 4 — 예 {@code 서울12가3456}</li>
 *   <li><b>영업용/택배 등</b> : 위 형식의 한글 자리에 영업용 문자가 오는 경우도 동일 패턴으로 수용</li>
 * </ul>
 *
 * <p>공백과 하이픈은 비교 전에 제거하므로 {@code "12 가 3456"}, {@code "12-가-3456"} 도
 * 유효하게 본다. 알파벳만으로 된 외국 번호판({@code XY-1234})은 무효다.
 */
public final class PlateFormatValidator {

    /** 신형: 12가3456 / 123나4567 */
    private static final Pattern MODERN =
            Pattern.compile("^[0-9]{2,3}[가-힣][0-9]{4}$");

    /** 구형: 서울12가3456 (지역 2글자 + 숫자 1~2 + 한글 + 숫자 4) */
    private static final Pattern LEGACY =
            Pattern.compile("^[가-힣]{2}[0-9]{1,2}[가-힣][0-9]{4}$");

    /** 외교/임시 등 영문 혼용을 배제하기 위한 제거 대상(공백, 하이픈, 점) */
    private static final Pattern NOISE = Pattern.compile("[\\s\\-.]");

    private PlateFormatValidator() {
    }

    /**
     * 번호판 문자열을 정규화한다(공백/하이픈/점 제거).
     *
     * @param raw 원본 문자열(null 허용)
     * @return 정규화된 문자열, 입력이 null 이면 null
     */
    public static String normalize(String raw) {
        if (raw == null) {
            return null;
        }
        String cleaned = NOISE.matcher(raw).replaceAll("");
        return cleaned.isEmpty() ? null : cleaned;
    }

    /**
     * 한국 번호판 형식인지 판정한다.
     *
     * @param raw 번호판 문자열(null/공백 허용)
     * @return 신형 또는 구형 형식에 맞으면 true
     */
    public static boolean isValid(String raw) {
        String plate = normalize(raw);
        if (plate == null) {
            return false;
        }
        return MODERN.matcher(plate).matches() || LEGACY.matcher(plate).matches();
    }
}
