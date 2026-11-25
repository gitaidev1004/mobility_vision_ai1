# -*- coding: utf-8 -*-
"""번호판 비식별화 — 마스킹 · 가명처리(해시) · k-익명성 점검.

이 파일은 법률 자문이 아니다. 개인정보 보호법과 소속 기관의 내부 지침이
요구하는 조치를 **기술적으로 어떻게 구현할 수 있는가**를 보여주는 예시다.
실제 적용 범위와 보유 근거는 반드시 기관의 개인정보 담당 부서와 확인해야 한다.

구현하는 조치
    1) 문자열 마스킹 — 화면·CSV·로그에 쓸 표시용 번호판 (12가3456 → 12가**56)
    2) 이미지 마스킹 — 번호판 영역을 가우시안 블러 또는 모자이크로 가린다
    3) 가명처리 — HMAC-SHA256 + 기관 비밀키(pepper)로 되돌릴 수 없는 토큰 생성
    4) k-익명성 점검 — 통계 공개 전에 소수 집단이 식별되지 않는지 확인

사용법
    python 13_privacy/deidentify.py
    python 13_privacy/deidentify.py --image sample.jpg --bbox 412,288,596,342
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# 신형(7~8자리) / 구형(지역명 + 본문) 번호판 정규식
RE_PLATE_NEW = re.compile(r"^\d{2,3}[가-힣]\d{4}$")
RE_PLATE_OLD = re.compile(r"^[가-힣]{2}\s?\d{2}[가-힣]\d{4}$")

#: 가명처리 토큰 길이(16자 = 64비트). 짧게 자르면 충돌이 생기므로
#: 차량 수가 수십만 대 이상이면 24자 이상으로 늘린다.
TOKEN_LEN = 16


# ---------------------------------------------------------------------------
# 1) 문자열 마스킹
# ---------------------------------------------------------------------------


def normalize_plate(text: str) -> str:
    """번호판 문자열에서 공백과 하이픈을 제거한다.

    Args:
        text: 원본 문자열.

    Returns:
        정규화된 문자열(입력이 비면 빈 문자열).
    """
    return re.sub(r"[\s\-]", "", (text or "").strip())


#: 구형 번호판의 지역명 17종(닫힌 집합)
REGION_NAMES = {"서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"}


def mask_plate(text: str, keep_head: int = 3, keep_tail: int = 2) -> str:
    """표시용으로 번호판 일부를 가린다.

    가리는 자리 선택 근거:
        끝 4자리는 차량을 거의 특정하는 정보이므로 앞쪽 2자리만 남긴다.
        반대로 앞의 분류번호·용도 문자는 차종 분포 통계에 필요한 경우가 많아
        남긴다. 운영 화면에서 "같은 차가 또 왔다" 를 사람이 눈으로 구분할
        최소 정보는 유지하면서 재식별 위험은 낮추는 균형점이다.

    구형 번호판의 지역명(서울·경기 등)은 가리지 않는다. 지역명은 수십만 대가
    공유하는 정보여서 식별력이 거의 없고, 통계에는 유용하다. 대신 본문은
    신형과 동일한 규칙으로 가린다.

    Args:
        text: 번호판 문자열.
        keep_head: 본문 앞에서 남길 글자 수.
        keep_tail: 본문 뒤에서 남길 글자 수.

    Returns:
        마스킹된 문자열. 너무 짧으면 전체를 가린다.
    """
    plate = normalize_plate(text)
    if not plate:
        return ""
    prefix = ""
    if len(plate) >= 2 and plate[:2] in REGION_NAMES:
        prefix, plate = plate[:2] + " ", plate[2:]
    if len(plate) <= keep_head + keep_tail:
        return prefix + "*" * len(plate)
    middle = len(plate) - keep_head - keep_tail
    return prefix + plate[:keep_head] + "*" * middle + plate[-keep_tail:]


def mask_text_block(text: str) -> str:
    """자유 텍스트(메모·로그 본문) 안의 번호판 패턴을 찾아 마스킹한다.

    운영자가 메모란에 "12가3456 차주 민원" 처럼 적어 넣는 일이 흔하다.
    정형 컬럼만 마스킹하고 메모를 놓치면 비식별화가 무의미해진다.

    Args:
        text: 원본 텍스트.

    Returns:
        번호판이 마스킹된 텍스트.
    """
    pattern = re.compile(r"(?:[가-힣]{2}\s?)?\d{2,3}[가-힣]\d{4}")
    return pattern.sub(lambda m: mask_plate(m.group(0)), text or "")


# ---------------------------------------------------------------------------
# 2) 가명처리 (HMAC)
# ---------------------------------------------------------------------------


def get_pepper() -> bytes:
    """가명처리용 비밀키를 환경변수에서 읽는다.

    왜 소금(salt)이 아니라 후추(pepper)인가?
        번호판은 경우의 수가 1억 안쪽이다. 단순 SHA-256 해시는 공격자가
        전수 계산(rainbow table)으로 1시간 안에 역추적할 수 있다. 레코드별
        salt 를 쓰면 같은 차량이 다른 토큰이 되어 통계가 불가능해지므로,
        **전역 비밀키(pepper)를 섞은 HMAC** 을 쓴다. 키는 DB 와 분리된
        비밀 관리소(환경변수·Vault)에 두고, 키가 유출되면 재식별이 가능해지므로
        키 교체 절차를 함께 설계한다.

    Returns:
        비밀키 바이트열. 환경변수가 없으면 실습용 기본값(경고 출력).
    """
    raw = os.environ.get("PLATE_PEPPER", "")
    if not raw:
        print("[경고] PLATE_PEPPER 환경변수가 없다. 실습용 기본키를 사용한다.")
        print("       운영에서는 32바이트 이상 난수를 비밀 관리소에 둔다.")
        raw = "mobility-ai-demo-pepper-do-not-use-in-production"
    return raw.encode("utf-8")


def pseudonymize(text: str, pepper: bytes | None = None,
                 length: int = TOKEN_LEN) -> str:
    """번호판을 되돌릴 수 없는 가명 토큰으로 바꾼다.

    같은 번호판은 항상 같은 토큰이 되므로, 토큰만으로도 "같은 차량의 반복
    방문" 같은 통계를 낼 수 있다. 반면 토큰에서 번호판을 복원할 수는 없다.

    Args:
        text: 번호판 문자열.
        pepper: 비밀키(None 이면 환경변수에서 읽는다).
        length: 토큰 길이(16진수 문자 수).

    Returns:
        ``PV-`` 접두사가 붙은 토큰(입력이 비면 빈 문자열).
    """
    plate = normalize_plate(text)
    if not plate:
        return ""
    key = pepper if pepper is not None else get_pepper()
    digest = hmac.new(key, plate.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"PV-{digest[:length]}"


def verify_pseudonym(text: str, token: str,
                     pepper: bytes | None = None) -> bool:
    """번호판과 토큰이 대응하는지 확인한다(단방향 검증).

    운영자가 "이 토큰이 12가3456 맞습니까?" 를 확인할 때 쓴다. 토큰 목록
    전체를 복원하는 것은 불가능하지만, 번호판을 이미 알고 있는 1건의
    대조는 가능하다. 이 검증 자체도 감사 로그에 남겨야 한다.

    Args:
        text: 확인하려는 번호판.
        token: 저장된 토큰.
        pepper: 비밀키.

    Returns:
        일치 여부.
    """
    return hmac.compare_digest(pseudonymize(text, pepper), token or "")


# ---------------------------------------------------------------------------
# 3) 이미지 마스킹
# ---------------------------------------------------------------------------


def mask_image_region(src: Path, dst: Path, bbox: tuple[int, int, int, int],
                      method: str = "blur") -> bool:
    """이미지의 번호판 영역을 가린다.

    방법 선택 근거:
        - ``blur``(가우시안): 보기 좋지만 커널이 작으면 복원 공격이 가능하다.
          커널 크기를 영역 폭의 1/4 이상으로 크게 잡아야 한다.
        - ``mosaic``(픽셀화): 다운샘플 후 확대. 블록 크기가 크면 정보가
          물리적으로 사라져 복원이 어렵다. 증빙 보관용으로는 이쪽이 안전하다.
        - ``fill``(단색): 완전히 지운다. 재판독이 영구히 불가능하므로
          원본을 별도 보관하는 경우에만 쓴다.

    Args:
        src: 원본 이미지 경로.
        dst: 저장할 경로.
        bbox: (x1, y1, x2, y2) 번호판 영역.
        method: ``blur`` | ``mosaic`` | ``fill``.

    Returns:
        성공 여부.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  [건너뜀] opencv-python 과 numpy 가 필요하다.")
        return False
    img = cv2.imread(str(src))
    if img is None:
        print(f"  [건너뜀] 이미지를 읽을 수 없다: {src}")
        return False
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (max(0, min(w, bbox[0])), max(0, min(h, bbox[1])),
                      max(0, min(w, bbox[2])), max(0, min(h, bbox[3])))
    if x2 <= x1 or y2 <= y1:
        print("  [건너뜀] bbox 가 유효하지 않다.")
        return False
    roi = img[y1:y2, x1:x2]
    if method == "mosaic":
        # 블록 크기: 영역 폭의 1/8. 번호판 문자 폭보다 크게 잡는다.
        small = cv2.resize(roi, (max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)),
                           interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (x2 - x1, y2 - y1),
                         interpolation=cv2.INTER_NEAREST)
    elif method == "fill":
        roi = np.zeros_like(roi)
    else:
        k = max(9, ((x2 - x1) // 4) | 1)   # 홀수 커널
        roi = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = roi
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(cv2.imwrite(str(dst), img))
    print(f"  {'저장' if ok else '실패'}: {dst} (방식 {method}, "
          f"영역 {x2 - x1}x{y2 - y1}px)")
    return ok


# ---------------------------------------------------------------------------
# 4) k-익명성 점검
# ---------------------------------------------------------------------------


@dataclass
class KAnonReport:
    """k-익명성 점검 결과."""

    k: int
    groups: int
    violating: list[tuple[tuple[str, ...], int]]

    @property
    def ok(self) -> bool:
        """모든 그룹이 k 이상인지."""
        return not self.violating


def check_k_anonymity(rows: list[dict[str, str]], quasi_ids: list[str],
                      k: int = 5) -> KAnonReport:
    """준식별자 조합별 건수를 세어 k-익명성을 점검한다.

    왜 번호판을 지워도 재식별될 수 있는가?
        "2026-10-24 09시 · 정문 · 전기 SUV · 장애인 구역" 같은 조합은
        번호판이 없어도 단 한 대만 해당할 수 있다. 통계를 외부에 공개하기
        전에 이런 소수 집단을 찾아내 집계 단위를 넓히거나(시 → 일),
        해당 행을 제외해야 한다.

    Args:
        rows: 레코드 목록.
        quasi_ids: 준식별자로 볼 컬럼 이름 목록.
        k: 각 조합이 가져야 할 최소 건수.

    Returns:
        KAnonReport.
    """
    counter: Counter[tuple[str, ...]] = Counter(
        tuple(str(r.get(q, "")) for q in quasi_ids) for r in rows)
    violating = [(key, n) for key, n in counter.items() if n < k]
    violating.sort(key=lambda kv: kv[1])
    return KAnonReport(k=k, groups=len(counter), violating=violating)


def generalize(rows: list[dict[str, str]], column: str,
              mapping: dict[str, str]) -> list[dict[str, str]]:
    """준식별자 값을 더 넓은 범주로 일반화한다.

    예: 시각 "09:14" → "09시", 차종 "전기 SUV" → "승용".
    k-익명성 위반이 나오면 가장 세밀한 컬럼부터 일반화해 다시 점검한다.

    Args:
        rows: 레코드 목록.
        column: 일반화할 컬럼.
        mapping: 원래 값 → 일반화된 값.

    Returns:
        새 레코드 목록(원본은 바꾸지 않는다).
    """
    out = []
    for r in rows:
        copy = dict(r)
        if column in copy:
            copy[column] = mapping.get(copy[column], copy[column])
        out.append(copy)
    return out


# ---------------------------------------------------------------------------
# 데모
# ---------------------------------------------------------------------------


def demo() -> None:
    """네 가지 조치를 차례로 보여준다."""
    print("=" * 70)
    print("번호판 비식별화 조치 데모")
    print("=" * 70)

    plates = ["12가3456", "서울 34나5678", "87로2345", "11바3456", "45다7890"]

    print("\n[1] 표시용 마스킹")
    print(f"  {'원본':<14}{'정규화':<13}{'마스킹':<13}{'형식검증':>8}")
    for p in plates:
        norm = normalize_plate(p)
        valid = bool(RE_PLATE_NEW.match(norm) or RE_PLATE_OLD.match(norm))
        print(f"  {p:<14}{norm:<13}{mask_plate(p):<13}{'유효' if valid else '무효':>8}")

    print("\n[2] 자유 텍스트 안의 번호판 마스킹")
    memo = "민원: 12가3456 차량이 장애인 구역에 주차. 담당자 확인 후 서울 34나5678 로 안내."
    print(f"  원본  : {memo}")
    print(f"  처리후: {mask_text_block(memo)}")

    print("\n[3] 가명처리 (HMAC-SHA256 + pepper)")
    pepper = b"demo-pepper-for-textbook-example-only"
    print(f"  {'번호판':<14}{'토큰':<24}{'재현성':>8}")
    for p in plates:
        t1 = pseudonymize(p, pepper)
        t2 = pseudonymize(p, pepper)
        print(f"  {p:<14}{t1:<24}{'동일' if t1 == t2 else '불일치':>8}")
    print(f"  키를 바꾸면 토큰도 바뀐다: "
          f"{pseudonymize('12가3456', b'another-key')}")
    print(f"  단방향 검증: 12가3456 ↔ 토큰 "
          f"{verify_pseudonym('12가3456', pseudonymize('12가3456', pepper), pepper)}")
    print(f"             87로2345 ↔ 위 토큰 "
          f"{verify_pseudonym('87로2345', pseudonymize('12가3456', pepper), pepper)}")

    print("\n[4] k-익명성 점검 (k=5)")
    rows = [
        {"hour": "09:14", "gate": "정문", "type": "전기SUV", "zone": "장애인"},
        {"hour": "09:22", "gate": "정문", "type": "승용",     "zone": "일반"},
        {"hour": "09:31", "gate": "정문", "type": "승용",     "zone": "일반"},
        {"hour": "09:40", "gate": "정문", "type": "승용",     "zone": "일반"},
        {"hour": "09:48", "gate": "정문", "type": "승용",     "zone": "일반"},
        {"hour": "09:55", "gate": "정문", "type": "승용",     "zone": "일반"},
        {"hour": "10:03", "gate": "후문", "type": "화물",     "zone": "하역"},
        {"hour": "10:11", "gate": "후문", "type": "화물",     "zone": "하역"},
    ]
    quasi = ["hour", "gate", "type", "zone"]
    rep = check_k_anonymity(rows, quasi, k=5)
    print(f"  준식별자 {quasi}")
    print(f"  그룹 수 {rep.groups} / 위반 {len(rep.violating)}건 "
          f"→ {'통과' if rep.ok else '공개 불가'}")
    for key, n in rep.violating[:4]:
        print(f"    {key} → {n}건")

    print("\n  1차 일반화 — 시각을 '시' 단위로, 차종·구역을 대분류로")
    hour_map = {r["hour"]: r["hour"].split(":")[0] + "시" for r in rows}
    coarse = generalize(rows, "hour", hour_map)
    coarse = generalize(coarse, "type",
                        {"전기SUV": "승용", "승용": "승용", "화물": "화물"})
    coarse = generalize(coarse, "zone",
                        {"장애인": "일반", "일반": "일반", "하역": "하역"})
    rep2 = check_k_anonymity(coarse, quasi, k=5)
    print(f"  그룹 수 {rep2.groups} / 위반 {len(rep2.violating)}건 "
          f"→ {'통과' if rep2.ok else '공개 불가'}")
    for key, n in rep2.violating[:4]:
        print(f"    {key} → {n}건  ← 후문 화물 2건이 여전히 k 미달")

    print("\n  2차 일반화 — 게이트까지 '전체' 로 묶는다")
    coarser = generalize(coarse, "gate", {"정문": "전체", "후문": "전체"})
    coarser = generalize(coarser, "hour", {"09시": "오전", "10시": "오전"})
    coarser = generalize(coarser, "type", {"승용": "전체", "화물": "전체"})
    coarser = generalize(coarser, "zone", {"일반": "전체", "하역": "전체"})
    rep3 = check_k_anonymity(coarser, quasi, k=5)
    print(f"  그룹 수 {rep3.groups} / 위반 {len(rep3.violating)}건 "
          f"→ {'통과' if rep3.ok else '공개 불가'}")
    print("  해석: 집계 단위를 넓히면 익명성은 올라가지만 분석 가치는 떨어진다.")
    print("        2차처럼 전부 '전체' 가 되면 통계 의미가 사라지므로, 실무에서는")
    print("        k 미달 그룹만 '기타' 로 합치거나 그 행을 공개 대상에서 뺀다.")

    print("\n[주의]")
    print("  이 코드는 기술적 조치의 예시다. 번호판 데이터의 처리 근거·보유기간·")
    print("  제3자 제공 가능 범위는 개인정보 보호법과 기관 내부 지침에 따라 정하고,")
    print("  개인정보 담당 부서의 검토를 받아야 한다.")


def main(argv: list[str] | None = None) -> int:
    """명령행 진입점."""
    ap = argparse.ArgumentParser(description="번호판 비식별화 도구")
    ap.add_argument("--image", type=Path, default=None, help="마스킹할 이미지")
    ap.add_argument("--bbox", default="", help="x1,y1,x2,y2")
    ap.add_argument("--method", default="mosaic",
                    choices=["blur", "mosaic", "fill"])
    ap.add_argument("--out", type=Path, default=Path("masked.jpg"))
    args = ap.parse_args(argv)

    demo()

    if args.image and args.bbox:
        print("\n[5] 이미지 영역 마스킹")
        parts = [int(x) for x in args.bbox.split(",")]
        if len(parts) == 4:
            mask_image_region(args.image, args.out,
                              (parts[0], parts[1], parts[2], parts[3]),
                              args.method)
        else:
            print("  [건너뜀] --bbox 는 x1,y1,x2,y2 형식이어야 한다.")
    else:
        print("\n[5] 이미지 영역 마스킹 — "
              "--image 와 --bbox 를 주면 실제 파일에 적용한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
