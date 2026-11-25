# -*- coding: utf-8 -*-
"""모빌리티 AI 프로젝트 개발 환경 검증 스크립트.

파이썬 패키지는 import 로, 외부 도구는 --version 명령으로 확인한다.
설치되지 않은 항목이 있어도 중단하지 않고 전체 결과를 표로 출력한다.
실행:  python 01_environment/check_env.py
"""
from __future__ import annotations

import importlib
import platform
import re
import shutil
import subprocess
import sys

# (표시 이름, import 이름, 기대 버전 접두사) 목록
PY_PACKAGES = [
    ("NumPy", "numpy", "2."),
    ("OpenCV", "cv2", "4."),
    ("Pillow", "PIL", "11."),
    ("PyTorch", "torch", "2.8"),
    ("TorchVision", "torchvision", "0.23"),
    ("Ultralytics", "ultralytics", "8.3"),
    ("ONNX Runtime", "onnxruntime", "1."),
    ("scikit-learn", "sklearn", "1."),
    ("pandas", "pandas", "2."),
    ("FastAPI", "fastapi", "0.1"),
]

# (표시 이름, 실행 파일, 인자, 기대 버전 접두사) 목록
CLI_TOOLS = [
    ("PostgreSQL", "psql", ["--version"], "16"),
    ("JDK", "java", ["-version"], "21"),
    ("Node", "node", ["--version"], "v20"),
    ("npm", "npm", ["--version"], "10"),
]

VERSION_RE = re.compile(r"(\d+(?:\.\d+)+)")


def pkg_version(mod_name: str) -> str | None:
    """패키지를 import 해 버전 문자열을 돌려준다. 실패하면 None."""
    try:
        mod = importlib.import_module(mod_name)
    except Exception:                      # 미설치, 의존성 오류 모두 포함
        return None
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
        if isinstance(v, tuple):
            return ".".join(str(x) for x in v)
    return "(버전 정보 없음)"


def cli_version(exe: str, args: list[str]) -> str | None:
    """외부 명령을 실행해 첫 번째 버전 숫자를 추출한다. 실패하면 None."""
    if shutil.which(exe) is None:
        return None
    try:
        # java -version 은 stderr 로 출력하므로 두 스트림을 합친다.
        out = subprocess.run([exe, *args], capture_output=True, text=True,
                             timeout=30, check=False)
    except Exception:
        return None
    text = (out.stdout or "") + "\n" + (out.stderr or "")
    # 도구에 따라 환경 변수 안내 등 잡음이 먼저 출력되므로 모든 줄을 훑는다.
    for line in text.splitlines():
        line = line.strip()
        if not line or "OPTIONS" in line.upper():
            continue
        m = VERSION_RE.search(line)
        if m:
            # node 는 v20.18.1 형태이므로 접두사 v 를 살려 둔다.
            return ("v" + m.group(1)) if line.startswith("v") else m.group(1)
    return "(버전 확인 불가)"


def judge(actual: str | None, expect: str) -> str:
    """기대 버전 접두사와 비교해 판정 문자열을 돌려준다."""
    if actual is None:
        return "미설치"
    return "OK" if actual.startswith(expect) else f"버전확인({expect} 기대)"


def print_row(name: str, actual: str, verdict: str) -> None:
    """고정 폭으로 한 행을 출력한다."""
    print(f"  {name:<14s} {actual:<22s} {verdict}")


def main() -> int:
    print("=" * 56)
    print(" 모빌리티 AI 개발 환경 검증")
    print("=" * 56)
    print(f"  플랫폼        {platform.system()} {platform.machine()}")
    print(f"  Python        {sys.version.split()[0]}")
    expected_py = "3.12"
    py_ok = sys.version.startswith(expected_py)
    print(f"  Python 판정   {'OK' if py_ok else f'{expected_py} 권장'}")

    missing: list[str] = []

    print("-" * 56)
    print(" 파이썬 패키지")
    for name, mod, expect in PY_PACKAGES:
        v = pkg_version(mod)
        verdict = judge(v, expect)
        print_row(name, v or "-", verdict)
        if verdict == "미설치":
            missing.append(name)

    print("-" * 56)
    print(" 외부 도구")
    for name, exe, args, expect in CLI_TOOLS:
        v = cli_version(exe, args)
        verdict = judge(v, expect)
        print_row(name, v or "-", verdict)
        if verdict == "미설치":
            missing.append(name)

    # PyTorch 가 있으면 CPU 설정까지 함께 확인한다.
    print("-" * 56)
    print(" CPU 추론 설정")
    try:
        import torch
        print_row("CUDA 사용", str(torch.cuda.is_available()),
                  "OK(CPU 전용)" if not torch.cuda.is_available() else "GPU 감지")
        print_row("스레드 수", str(torch.get_num_threads()), "1.5절에서 조정")
        print_row("빌드 태그", torch.__version__,
                  "OK" if "+cpu" in torch.__version__ else "CPU 휠 권장")
    except Exception:
        print_row("PyTorch", "-", "미설치로 생략")

    print("=" * 56)
    if missing:
        print(f" 미설치 {len(missing)}건: {', '.join(missing)}")
        print(" 1.4절의 해당 소절을 참고해 설치한다.")
        return 1
    print(" 모든 구성 요소가 확인되었다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
