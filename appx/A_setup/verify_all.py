# -*- coding: utf-8 -*-
"""부록 A — 개발 환경 종합 점검 스크립트.

파이썬 버전, 가상환경, 필수 패키지, 외부 명령(java/node/npm/psql),
PostgreSQL 접속, 서비스 포트(8000/8080/5173), 헬스체크 응답을
한 번에 점검해 표로 출력한다.

설치되지 않은 항목은 FAIL 로 표시하고, 스크립트 자체는 항상 정상 종료한다.
외부 네트워크나 추가 패키지가 없어도 동작한다(표준 라이브러리만 사용).

사용법:
    (mobility) d:\\mobility_ai> python appx/A_setup/verify_all.py
    (mobility) ~/mobility_ai$  python appx/A_setup/verify_all.py --json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------- 점검 결과 수집

# 결과 행: (구분, 점검 항목, 기대값, 실제값, 판정)
ROWS: list[tuple[str, str, str, str, str]] = []

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def add(group: str, name: str, expect: str, actual: str, verdict: str) -> None:
    """점검 결과 한 줄을 추가한다. 긴 실제값은 44자로 잘라 표 폭을 유지한다."""
    if len(actual) > 44:
        actual = actual[:41] + "..."
    ROWS.append((group, name, expect, actual, verdict))


def run_cmd(args: list[str], timeout: float = 20.0) -> tuple[int, str]:
    """외부 명령을 실행해 (종료코드, 출력 첫 줄들)을 돌려준다.

    명령이 없거나 시간이 초과되면 종료코드 -1 과 사유를 돌려준다.
    """
    exe = shutil.which(args[0])
    if exe is None:
        return -1, "명령을 찾을 수 없음"
    try:
        proc = subprocess.run(                      # noqa: S603
            [exe, *args[1:]],
            capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return -1, "시간 초과"
    except OSError as exc:
        return -1, f"실행 실패: {exc}"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def first_line(text: str) -> str:
    """여러 줄 출력에서 의미 있는 첫 줄만 뽑는다.

    JVM 이 출력하는 `Picked up JAVA_TOOL_OPTIONS` 같은 잡음 줄은 건너뛴다.
    """
    noise = ("Picked up ", "WARNING", "Warning:", "NOTE:")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(noise):
            continue
        return line[:44]
    return "(출력 없음)"


# ---------------------------------------------------------------- 1) 파이썬

def check_python() -> None:
    """파이썬 버전과 가상환경 활성화 여부를 점검한다."""
    ver = ".".join(str(v) for v in sys.version_info[:3])
    major_minor = sys.version_info[:2]
    verdict = PASS if major_minor == (3, 12) else WARN
    add("파이썬", "python 버전", "3.12.x", ver, verdict)

    # sys.prefix 와 sys.base_prefix 가 다르면 가상환경이 활성화된 상태다.
    in_venv = sys.prefix != sys.base_prefix
    env_name = os.path.basename(sys.prefix) if in_venv else "(시스템 전역)"
    add("파이썬", "가상환경 활성화", "mobility",
        env_name, PASS if in_venv else FAIL)

    add("파이썬", "플랫폼", "Windows/Linux/macOS",
        f"{platform.system()} {platform.machine()}", PASS)
    try:
        cores = os.cpu_count() or 0
    except Exception:                               # noqa: BLE001
        cores = 0
    add("파이썬", "CPU 코어 수", "2 이상",
        str(cores), PASS if cores >= 2 else WARN)


# ---------------------------------------------------------------- 2) 패키지

# (import 이름, 표시 이름, 기대 버전, 필수 여부)
PACKAGES = [
    ("numpy",       "numpy",        "2.1.x",     True),
    ("cv2",         "opencv",       "4.12.x",    True),
    ("PIL",         "pillow",       "11.x",      True),
    ("torch",       "torch",        "2.8.0",     True),
    ("torchvision", "torchvision",  "0.23.0",    True),
    ("ultralytics", "ultralytics",  "8.3.x",     True),
    ("onnxruntime", "onnxruntime",  "1.22.x",    True),
    ("sklearn",     "scikit-learn", "1.7.x",     True),
    ("pandas",      "pandas",       "2.3.x",     True),
    ("fastapi",     "fastapi",      "0.118.x",   True),
    ("uvicorn",     "uvicorn",      "0.37.x",    True),
    ("pydantic",    "pydantic",     "2.11.x",    True),
    ("psycopg2",    "psycopg2",     "2.9.x",     False),
    ("pytest",      "pytest",       "8.4.x",     False),
]


def module_version(mod_name: str) -> str:
    """모듈을 import 하지 않고 배포 버전을 알아낸다(가능한 경우)."""
    # importlib.metadata 는 설치 메타데이터만 읽으므로 import 비용이 없다.
    dist_alias = {"cv2": "opencv-python", "PIL": "pillow",
                  "sklearn": "scikit-learn"}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                             # pragma: no cover
        return "설치됨"
    for cand in (dist_alias.get(mod_name, mod_name),
                 mod_name, f"{mod_name}-headless"):
        try:
            return version(cand)
        except PackageNotFoundError:
            continue
        except Exception:                           # noqa: BLE001
            break
    return "설치됨(버전 미확인)"


def check_packages() -> None:
    """requirements 의 주요 패키지 설치 여부를 하나씩 점검한다."""
    for mod_name, label, expect, required in PACKAGES:
        try:
            spec = importlib.util.find_spec(mod_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            add("패키지", label, expect, "미설치",
                FAIL if required else WARN)
        else:
            add("패키지", label, expect, module_version(mod_name), PASS)


def check_torch_threads() -> None:
    """torch 가 설치된 경우 CPU 스레드 설정까지 확인한다."""
    if importlib.util.find_spec("torch") is None:
        add("패키지", "torch CPU 스레드", "4", "torch 미설치", FAIL)
        return
    try:
        import torch                                # noqa: PLC0415
        add("패키지", "torch CPU 스레드", "1 이상",
            str(torch.get_num_threads()), PASS)
        add("패키지", "torch CUDA 사용",
            "False(CPU 전용)", str(torch.cuda.is_available()), PASS)
    except Exception as exc:                        # noqa: BLE001
        add("패키지", "torch CPU 스레드", "4",
            f"로드 실패: {type(exc).__name__}", FAIL)


# ---------------------------------------------------------------- 3) 외부 명령

# (명령, 인자, 표시 이름, 기대값)
COMMANDS = [
    ("java",   ["-version"],  "Zulu OpenJDK", "21.x"),
    ("javac",  ["-version"],  "javac",        "21.x"),
    ("node",   ["-v"],        "Node.js",      "v20 이상"),
    ("npm",    ["-v"],        "npm",          "10.x"),
    ("psql",   ["--version"], "psql",         "16.x"),
    ("git",    ["--version"], "git",          "2.x"),
    ("docker", ["--version"], "docker(선택)",  "최신"),
]


def check_commands() -> None:
    """PATH 에 등록된 외부 도구의 버전을 확인한다."""
    for cmd, args, label, expect in COMMANDS:
        code, out = run_cmd([cmd, *args])
        if code == -1:
            optional = "선택" in label
            add("외부 도구", label, expect, out,
                WARN if optional else FAIL)
        else:
            add("외부 도구", label, expect, first_line(out), PASS)


def check_java_home() -> None:
    """JAVA_HOME 환경 변수가 JDK 21 을 가리키는지 확인한다."""
    jh = os.environ.get("JAVA_HOME", "")
    if not jh:
        add("외부 도구", "JAVA_HOME", "JDK 21 경로", "설정되지 않음", FAIL)
        return
    exists = os.path.isdir(jh)
    looks21 = "21" in jh
    verdict = PASS if (exists and looks21) else WARN
    shown = jh if len(jh) <= 52 else "..." + jh[-49:]
    add("외부 도구", "JAVA_HOME", "JDK 21 경로", shown, verdict)


# ---------------------------------------------------------------- 4) 포트/서비스

# (호스트, 포트, 표시 이름)
PORTS = [
    ("127.0.0.1", 5432, "PostgreSQL :5432"),
    ("127.0.0.1", 8000, "AI 서비스 :8000"),
    ("127.0.0.1", 8080, "백엔드 :8080"),
    ("127.0.0.1", 5173, "프론트엔드 :5173"),
]


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """TCP 연결이 성립하면 포트가 열린 것으로 본다."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_ports() -> None:
    """서비스 포트의 수신 여부를 점검한다."""
    for host, port, label in PORTS:
        ok = port_open(host, port)
        add("포트", label, "LISTEN",
            "열려 있음" if ok else "닫혀 있음", PASS if ok else FAIL)


# (URL, 표시 이름, 기대 JSON 키)
HEALTH = [
    ("http://127.0.0.1:8000/health", "AI 서비스 헬스체크", "status"),
    ("http://127.0.0.1:8080/actuator/health", "백엔드 헬스체크", "status"),
]


def http_get_json(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    """헬스체크 URL 을 호출해 (성공여부, 요약문자열)을 돌려준다."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(4096).decode("utf-8", "replace")
            code = resp.status
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"연결 실패: {exc.reason}"
    except OSError as exc:
        return False, f"연결 실패: {exc}"
    try:
        data = json.loads(raw)
    except ValueError:
        return code == 200, f"HTTP {code} (JSON 아님)"
    status = data.get("status", "?") if isinstance(data, dict) else "?"
    return code == 200 and status == "UP", f"HTTP {code} status={status}"


def check_health() -> None:
    """두 서비스의 헬스체크 엔드포인트를 호출한다."""
    for url, label, _key in HEALTH:
        ok, summary = http_get_json(url)
        add("헬스체크", label, "200 / status=UP", summary,
            PASS if ok else FAIL)


# ---------------------------------------------------------------- 5) 데이터베이스

def check_database() -> None:
    """psql 로 mobility_ai 데이터베이스에 접속해 테이블 수를 센다.

    psql 이 없거나 접속이 실패하면 FAIL 로 기록하고 넘어간다.
    """
    if shutil.which("psql") is None:
        add("데이터베이스", "mobility_ai 접속", "연결 성공",
            "psql 미설치", FAIL)
        return
    env = dict(os.environ)
    env.setdefault("PGPASSWORD", os.environ.get("DB_PASSWORD", "mobility123"))
    sql = ("SELECT count(*) FROM information_schema.tables "
           "WHERE table_schema='public';")
    try:
        proc = subprocess.run(                      # noqa: S603
            [shutil.which("psql"),                  # type: ignore[list-item]
             "-h", os.environ.get("DB_HOST", "localhost"),
             "-p", os.environ.get("DB_PORT", "5432"),
             "-U", os.environ.get("DB_USER", "mobility"),
             "-d", os.environ.get("DB_NAME", "mobility_ai"),
             "-t", "-A", "-c", sql],
            capture_output=True, text=True, timeout=10,
            env=env, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        add("데이터베이스", "mobility_ai 접속", "연결 성공", "시간 초과", FAIL)
        return
    except OSError as exc:
        add("데이터베이스", "mobility_ai 접속", "연결 성공",
            f"실행 실패: {exc}", FAIL)
        return

    if proc.returncode != 0:
        add("데이터베이스", "mobility_ai 접속", "연결 성공",
            first_line(proc.stderr or "접속 실패") or "접속 실패", FAIL)
        add("데이터베이스", "public 테이블 수", "11 (10 + flyway)",
            "확인 불가", FAIL)
        return
    count_text = (proc.stdout or "").strip() or "0"
    add("데이터베이스", "mobility_ai 접속", "연결 성공", "연결 성공", PASS)
    try:
        n = int(count_text)
    except ValueError:
        n = -1
    # 계약서의 테이블 10개 + Flyway 이력 테이블 1개 = 11개가 기준이다.
    add("데이터베이스", "public 테이블 수", "11 (10 + flyway)",
        str(n), PASS if n >= 11 else WARN)


# ---------------------------------------------------------------- 6) 디렉터리

DIRS = [
    ("mobility-ai-project", "완성 프로젝트 루트"),
    ("mobility-ai-project/ai-service", "AI 서비스"),
    ("mobility-ai-project/backend", "백엔드"),
    ("mobility-ai-project/frontend", "프론트엔드"),
]


def check_layout() -> None:
    """작업 디렉터리 구조가 교재 규칙과 맞는지 확인한다."""
    root = os.getcwd()
    add("디렉터리", "현재 작업 경로", "mobility_ai",
        root if len(root) <= 52 else "..." + root[-49:], PASS)
    for rel, label in DIRS:
        ok = os.path.isdir(os.path.join(root, rel))
        add("디렉터리", label, rel,
            "있음" if ok else "없음", PASS if ok else WARN)


# ---------------------------------------------------------------- 표 출력

def render_table(rows: list[tuple[str, str, str, str, str]]) -> str:
    """결과 행을 고정폭 표로 렌더링한다(한글 폭을 2로 계산)."""
    head = ("구분", "점검 항목", "기대값", "실제값", "판정")
    data = [head, *rows]

    def width(text: str) -> int:
        # 한글·한자 등 전각 문자는 폭 2로 계산해 열을 맞춘다.
        return sum(2 if ord(ch) > 0x1100 else 1 for ch in text)

    cols = len(head)
    widths = [max(width(r[c]) for r in data) for c in range(cols)]

    def pad(text: str, w: int) -> str:
        return text + " " * max(0, w - width(text))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [sep]
    for i, r in enumerate(data):
        lines.append("| " + " | ".join(pad(r[c], widths[c])
                                       for c in range(cols)) + " |")
        if i == 0:
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)


def main() -> int:
    """모든 점검을 수행하고 결과 표와 요약을 출력한다."""
    parser = argparse.ArgumentParser(
        description="모빌리티 비전 AI 개발 환경 종합 점검")
    parser.add_argument("--json", action="store_true",
                        help="표 대신 JSON 으로 출력한다")
    parser.add_argument("--skip-service", action="store_true",
                        help="포트·헬스체크·DB 점검을 건너뛴다")
    args = parser.parse_args()

    check_python()
    check_packages()
    check_torch_threads()
    check_commands()
    check_java_home()
    check_layout()
    if not args.skip_service:
        check_ports()
        check_health()
        check_database()

    if args.json:
        payload = [{"group": g, "name": n, "expect": e,
                    "actual": a, "verdict": v} for g, n, e, a, v in ROWS]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("=" * 70)
    print(" 모빌리티 비전 AI — 개발 환경 종합 점검")
    print("=" * 70)
    print(render_table(ROWS))

    n_pass = sum(1 for r in ROWS if r[4] == PASS)
    n_warn = sum(1 for r in ROWS if r[4] == WARN)
    n_fail = sum(1 for r in ROWS if r[4] == FAIL)
    print(f"\n합계 {len(ROWS)}항목 — PASS {n_pass} / WARN {n_warn} / FAIL {n_fail}")
    if n_fail:
        print("FAIL 항목은 부록 A 의 설치 문제 해결표에서 해당 증상을 찾아 조치한다.")
    else:
        print("모든 필수 항목이 통과했다. 완성 프로젝트를 기동할 수 있다.")
    # 점검 스크립트는 실패 항목이 있어도 정상 종료한다(CI 에서 표만 수집).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
