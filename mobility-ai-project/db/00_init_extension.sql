-- PostgreSQL 컨테이너 최초 기동 시 1회 실행된다.
-- 스키마(테이블)는 백엔드의 Flyway 마이그레이션이 생성하므로
-- 여기서는 확장과 타임존만 준비한다.

-- 유사 문자열 검색(번호판 부분 일치 검색)에 사용한다.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- UUID 가 필요한 경우를 대비해 준비해 둔다.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 세션 기본 시간대를 한국 표준시로 맞춘다.
ALTER DATABASE mobility_ai SET timezone TO 'Asia/Seoul';
