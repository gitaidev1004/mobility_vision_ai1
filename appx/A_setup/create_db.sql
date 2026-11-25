-- 부록 A (7) 데이터베이스 구축
-- 계약서가 정한 값: DB mobility_ai / 사용자 mobility / 비밀번호 mobility123
-- 슈퍼유저(postgres)로 접속한 상태에서 실행한다.

-- 1) 애플리케이션 전용 계정 생성.
--    LOGIN 권한만 주고 슈퍼유저 권한은 주지 않는다(최소 권한 원칙).
CREATE USER mobility WITH LOGIN PASSWORD 'mobility123';

-- 2) 데이터베이스 생성. 소유자를 mobility 로 지정해야
--    Flyway 마이그레이션이 테이블을 만들 수 있다.
--    인코딩은 UTF8 로 고정한다(한글 소유자명·번호판 저장).
CREATE DATABASE mobility_ai
    OWNER    mobility
    ENCODING 'UTF8'
    TEMPLATE template0;

-- 3) 세션 기본 시간대를 한국 표준시로 맞춘다.
--    백엔드는 timezone 없는 TIMESTAMP 를 쓰므로 서버 시간대가 중요하다.
ALTER DATABASE mobility_ai SET timezone TO 'Asia/Seoul';

-- 4) public 스키마 사용·생성 권한을 명시적으로 부여한다.
--    PostgreSQL 15 부터 public 스키마의 기본 CREATE 권한이 회수되었으므로
--    이 구문을 빼면 Flyway 가 테이블을 만들지 못한다.
\connect mobility_ai
GRANT ALL ON SCHEMA public TO mobility;
ALTER SCHEMA public OWNER TO mobility;

-- 5) 번호판 부분 일치 검색에 사용할 확장을 설치한다.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 6) 결과 확인
\du mobility
\l mobility_ai
