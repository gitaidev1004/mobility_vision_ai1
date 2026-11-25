# db 디렉터리

| 파일 | 용도 |
|---|---|
| `00_init_extension.sql` | PostgreSQL 컨테이너 최초 기동 시 1회 실행. 확장(pg_trgm, uuid-ossp)과 시간대만 설정한다. |
| `01_schema.sql` | 테이블/인덱스 생성 DDL. `backend/src/main/resources/db/migration/V1__init.sql` 의 사본이다. |
| `02_seed.sql` | 초기 데이터(계정·차량·정책·센서·예측·벤치마크). `V2__seed.sql` 의 사본이다. |

## 주의

스키마의 **정본은 백엔드의 Flyway 마이그레이션**이다. 백엔드를 기동하면
`V1__init.sql` → `V2__seed.sql` 이 자동으로 적용되므로 보통은 이 디렉터리의
SQL 을 직접 실행할 필요가 없다.

여기 있는 사본은 다음 경우에 쓴다.

- 백엔드 없이 DB 구조만 먼저 확인하고 싶을 때
- 교재 6.4절의 DDL 을 터미널에서 직접 따라 해 볼 때
- DBA 에게 스키마만 전달해야 할 때

직접 실행하려면 아래 순서를 따른다. 이 경우 Flyway 이력 테이블이 없으므로,
이후 백엔드를 기동하면 Flyway 가 같은 마이그레이션을 다시 적용하려다 실패한다.
백엔드와 함께 쓸 DB 에는 직접 실행하지 않는 편이 안전하다.

```bash
psql -U mobility -d mobility_ai -f 01_schema.sql
psql -U mobility -d mobility_ai -f 02_seed.sql
```
