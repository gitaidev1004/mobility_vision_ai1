<script setup>
/**
 * 예지보전 화면 (/pdm)
 * - 차량 선택 → 건강점수 게이지 + 센서 추이(LineChart) + 부품별 점수 + 정비 권고
 * - 엔드포인트: GET /pdm/vehicles/{id}/latest, POST /pdm/vehicles/{id}/predict,
 *               GET /sensors?vehicleId=&limit=, GET /pdm/risky?level=
 */
import { computed, onMounted } from 'vue'
import PageHeader from '@/components/common/PageHeader.vue'
import StatCard from '@/components/common/StatCard.vue'
import DataTable from '@/components/common/DataTable.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import AppIcon from '@/components/common/AppIcon.vue'
import GaugeChart from '@/components/charts/GaugeChart.vue'
import LineChart from '@/components/charts/LineChart.vue'
import BarChart from '@/components/charts/BarChart.vue'
import { usePdmStore, COMPONENT_LABEL } from '@/stores/pdm'
import { useAuthStore } from '@/stores/auth'
import {
  formatDateTime,
  formatPercent,
  formatDays,
  formatKm,
  formatNumber,
} from '@/utils/format'
import { toastSuccess, toastError } from '@/utils/toast'

const store = usePdmStore()
const auth = useAuthStore()

/** 위험 차량 표 열 정의 */
const riskyColumns = [
  { key: 'plateNumber', label: '번호판', width: '120px', cellClass: 'cell-plate' },
  { key: 'healthScore', label: '건강점수', width: '90px', align: 'right' },
  { key: 'riskLevel', label: '위험도', width: '90px', align: 'center' },
  { key: 'rulDays', label: '잔존수명', width: '90px', align: 'right' },
  { key: 'topComponent', label: '주의 부품', width: '100px' },
  { key: 'predictedAt', label: '예측 시각', width: '140px', cellClass: 'cell-time' },
]

/** 부품별 점수 가로 막대그래프 데이터 */
const componentBars = computed(() => ({
  labels: store.components.map((c) => c.label),
  series: [
    {
      name: '부품 점수',
      color: 'var(--c-chart-1)',
      values: store.components.map((c) => c.score ?? 0),
    },
  ],
}))

/** 타이어 공기압 막대그래프 데이터 */
const tireBars = computed(() => ({
  labels: store.tirePressures.map((t) => t.pos),
  series: [
    {
      name: '공기압(psi)',
      color: 'var(--c-chart-6)',
      values: store.tirePressures.map((t) => t.value),
    },
  ],
}))

/** 차량 선택 변경 */
function onVehicleChange(e) {
  store.selectVehicle(e.target.value)
}

/** 예측 재실행 */
async function onPredict() {
  const data = await store.runPredict()
  if (!data) {
    toastError(store.error || '예측에 실패했습니다.')
    return
  }
  toastSuccess(`예측 완료 — 건강점수 ${data.healthScore}점 (${data.riskLevel})`)
}

/** 위험도 등급 필터 변경 */
function onRiskyLevelChange(e) {
  store.fetchRisky(e.target.value)
}

onMounted(async () => {
  await store.fetchVehicles()
  if (store.selectedId) await store.loadDetail()
  store.fetchRisky()
})
</script>

<template>
  <div class="pdm">
    <PageHeader
      title="예지보전"
      desc="센서 시계열을 기반으로 차량 건강 상태와 고장 위험을 예측합니다."
    >
      <template #actions>
        <div class="field pdm__vehicle-select">
          <label class="sr-only" for="pdm-vehicle">차량 선택</label>
          <select
            id="pdm-vehicle"
            class="select"
            :value="store.selectedId ?? ''"
            @change="onVehicleChange"
          >
            <option value="" disabled>차량을 선택하세요</option>
            <option v-for="v in store.vehicles" :key="v.id" :value="v.id">
              {{ v.plateNumber }} · {{ v.ownerName }}
            </option>
          </select>
        </div>
        <button
          type="button"
          class="btn btn--primary"
          :disabled="!store.selectedId || store.predicting || !auth.canWrite"
          aria-label="예측 재실행"
          @click="onPredict"
        >
          <span v-if="store.predicting" class="spinner" aria-hidden="true" />
          <AppIcon v-if="!store.predicting" name="pulse" :size="16" />
          {{ store.predicting ? '예측 중…' : '예측 실행' }}
        </button>
      </template>
    </PageHeader>

    <!-- 차량이 선택되지 않았을 때 -->
    <section v-if="!store.selectedId" class="card">
      <div class="card__body">
        <EmptyState
          title="차량을 선택하세요"
          desc="상단 드롭다운에서 차량을 고르면 건강 상태와 센서 추이가 표시됩니다."
          icon="car"
        />
      </div>
    </section>

    <template v-else>
      <!-- 예측 결과가 아직 없을 때 -->
      <section v-if="!store.prediction && !store.loading" class="card">
        <div class="card__body">
          <EmptyState
            title="예측 이력이 없습니다"
            :desc="`${store.selectedVehicle?.plateNumber ?? ''} 차량의 예측을 실행하면 결과가 저장됩니다.`"
            icon="pulse"
          >
            <button
              type="button"
              class="btn btn--primary"
              :disabled="!auth.canWrite || store.predicting"
              @click="onPredict"
            >
              예측 실행
            </button>
          </EmptyState>
        </div>
      </section>

      <template v-else>
        <!-- ① 요약 카드 -->
        <div class="grid grid--stats">
          <StatCard
            label="고장 확률"
            :value="store.failureRisk"
            format="percent"
            tone="danger"
            icon="alert"
            :loading="store.loading"
          />
          <StatCard
            label="이상치 점수"
            :value="store.anomalyScore"
            format="percent"
            tone="warn"
            icon="pulse"
            :loading="store.loading"
          />
          <StatCard
            label="잔존수명(RUL)"
            :value="store.rulDays"
            unit="일"
            tone="info"
            icon="clock"
            :loading="store.loading"
          />
          <StatCard
            label="주의 부품"
            :value="COMPONENT_LABEL[store.topComponent] || store.topComponent || '-'"
            tone="primary"
            icon="wrench"
            format="text"
            :loading="store.loading"
          />
        </div>

        <!-- ② 게이지 + 부품별 점수 -->
        <div class="grid grid--side-main">
          <section class="card">
            <div class="card__head">
              <h3 class="card__title">건강 점수</h3>
              <StatusBadge kind="risk" :value="store.riskLevel" />
            </div>
            <div class="card__body">
              <GaugeChart
                :value="store.healthScore"
                :min="0"
                :max="100"
                :tone="store.healthTone"
                label="Health Score"
                unit="점"
              />

              <dl class="kv pdm__kv">
                <dt>차량</dt>
                <dd>{{ store.selectedVehicle?.plateNumber ?? '-' }}</dd>
                <dt>차주</dt>
                <dd>{{ store.selectedVehicle?.ownerName ?? '-' }}</dd>
                <dt>모델</dt>
                <dd>{{ store.selectedVehicle?.modelName ?? '-' }}</dd>
                <dt>예측 시각</dt>
                <dd>{{ formatDateTime(store.prediction?.computedAt ?? store.prediction?.predictedAt) }}</dd>
                <dt>모델 버전</dt>
                <dd class="pdm__mono">{{ store.prediction?.modelVersion ?? '-' }}</dd>
              </dl>
            </div>
          </section>

          <section class="card">
            <div class="card__head">
              <h3 class="card__title">부품별 건강 점수</h3>
              <span class="card__hint">낮은 점수부터 정렬</span>
            </div>
            <div class="card__body">
              <BarChart
                v-if="componentBars.labels.length"
                :labels="componentBars.labels"
                :series="componentBars.series"
                :width="720"
                :height="210"
                :max-value="100"
                :ticks="4"
                horizontal
                show-values
                :value-format="(v) => v.toFixed(1)"
                :legend="false"
                unit="점"
              />
              <EmptyState
                v-else
                title="부품 점수 데이터가 없습니다"
                icon="chip"
              />

              <!-- 부품 상태/사유 목록 -->
              <ul v-if="store.components.length" class="comp-list">
                <li v-for="c in store.components" :key="c.name" class="comp-list__item">
                  <span class="comp-list__name">
                    {{ c.label }}
                    <span v-if="c.isTop" class="badge badge--danger">주의</span>
                  </span>
                  <span class="comp-list__score num">{{ (c.score ?? 0).toFixed(1) }}</span>
                  <StatusBadge kind="component" :value="c.status" />
                  <span class="comp-list__reason">{{ c.reason || '-' }}</span>
                </li>
              </ul>
            </div>
          </section>
        </div>

        <!-- ③ 센서 추이 -->
        <section class="card">
          <div class="card__head">
            <h3 class="card__title">센서 추이</h3>
            <span class="card__hint">
              최근 {{ store.readings.length }}개 측정값
              <template v-if="store.latestReading">
                · 주행 {{ formatKm(store.latestReading.odometerKm ?? store.latestReading.odometer_km) }}
              </template>
            </span>
          </div>
          <div class="card__body">
            <div v-if="store.readings.length" class="grid grid--2 pdm__sensor-grid">
              <div class="pdm__sensor">
                <div class="pdm__sensor-title">냉각수 온도 (°C)</div>
                <LineChart
                  :labels="store.sensorLabels"
                  :series="store.engineTempSeries"
                  :height="170"
                  :label-every="6"
                  :zero-based="false"
                  area
                  :legend="false"
                  unit="°C"
                />
              </div>

              <div class="pdm__sensor">
                <div class="pdm__sensor-title">배터리 전압 (V)</div>
                <LineChart
                  :labels="store.sensorLabels"
                  :series="store.batterySeries"
                  :height="170"
                  :label-every="6"
                  :zero-based="false"
                  area
                  :legend="false"
                  unit="V"
                />
              </div>

              <div class="pdm__sensor">
                <div class="pdm__sensor-title">브레이크 패드 두께 (mm)</div>
                <LineChart
                  :labels="store.sensorLabels"
                  :series="store.brakeSeries"
                  :height="170"
                  :label-every="6"
                  :zero-based="false"
                  area
                  :legend="false"
                  unit="mm"
                />
              </div>

              <div class="pdm__sensor">
                <div class="pdm__sensor-title">진동 RMS</div>
                <LineChart
                  :labels="store.sensorLabels"
                  :series="store.vibrationSeries"
                  :height="170"
                  :label-every="6"
                  :zero-based="false"
                  area
                  :legend="false"
                />
              </div>
            </div>

            <EmptyState
              v-else
              title="센서 데이터가 없습니다"
              desc="시뮬레이터로 센서 데이터를 적재하면 추이가 표시됩니다."
              icon="chip"
            />

            <!-- 타이어 공기압 -->
            <div v-if="store.tirePressures.length" class="pdm__tire">
              <div class="pdm__sensor-title">타이어 공기압 (psi, 최신값)</div>
              <BarChart
                :labels="tireBars.labels"
                :series="tireBars.series"
                :width="720"
                :height="150"
                :ticks="4"
                show-values
                :value-format="(v) => v.toFixed(1)"
                :legend="false"
                unit="psi"
              />
            </div>
          </div>
        </section>

        <!-- ④ 정비 권고 -->
        <section class="card">
          <div class="card__head">
            <h3 class="card__title">정비 권고</h3>
            <span class="card__hint">우선순위 순</span>
            <div class="card__tools">
              <RouterLink to="/maintenance" class="btn btn--sm">정비 등록</RouterLink>
            </div>
          </div>
          <div class="card__body">
            <ol v-if="store.recommendations.length" class="reco">
              <li v-for="r in store.recommendations" :key="r.priority" class="reco__item">
                <span class="reco__priority">{{ r.priority }}</span>
                <div class="reco__body">
                  <div class="reco__title">{{ r.item }}</div>
                  <div class="reco__meta">
                    <span v-if="r.dueKm">{{ formatKm(r.dueKm) }} 이내</span>
                    <span v-if="r.dueDays">{{ formatDays(r.dueDays) }} 이내</span>
                  </div>
                </div>
              </li>
            </ol>
            <EmptyState
              v-else
              title="권고 사항이 없습니다"
              desc="현재 센서 데이터로는 즉시 정비가 필요한 항목이 없습니다."
              icon="check"
            />
          </div>
        </section>
      </template>
    </template>

    <!-- ⑤ 위험 차량 목록 -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">위험 차량</h3>
        <span class="card__hint">{{ store.risky.length }}대</span>
        <div class="card__tools">
          <div class="field pdm__risky-filter">
            <label class="sr-only" for="pdm-level">위험 등급</label>
            <select
              id="pdm-level"
              class="select"
              :value="store.riskyLevel"
              @change="onRiskyLevelChange"
            >
              <option value="MEDIUM">보통 이상</option>
              <option value="HIGH">높음 이상</option>
              <option value="CRITICAL">심각</option>
            </select>
          </div>
        </div>
      </div>

      <div class="card__body card__body--flush">
        <DataTable
          caption="위험 차량 목록. 번호판, 건강점수, 위험도, 잔존수명, 주의 부품, 예측 시각을 보여줍니다."
          :columns="riskyColumns"
          :rows="store.risky"
          empty-title="위험 차량이 없습니다"
          empty-desc="선택한 등급 이상의 위험 차량이 없습니다."
          empty-icon="check"
          compact
          @row-click="(row) => row.vehicleId && store.selectVehicle(row.vehicleId)"
        >
          <template #cell-healthScore="{ row }">
            {{ formatNumber(row.healthScore, 1) }}
          </template>
          <template #cell-riskLevel="{ row }">
            <StatusBadge kind="risk" :value="row.riskLevel" />
          </template>
          <template #cell-rulDays="{ row }">
            {{ formatDays(row.rulDays) }}
          </template>
          <template #cell-topComponent="{ row }">
            {{ COMPONENT_LABEL[row.topComponent] || row.topComponent || '-' }}
          </template>
          <template #cell-predictedAt="{ row }">
            {{ formatDateTime(row.predictedAt) }}
          </template>
        </DataTable>
      </div>
    </section>
  </div>
</template>

<style scoped>
.pdm {
  display: flex;
  flex-direction: column;
  gap: var(--sp-5);
}

.pdm__vehicle-select {
  width: 230px;
}

.pdm__risky-filter {
  width: 130px;
}

.pdm__kv {
  margin-top: var(--sp-5);
  padding-top: var(--sp-4);
  border-top: 1px solid var(--c-border);
}

.pdm__mono {
  font-family: var(--font-mono);
  font-size: var(--fs-sm);
}

/* ---- 부품별 상태 목록 ---- */
.comp-list {
  margin-top: var(--sp-5);
  padding-top: var(--sp-4);
  border-top: 1px solid var(--c-border);
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.comp-list__item {
  display: grid;
  grid-template-columns: 110px 54px 72px minmax(0, 1fr);
  align-items: center;
  gap: var(--sp-3);
  padding: 7px 0;
  font-size: var(--fs-base);
  border-bottom: 1px solid var(--c-border);
}

.comp-list__item:last-child {
  border-bottom: 0;
}

.comp-list__name {
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 6px;
}

.comp-list__score {
  font-weight: 700;
  text-align: right;
}

.comp-list__reason {
  color: var(--c-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ---- 센서 그리드 ---- */
.pdm__sensor-grid {
  gap: var(--sp-5);
}

.pdm__sensor-title {
  font-size: var(--fs-sm);
  font-weight: 700;
  color: var(--c-text-sub);
  margin-bottom: var(--sp-2);
}

.pdm__tire {
  margin-top: var(--sp-5);
  padding-top: var(--sp-4);
  border-top: 1px dashed var(--c-border);
}

/* ---- 정비 권고 ---- */
.reco {
  display: flex;
  flex-direction: column;
  gap: var(--sp-3);
  counter-reset: reco;
}

.reco__item {
  display: flex;
  align-items: flex-start;
  gap: var(--sp-3);
  padding: var(--sp-3) var(--sp-4);
  background: var(--c-surface-alt);
  border: 1px solid var(--c-border);
  border-radius: var(--radius-sm);
}

.reco__priority {
  width: 24px;
  height: 24px;
  flex: 0 0 24px;
  border-radius: 50%;
  background: var(--c-primary);
  color: #fff;
  font-size: var(--fs-sm);
  font-weight: 800;
  display: grid;
  place-items: center;
}

.reco__title {
  font-size: var(--fs-md);
  font-weight: 700;
}

.reco__meta {
  display: flex;
  gap: var(--sp-3);
  font-size: var(--fs-sm);
  color: var(--c-text-muted);
  margin-top: 2px;
  font-variant-numeric: tabular-nums;
}

@media (max-width: 1280px) {
  .comp-list__item {
    grid-template-columns: 96px 50px 66px minmax(0, 1fr);
  }
}
</style>
