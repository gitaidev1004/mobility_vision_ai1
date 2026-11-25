<script setup>
/**
 * 번호판 인식 화면 (/recognition)
 * - 구성: 이미지 업로드(드래그&드롭) → 검출 박스 오버레이 → 인식 결과 패널 → 인식 이력 표
 * - POST /api/v1/recognitions (multipart: file, gate, eventType) 를 호출한다.
 * - 검출 박스는 응답의 detections[].bbox(원본 픽셀 좌표)를 퍼센트로 변환해 CSS 로 겹친다.
 *   따라서 이미지가 어떤 크기로 축소되어도 박스가 어긋나지 않는다.
 */
import { computed, onMounted, ref } from 'vue'
import PageHeader from '@/components/common/PageHeader.vue'
import DataTable from '@/components/common/DataTable.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import PaginationBar from '@/components/common/PaginationBar.vue'
import AppIcon from '@/components/common/AppIcon.vue'
import BarChart from '@/components/charts/BarChart.vue'
import { useRecognitionStore } from '@/stores/recognition'
import { useDashboardStore } from '@/stores/dashboard'
import {
  formatDateTime,
  formatDuration,
  formatPercent,
  formatBytes,
  formatNumber,
} from '@/utils/format'
import { toastSuccess, toastError, toastWarn } from '@/utils/toast'

const store = useRecognitionStore()
const dashboard = useDashboardStore()

/** 숨겨진 file input 참조 */
const fileInput = ref(null)
/** 드래그 중 하이라이트 */
const dragging = ref(false)

/** 게이트 선택 후보 (계약서 시드 기준) */
const gates = ['정문', '후문', '지하1층']

/** 이력 표 열 정의 */
const historyColumns = [
  { key: 'plateNumber', label: '번호판', width: '120px', cellClass: 'cell-plate' },
  { key: 'confidence', label: '신뢰도', width: '90px', align: 'right' },
  { key: 'status', label: '상태', width: '120px' },
  { key: 'detectorName', label: '검출기', width: '110px' },
  { key: 'ocrName', label: 'OCR', width: '100px' },
  { key: 'totalMs', label: '소요시간', width: '90px', align: 'right' },
  { key: 'qualityScore', label: '품질', width: '80px', align: 'right' },
  { key: 'createdAt', label: '인식 시각', width: '140px', cellClass: 'cell-time' },
]

/** 인식 상태 필터 후보 */
const statusOptions = [
  { value: '', label: '전체 상태' },
  { value: 'SUCCESS', label: '성공' },
  { value: 'LOW_CONFIDENCE', label: '신뢰도 낮음' },
  { value: 'NO_PLATE', label: '번호판 미검출' },
  { value: 'FORMAT_INVALID', label: '형식 오류' },
]

/** 단계별 소요시간 막대그래프 시리즈 */
const elapsedSeries = computed(() => {
  const e = store.elapsed
  return [
    {
      name: '소요시간(ms)',
      color: 'var(--c-chart-1)',
      values: [e.detect ?? 0, e.ocr ?? 0, e.total ?? 0],
    },
  ]
})

/** 문자별 신뢰도(응답에 있으면 표시). */
const charConfidences = computed(() => store.result?.charConfidences ?? [])

/** 파일 선택 버튼 → 숨겨진 input 클릭 */
function openPicker() {
  fileInput.value?.click()
}

/** input change 처리 */
function onFileChange(e) {
  const f = e.target.files?.[0]
  if (f) accept(f)
}

/** 드롭 처리 */
function onDrop(e) {
  dragging.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) accept(f)
}

/** 이미지 파일만 허용한다. */
function accept(f) {
  if (!f.type.startsWith('image/')) {
    toastWarn('이미지 파일만 업로드할 수 있습니다.')
    return
  }
  store.selectFile(f)
}

/** <img> 로드 시 원본 크기를 스토어에 기록한다(박스 좌표 변환용). */
function onImageLoad(e) {
  store.setImageSize(e.target.naturalWidth, e.target.naturalHeight)
}

/** 인식 실행 */
async function onRecognize() {
  const data = await store.submit()
  if (!data) {
    toastError(store.error || '인식에 실패했습니다.')
    return
  }
  if (data.status === 'SUCCESS') {
    toastSuccess(`${data.plateNumber} 인식 성공 (${formatPercent(data.confidence)})`)
  } else {
    toastWarn(`인식 상태: ${data.status}`)
  }
  // 대시보드 지표도 바뀌므로 함께 갱신한다.
  dashboard.fetchSummary()
}

/** 상태 필터 변경 */
function onStatusChange(e) {
  store.fetchHistory({ status: e.target.value, page: 0 })
}

onMounted(() => {
  store.fetchHistory({ page: 0 })
})
</script>

<template>
  <div class="recognition">
    <PageHeader
      title="번호판 인식"
      desc="차량 이미지를 업로드하면 AI 서비스가 번호판을 검출·판독하고 출입 이벤트를 생성합니다."
    >
      <template #actions>
        <RouterLink to="/events" class="btn btn--sm">출입 이벤트</RouterLink>
      </template>
    </PageHeader>

    <!-- 상단: 업로드/오버레이(좌) + 결과 패널(우) -->
    <div class="grid grid--main-side recognition__top">
      <!-- ① 업로드 + 검출 박스 오버레이 -->
      <section class="card">
        <div class="card__head">
          <h3 class="card__title">이미지 업로드 · 검출 결과</h3>
          <div class="card__tools">
            <div class="field recognition__inline-field">
              <label class="sr-only" for="rec-gate">게이트</label>
              <select id="rec-gate" v-model="store.gate" class="select">
                <option v-for="g in gates" :key="g" :value="g">{{ g }}</option>
              </select>
            </div>
            <div class="field recognition__inline-field">
              <label class="sr-only" for="rec-type">이벤트 구분</label>
              <select id="rec-type" v-model="store.eventType" class="select">
                <option value="ENTRY">입차</option>
                <option value="EXIT">출차</option>
              </select>
            </div>
          </div>
        </div>

        <div class="card__body">
          <!-- 드롭존 / 미리보기 -->
          <div
            class="dropzone"
            :class="{ 'is-dragging': dragging, 'has-image': !!store.previewUrl }"
            @dragover.prevent="dragging = true"
            @dragleave.prevent="dragging = false"
            @drop.prevent="onDrop"
          >
            <!-- 이미지가 없을 때: 안내 -->
            <template v-if="!store.previewUrl">
              <AppIcon name="upload" :size="40" :width="1.4" class="dropzone__icon" />
              <div class="dropzone__title">차량 이미지를 끌어다 놓으세요</div>
              <div class="dropzone__desc">JPG · PNG · WebP · 최대 10MB</div>
              <button type="button" class="btn btn--primary" @click="openPicker">
                파일 선택
              </button>
            </template>

            <!-- 이미지가 있을 때: 미리보기 + 검출 박스 -->
            <div v-else class="canvas">
              <img
                :src="store.previewUrl"
                alt="업로드한 차량 이미지"
                class="canvas__img"
                @load="onImageLoad"
              />

              <!-- 검출 박스 오버레이: 퍼센트 좌표라 이미지 크기와 무관하게 정확히 겹친다. -->
              <div
                v-for="b in store.detectionBoxes"
                :key="b.index"
                class="detbox"
                :style="{
                  left: `${b.left}%`,
                  top: `${b.top}%`,
                  width: `${b.width}%`,
                  height: `${b.height}%`,
                }"
              >
                <span class="detbox__label">
                  {{ b.label }} {{ b.score.toFixed(2) }}
                </span>
              </div>
            </div>
          </div>

          <!-- 업로드 진행 / 조작 버튼 -->
          <div class="recognition__controls">
            <div v-if="store.file" class="recognition__fileinfo">
              <AppIcon name="scan" :size="15" />
              <span class="recognition__filename">{{ store.file.name }}</span>
              <span class="recognition__filesize">{{ formatBytes(store.file.size) }}</span>
              <span v-if="store.imageSize.width" class="recognition__filesize">
                {{ store.imageSize.width }}×{{ store.imageSize.height }}
              </span>
            </div>
            <div class="recognition__btns">
              <button type="button" class="btn btn--sm" @click="openPicker">
                다른 이미지
              </button>
              <button
                type="button"
                class="btn btn--sm"
                :disabled="!store.file"
                @click="store.clearResult()"
              >
                초기화
              </button>
              <button
                type="button"
                class="btn btn--primary"
                :disabled="!store.canSubmit"
                @click="onRecognize"
              >
                <span v-if="store.recognizing" class="spinner" aria-hidden="true" />
                {{ store.recognizing ? '인식 중…' : '인식 실행' }}
              </button>
            </div>
          </div>

          <!-- 업로드 진행 바 -->
          <div v-if="store.recognizing" class="meter recognition__progress">
            <div class="meter__fill" :style="{ width: `${store.progress}%` }" />
          </div>

          <!-- 검출 상세: AI 응답의 detections[] 를 그대로 표로 보여준다. -->
          <div v-if="store.detections.length" class="recognition__detections">
            <div class="recognition__block-title">검출 상세 ({{ store.detections.length }}건)</div>
            <table class="table table--compact">
              <caption>
                검출된 객체 목록. 번호, 라벨, 신뢰도, 바운딩 박스 좌표, 크기를 보여줍니다.
              </caption>
              <thead>
                <tr>
                  <th scope="col" style="width: 50px">#</th>
                  <th scope="col" style="width: 90px">라벨</th>
                  <th scope="col" class="is-num" style="width: 90px">신뢰도</th>
                  <th scope="col">bbox [x1, y1, x2, y2]</th>
                  <th scope="col" class="is-num" style="width: 110px">크기(px)</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(d, i) in store.detections" :key="i">
                  <td>{{ i + 1 }}</td>
                  <td>
                    <span class="badge badge--success">{{ d.label || 'plate' }}</span>
                  </td>
                  <td class="is-num">{{ formatPercent(d.score) }}</td>
                  <td class="recognition__bbox">[{{ (d.bbox || []).join(', ') }}]</td>
                  <td class="is-num">
                    {{ (d.bbox?.[2] ?? 0) - (d.bbox?.[0] ?? 0) }}×{{ (d.bbox?.[3] ?? 0) - (d.bbox?.[1] ?? 0) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 품질 / 문자별 신뢰도: 넓은 좌측 카드에 두어 두 열의 높이를 맞춘다. -->
          <div v-if="store.result" class="recognition__quality-grid">
            <div v-if="store.quality">
              <div class="recognition__block-title">이미지 품질</div>
              <div class="meter">
                <div
                  class="meter__fill"
                  :class="store.quality.readable ? 'meter__fill--success' : 'meter__fill--warn'"
                  :style="{ width: `${(store.quality.score ?? 0) * 100}%` }"
                />
              </div>
              <div class="recognition__quality-row">
                <span>점수 {{ formatPercent(store.quality.score) }}</span>
                <span>{{ store.quality.readable ? '판독 가능' : '판독 어려움' }}</span>
              </div>
            </div>

            <div v-if="charConfidences.length">
              <div class="recognition__block-title">문자별 신뢰도</div>
              <div class="charconf">
                <span
                  v-for="(c, i) in charConfidences"
                  :key="i"
                  class="charconf__item"
                  :class="{ 'is-low': c < 0.9 }"
                >
                  <span class="charconf__char">
                    {{ (store.result.plateNumber || '').charAt(i) || '?' }}
                  </span>
                  <span class="charconf__val">{{ (c * 100).toFixed(0) }}</span>
                </span>
              </div>
            </div>
          </div>

          <!-- 숨겨진 파일 입력 -->
          <input
            ref="fileInput"
            type="file"
            accept="image/*"
            class="sr-only"
            aria-label="차량 이미지 파일 선택"
            @change="onFileChange"
          />
        </div>
      </section>

      <!-- ② 인식 결과 패널 -->
      <section class="card">
        <div class="card__head">
          <h3 class="card__title">인식 결과</h3>
          <StatusBadge
            v-if="store.result"
            kind="recognition"
            :value="store.result.status"
          />
        </div>

        <div class="card__body">
          <!-- 아직 결과가 없을 때 -->
          <div v-if="!store.result" class="recognition__placeholder">
            <AppIcon name="scan" :size="40" :width="1.4" />
            <p>이미지를 업로드하고 <strong>인식 실행</strong>을 누르면<br />결과가 표시됩니다.</p>
          </div>

          <template v-else>
            <!-- 판독된 번호판 -->
            <div class="plate-result">
              <div class="plate-result__plate">
                {{ store.result.plateNumber || '미검출' }}
              </div>
              <div class="plate-result__conf">
                신뢰도 {{ formatPercent(store.result.confidence) }}
                <span v-if="store.result.validFormat" class="badge badge--success">형식 정상</span>
                <span v-else class="badge badge--danger">형식 오류</span>
              </div>
            </div>

            <!-- 등록 차량 매칭 -->
            <div class="recognition__block">
              <div class="recognition__block-title">등록 차량 매칭</div>
              <dl v-if="store.matchedVehicle" class="kv">
                <dt>차주</dt>
                <dd>{{ store.matchedVehicle.ownerName || '-' }}</dd>
                <dt>번호판</dt>
                <dd>{{ store.matchedVehicle.plateNumber }}</dd>
                <dt>정기권</dt>
                <dd>{{ store.matchedVehicle.passType || '-' }}</dd>
              </dl>
              <p v-else class="recognition__muted">등록되지 않은 차량입니다.</p>
            </div>

            <!-- 출입 처리 결과 -->
            <div class="recognition__block">
              <div class="recognition__block-title">출입 처리</div>
              <dl class="kv">
                <dt>구분</dt>
                <dd>
                  <StatusBadge kind="event" :value="store.result.eventType" />
                </dd>
                <dt>허용 여부</dt>
                <dd>
                  <StatusBadge kind="allowed" :value="store.result.allowed" />
                </dd>
                <dt v-if="store.result.rejectReason">거부 사유</dt>
                <dd v-if="store.result.rejectReason">{{ store.result.rejectReason }}</dd>
                <dt>이벤트 ID</dt>
                <dd>{{ store.result.eventId ?? '-' }}</dd>
                <dt>주차 세션</dt>
                <dd>{{ store.result.parkingSessionId ?? '-' }}</dd>
                <dt v-if="store.result.parkedMinutes !== null">주차 시간</dt>
                <dd v-if="store.result.parkedMinutes !== null">
                  {{ formatNumber(store.result.parkedMinutes) }}분
                </dd>
              </dl>
            </div>

            <!-- 소요시간 -->
            <div class="recognition__block">
              <div class="recognition__block-title">추론 소요시간</div>
              <!-- 좁은 사이드 패널이므로 viewBox 폭을 줄여 글자가 뭉개지지 않게 한다. -->
              <BarChart
                :labels="['검출', 'OCR', '합계']"
                :series="elapsedSeries"
                :width="340"
                :height="150"
                :ticks="3"
                :legend="false"
                show-values
                :value-format="(v) => v.toFixed(1)"
                unit="ms"
              />
              <dl class="kv recognition__elapsed-kv">
                <dt>검출</dt>
                <dd>{{ formatDuration(store.elapsed.detect) }}</dd>
                <dt>OCR</dt>
                <dd>{{ formatDuration(store.elapsed.ocr) }}</dd>
                <dt>합계</dt>
                <dd>{{ formatDuration(store.elapsed.total) }}</dd>
              </dl>
            </div>

          </template>
        </div>
      </section>
    </div>

    <!-- ③ 인식 이력 -->
    <section class="card">
      <div class="card__head">
        <h3 class="card__title">인식 이력</h3>
        <span class="card__hint">총 {{ formatNumber(store.historyTotal) }}건</span>
        <div class="card__tools">
          <div class="field recognition__inline-field">
            <label class="sr-only" for="rec-filter">인식 상태 필터</label>
            <select
              id="rec-filter"
              class="select"
              :value="store.historyStatus"
              @change="onStatusChange"
            >
              <option v-for="o in statusOptions" :key="o.value" :value="o.value">
                {{ o.label }}
              </option>
            </select>
          </div>
          <button
            type="button"
            class="btn btn--sm"
            aria-label="인식 이력 새로고침"
            @click="store.fetchHistory()"
          >
            <AppIcon name="refresh" :size="15" />
          </button>
        </div>
      </div>

      <div class="card__body card__body--flush">
        <DataTable
          caption="번호판 인식 이력. 번호판, 신뢰도, 상태, 검출기, OCR 모델, 소요시간, 품질, 인식 시각을 보여줍니다."
          :columns="historyColumns"
          :rows="store.history"
          :loading="store.historyLoading"
          empty-title="인식 이력이 없습니다"
          empty-desc="이미지를 업로드해 첫 인식을 실행해 보세요."
          empty-icon="scan"
          compact
        >
          <template #cell-confidence="{ row }">
            {{ formatPercent(row.confidence) }}
          </template>
          <template #cell-status="{ row }">
            <StatusBadge kind="recognition" :value="row.status" />
          </template>
          <template #cell-totalMs="{ row }">
            {{ formatDuration(row.totalMs) }}
          </template>
          <template #cell-qualityScore="{ row }">
            {{ formatPercent(row.qualityScore) }}
          </template>
          <template #cell-createdAt="{ row }">
            {{ formatDateTime(row.createdAt) }}
          </template>
        </DataTable>
      </div>

      <div v-if="store.historyTotalPages > 1" class="card__foot">
        <PaginationBar
          :page="store.historyPage"
          :total-pages="store.historyTotalPages"
          :total-elements="store.historyTotal"
          :size="store.historySize"
          @change="(p) => store.fetchHistory({ page: p })"
        />
      </div>
    </section>
  </div>
</template>

<style scoped>
.recognition {
  display: flex;
  flex-direction: column;
  gap: var(--sp-5);
}

/* 카드 머리글 안의 셀렉트는 라벨 없이 폭만 고정한다. */
.recognition__inline-field {
  width: 112px;
}

/*
 * 좌(업로드)·우(결과) 카드의 높이가 크게 다르므로 stretch 를 끈다.
 * 그렇지 않으면 짧은 쪽 카드 아래에 큰 빈 공간이 생긴다.
 */
.recognition__top {
  align-items: start;
}

/* ---- 드롭존 ---- */
.dropzone {
  border: 2px dashed var(--c-border-strong);
  border-radius: var(--radius);
  background: var(--c-surface-alt);
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--sp-3);
  padding: var(--sp-6);
  text-align: center;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.dropzone.is-dragging {
  border-color: var(--c-primary);
  background: var(--c-primary-soft);
}

.dropzone.has-image {
  padding: var(--sp-3);
  background: #0f172a;
  border-style: solid;
  border-color: var(--c-border);
}

.dropzone__icon {
  color: var(--c-text-muted);
}

.dropzone__title {
  font-size: var(--fs-md);
  font-weight: 700;
}

.dropzone__desc {
  font-size: var(--fs-base);
  color: var(--c-text-muted);
}

/* ---- 검출 박스 오버레이 캔버스 ---- */
/*
 * 오버레이 기준 상자. position:relative 로 두고 검출 박스를 퍼센트 좌표로 올린다.
 * inline-block 이라 상자 크기가 이미지 크기와 정확히 일치한다(박스가 어긋나지 않음).
 */
.canvas {
  position: relative;
  display: inline-block;
  max-width: 100%;
  line-height: 0;
}

.canvas__img {
  max-width: 100%;
  max-height: 440px;
  width: auto;
  height: auto;
  display: block;
  border-radius: var(--radius-sm);
}

.detbox {
  position: absolute;
  border: 2px solid #22c55e;
  border-radius: 2px;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);
  pointer-events: none;
}

.detbox__label {
  position: absolute;
  left: -2px;
  top: -20px;
  background: #22c55e;
  color: #052e16;
  font-size: 11px;
  font-weight: 800;
  line-height: 1;
  padding: 3px 6px;
  border-radius: 3px;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

/* ---- 조작 영역 ---- */
.recognition__controls {
  display: flex;
  align-items: center;
  gap: var(--sp-3);
  margin-top: var(--sp-4);
  flex-wrap: wrap;
}

.recognition__fileinfo {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  font-size: var(--fs-sm);
  color: var(--c-text-sub);
  min-width: 0;
}

.recognition__filename {
  font-weight: 600;
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.recognition__filesize {
  color: var(--c-text-muted);
  font-variant-numeric: tabular-nums;
}

.recognition__btns {
  margin-left: auto;
  display: flex;
  gap: var(--sp-2);
}

.recognition__progress {
  margin-top: var(--sp-3);
}

/* 검출 상세 표 */
.recognition__detections {
  margin-top: var(--sp-5);
  padding-top: var(--sp-4);
  border-top: 1px solid var(--c-border);
}

.recognition__bbox {
  font-family: var(--font-mono);
  font-size: var(--fs-sm);
  color: var(--c-text-sub);
}

/* 품질 + 문자별 신뢰도를 2열로 나란히 둔다. */
.recognition__quality-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: var(--sp-5);
  margin-top: var(--sp-5);
  padding-top: var(--sp-4);
  border-top: 1px solid var(--c-border);
}

@media (max-width: 860px) {
  .recognition__quality-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}

/* ---- 결과 패널 ---- */
.recognition__placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--sp-3);
  padding: var(--sp-8) var(--sp-4);
  text-align: center;
  color: var(--c-text-muted);
  font-size: var(--fs-base);
  line-height: 1.7;
}

.plate-result {
  background: var(--c-bg);
  border-radius: var(--radius);
  padding: var(--sp-5);
  text-align: center;
}

.plate-result__plate {
  font-size: 30px;
  font-weight: 800;
  letter-spacing: 0.02em;
  color: #fff;
  font-variant-numeric: tabular-nums;
}

.plate-result__conf {
  margin-top: var(--sp-2);
  font-size: var(--fs-base);
  color: #a8b6cc;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--sp-2);
}

.recognition__block {
  margin-top: var(--sp-5);
  padding-top: var(--sp-4);
  border-top: 1px solid var(--c-border);
}

.recognition__block-title {
  font-size: var(--fs-sm);
  font-weight: 700;
  color: var(--c-text-sub);
  margin-bottom: var(--sp-3);
}

.recognition__muted {
  font-size: var(--fs-base);
  color: var(--c-text-muted);
}

.recognition__elapsed-kv {
  margin-top: var(--sp-3);
}

.recognition__quality-row {
  display: flex;
  justify-content: space-between;
  font-size: var(--fs-sm);
  color: var(--c-text-sub);
  font-weight: 600;
  margin-top: var(--sp-2);
}

/* ---- 문자별 신뢰도 ---- */
.charconf {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
}

.charconf__item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  min-width: 34px;
  padding: 5px 4px;
  border: 1px solid var(--c-border);
  border-radius: var(--radius-sm);
  background: var(--c-surface-alt);
}

.charconf__item.is-low {
  border-color: var(--c-warn);
  background: var(--c-warn-soft);
}

.charconf__char {
  font-size: var(--fs-md);
  font-weight: 800;
}

.charconf__val {
  font-size: 10px;
  font-weight: 700;
  color: var(--c-text-muted);
  font-variant-numeric: tabular-nums;
}
</style>
