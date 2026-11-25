<script setup>
/**
 * 순수 SVG 꺾은선 그래프 (차트 라이브러리 미사용)
 * - 다중 시리즈, 영역 채움(area), 점 표시(dots)를 지원한다.
 * - y축 범위는 데이터 최소/최대에 여유를 둬서 선이 테두리에 붙지 않게 한다.
 *
 * 사용 예)
 *   <LineChart :labels="labels" :series="[{name:'입차',color:'#2563eb',values:[...]}]" area />
 */
import { computed } from 'vue'

const props = defineProps({
  /** x축 라벨 배열 */
  labels: { type: Array, default: () => [] },
  /** [{name, color, values:[]}] */
  series: { type: Array, default: () => [] },
  width: { type: Number, default: 720 },
  height: { type: Number, default: 240 },
  /** y축 눈금 개수 */
  ticks: { type: Number, default: 4 },
  /** y축 하한 고정(미지정이면 자동) */
  minValue: { type: Number, default: null },
  /** y축 상한 고정(미지정이면 자동) */
  maxValue: { type: Number, default: null },
  /** y축을 항상 0부터 시작 */
  zeroBased: { type: Boolean, default: true },
  /** 선 아래를 옅게 채운다. */
  area: { type: Boolean, default: false },
  /** 데이터 점을 원으로 표시 */
  dots: { type: Boolean, default: false },
  /** x축 라벨을 n개마다 하나씩만 표기(0 이면 전부) */
  labelEvery: { type: Number, default: 0 },
  /** 눈금 라벨 포맷 함수 */
  tickFormat: { type: Function, default: null },
  /** 선 두께 */
  strokeWidth: { type: Number, default: 2 },
  /** 범례 표시 */
  legend: { type: Boolean, default: true },
  /** 단위(툴팁용) */
  unit: { type: String, default: '' },
})

/** 플롯 여백 */
const pad = { top: 16, right: 14, bottom: 28, left: 46 }

const plot = computed(() => ({
  x: pad.left,
  y: pad.top,
  w: Math.max(10, props.width - pad.left - pad.right),
  h: Math.max(10, props.height - pad.top - pad.bottom),
}))

/** 값 배열이 있는 시리즈만 사용한다. */
const validSeries = computed(() =>
  props.series.filter((s) => Array.isArray(s.values) && s.values.length),
)

/**
 * 데이터 최대값을 '보기 좋은 수'로 올림해 눈금을 깔끔하게 만든다.
 * BarChart.vue 와 동일한 단계를 쓴다.
 */
const NICE_STEPS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]

function niceCeil(max) {
  if (max <= 0) return 10
  const exp = Math.floor(Math.log10(max))
  const base = 10 ** exp
  const n = max / base
  const step = NICE_STEPS.find((s) => n <= s + 1e-9) ?? 10
  return step * base
}

/** x축 항목 개수 */
const count = computed(() => {
  const fromSeries = Math.max(0, ...validSeries.value.map((s) => s.values.length))
  return props.labels.length || fromSeries
})

/** 모든 값을 평탄화 */
const allValues = computed(() => {
  const out = []
  validSeries.value.forEach((s) => {
    s.values.forEach((v) => {
      const n = Number(v)
      if (!Number.isNaN(n)) out.push(n)
    })
  })
  return out
})

/**
 * y축 범위 계산.
 * - zeroBased 이면 하한을 0으로 둔다(교통량처럼 0 기준이 자연스러운 지표).
 * - 그렇지 않으면 min/max 에 10% 여유를 둔다(온도/전압처럼 변동 폭이 좁은 지표).
 */
const yRange = computed(() => {
  if (!allValues.value.length) return { min: 0, max: 10 }

  let min = Math.min(...allValues.value)
  let max = Math.max(...allValues.value)

  if (props.minValue !== null) min = props.minValue
  else if (props.zeroBased) min = Math.min(0, min)

  if (props.maxValue !== null) max = props.maxValue

  // 모든 값이 같으면 범위가 0이 되어 나눗셈이 깨지므로 폭을 만든다.
  if (max === min) {
    max = min + (Math.abs(min) * 0.1 || 1)
  }

  if (props.minValue === null && props.maxValue === null && !props.zeroBased) {
    const margin = (max - min) * 0.12
    min -= margin
    max += margin
  } else if (props.maxValue === null) {
    // 0 기준 축이면 눈금이 정수로 나뉘도록 상한을 보기 좋은 수로 올린다.
    max = min === 0 ? niceCeil(max) : max + (max - min) * 0.08
  }

  return { min, max }
})

/** 값 → y 픽셀 좌표 */
function toY(v) {
  const { min, max } = yRange.value
  const p = plot.value
  const ratio = (Number(v) - min) / (max - min)
  return p.y + p.h - Math.min(1, Math.max(0, ratio)) * p.h
}

/** 인덱스 → x 픽셀 좌표 */
function toX(i) {
  const p = plot.value
  if (count.value <= 1) return p.x + p.w / 2
  return p.x + (p.w / (count.value - 1)) * i
}

/**
 * 시리즈별 폴리라인 점 좌표/패스 문자열.
 * area 용 닫힌 패스도 함께 만든다.
 */
const paths = computed(() => {
  const p = plot.value
  return validSeries.value.map((s, si) => {
    const pts = s.values.map((v, i) => ({ x: toX(i), y: toY(v), value: Number(v) || 0, label: props.labels[i] ?? '' }))
    const line = pts.map((pt, i) => `${i === 0 ? 'M' : 'L'}${pt.x.toFixed(2)},${pt.y.toFixed(2)}`).join(' ')
    const areaPath = pts.length
      ? `${line} L${pts[pts.length - 1].x.toFixed(2)},${(p.y + p.h).toFixed(2)} L${pts[0].x.toFixed(2)},${(p.y + p.h).toFixed(2)} Z`
      : ''
    return {
      key: s.name || `s${si}`,
      name: s.name || '',
      color: s.color || 'var(--c-chart-1)',
      line,
      areaPath,
      points: pts,
    }
  })
})

/** y축 눈금선 */
const gridLines = computed(() => {
  const p = plot.value
  const { min, max } = yRange.value
  const out = []
  for (let i = 0; i <= props.ticks; i += 1) {
    const value = min + ((max - min) / props.ticks) * i
    const y = toY(value)
    out.push({ key: `g${i}`, y, value, x1: p.x, x2: p.x + p.w })
  }
  return out
})

/** 눈금 라벨 문자열 */
function tickText(v) {
  if (props.tickFormat) return props.tickFormat(v)
  const abs = Math.abs(v)
  if (abs >= 10000) return `${Math.round(v / 1000)}k`
  if (abs >= 100) return String(Math.round(v))
  // 정수로 떨어지는 눈금은 '0.0' 대신 '0' 으로 표기한다.
  if (Number.isInteger(v)) return String(v)
  if (abs >= 10) return v.toFixed(0)
  return v.toFixed(1)
}

/** x축 라벨 좌표 */
const axisLabels = computed(() => {
  const p = plot.value
  const out = []
  for (let i = 0; i < count.value; i += 1) {
    if (props.labelEvery > 0 && i % props.labelEvery !== 0) continue
    out.push({ key: `a${i}`, x: toX(i), y: p.y + p.h + 18, text: props.labels[i] ?? '' })
  }
  return out
})

/** 범례 항목 */
const legendItems = computed(() => validSeries.value.filter((s) => s.name))

const hasData = computed(() => allValues.value.length > 0)
</script>

<template>
  <div class="line-chart">
    <svg
      class="chart"
      :viewBox="`0 0 ${width} ${height}`"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      :aria-label="`꺾은선그래프. ${legendItems.map((s) => s.name).join(', ')}`"
    >
      <!-- 눈금선 -->
      <g>
        <line
          v-for="g in gridLines"
          :key="g.key"
          class="chart__grid-line"
          :x1="g.x1"
          :y1="g.y"
          :x2="g.x2"
          :y2="g.y"
        />
      </g>

      <!-- y축 눈금 라벨 -->
      <g>
        <text
          v-for="g in gridLines"
          :key="`t${g.key}`"
          :x="plot.x - 8"
          :y="g.y + 3.5"
          text-anchor="end"
        >
          {{ tickText(g.value) }}
        </text>
      </g>

      <!-- 영역 채움 -->
      <g v-if="area">
        <path
          v-for="s in paths"
          :key="`ar-${s.key}`"
          :d="s.areaPath"
          :fill="s.color"
          opacity="0.1"
        />
      </g>

      <!-- 꺾은선 -->
      <g>
        <path
          v-for="s in paths"
          :key="`ln-${s.key}`"
          :d="s.line"
          fill="none"
          :stroke="s.color"
          :stroke-width="strokeWidth"
          stroke-linejoin="round"
          stroke-linecap="round"
        />
      </g>

      <!-- 데이터 점 -->
      <g v-if="dots">
        <template v-for="s in paths" :key="`dt-${s.key}`">
          <circle
            v-for="(pt, i) in s.points"
            :key="`${s.key}-${i}`"
            :cx="pt.x"
            :cy="pt.y"
            r="3"
            :fill="s.color"
            stroke="#fff"
            stroke-width="1.4"
          >
            <title>{{ pt.label }} {{ s.name }} {{ pt.value }}{{ unit }}</title>
          </circle>
        </template>
      </g>

      <!-- x축 -->
      <line
        class="chart__axis-line"
        :x1="plot.x"
        :y1="plot.y + plot.h"
        :x2="plot.x + plot.w"
        :y2="plot.y + plot.h"
      />

      <!-- x축 라벨 -->
      <g>
        <text
          v-for="a in axisLabels"
          :key="a.key"
          :x="a.x"
          :y="a.y"
          text-anchor="middle"
        >
          {{ a.text }}
        </text>
      </g>

      <!-- 데이터 없음 -->
      <text
        v-if="!hasData"
        :x="width / 2"
        :y="height / 2"
        text-anchor="middle"
        class="line-chart__nodata"
      >
        표시할 데이터가 없습니다
      </text>
    </svg>

    <!-- 범례 -->
    <div v-if="legend && legendItems.length" class="chart-legend">
      <span v-for="s in legendItems" :key="s.name" class="chart-legend__item">
        <span class="chart-legend__swatch" :style="{ background: s.color }" />
        {{ s.name }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.line-chart {
  width: 100%;
}

.line-chart .chart {
  height: auto;
}

.line-chart__nodata {
  font-size: 13px;
  fill: var(--c-text-muted);
}
</style>
