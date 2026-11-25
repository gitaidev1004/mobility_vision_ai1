<script setup>
/**
 * 순수 SVG 막대그래프 (차트 라이브러리 미사용)
 * - 단일 시리즈 또는 다중 시리즈(그룹 막대)를 모두 지원한다.
 * - 좌표 계산은 모두 computed 안에서 수행하고, 템플릿은 계산 결과만 그린다.
 * - viewBox + preserveAspectRatio 로 컨테이너 폭에 맞춰 늘어난다.
 *
 * 사용 예)
 *   <BarChart :labels="['9시','10시']" :series="[{name:'입차',color:'#2563eb',values:[12,18]}]" />
 */
import { computed } from 'vue'

const props = defineProps({
  /** x축 라벨 배열 */
  labels: { type: Array, default: () => [] },
  /** [{name, color, values:[]}] 형태의 시리즈 배열 */
  series: { type: Array, default: () => [] },
  /** SVG 내부 좌표계 폭 */
  width: { type: Number, default: 720 },
  /** SVG 내부 좌표계 높이 */
  height: { type: Number, default: 240 },
  /** y축 눈금 개수 */
  ticks: { type: Number, default: 4 },
  /** y축 최대값 고정(미지정이면 데이터 최대값 기준 자동) */
  maxValue: { type: Number, default: null },
  /** 가로 막대 모드 */
  horizontal: { type: Boolean, default: false },
  /** 막대 위에 값을 표기 */
  showValues: { type: Boolean, default: false },
  /** 값 포맷 함수 */
  valueFormat: { type: Function, default: (v) => String(v) },
  /** y축 단위 라벨(예: '건') */
  unit: { type: String, default: '' },
  /** x축 라벨을 n개마다 하나씩만 표기(0 이면 전부) */
  labelEvery: { type: Number, default: 0 },
  /** 범례 표시 */
  legend: { type: Boolean, default: true },
  /**
   * 가로 막대 모드에서 좌측 라벨이 차지할 폭(viewBox 단위).
   * 항목명이 길면 이 값을 키워야 글자가 잘리지 않는다.
   */
  labelWidth: { type: Number, default: 108 },
})

/** 플롯 영역 여백. 가로 막대 모드는 라벨 공간을 더 확보한다. */
const pad = computed(() =>
  props.horizontal
    ? { top: 10, right: 46, bottom: 26, left: props.labelWidth }
    : { top: 16, right: 10, bottom: 30, left: 44 },
)

/** 플롯 영역(축 안쪽) 크기 */
const plot = computed(() => ({
  x: pad.value.left,
  y: pad.value.top,
  w: Math.max(10, props.width - pad.value.left - pad.value.right),
  h: Math.max(10, props.height - pad.value.top - pad.value.bottom),
}))

/** 유효한 시리즈만 추출(값 배열이 있는 것) */
const validSeries = computed(() =>
  props.series.filter((s) => Array.isArray(s.values) && s.values.length),
)

/** x축 항목 개수 */
const count = computed(() => {
  if (props.labels.length) return props.labels.length
  return validSeries.value[0]?.values.length ?? 0
})

/**
 * y축 최대값.
 * - maxValue prop 이 있으면 그 값을 쓴다.
 * - 없으면 데이터 최대값을 '보기 좋은 수'로 올림한다(눈금이 깔끔해진다).
 */
const yMax = computed(() => {
  if (props.maxValue !== null) return props.maxValue
  let max = 0
  validSeries.value.forEach((s) => {
    s.values.forEach((v) => {
      const n = Number(v) || 0
      if (n > max) max = n
    })
  })
  if (max <= 0) return 10
  return niceCeil(max)
})

/**
 * 데이터 최대값을 '보기 좋은 수'로 올림한다.
 * 1·1.5·2·2.5·3·4·5·6·8·10 단계를 써서 축 상단이 데이터에서 너무 멀어지지 않게 한다.
 * (예: 51 → 60. 1·2·5 단계만 쓰면 100 이 되어 그래프가 납작해진다.)
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

/** y축(가로 막대면 x축) 눈금 값 배열 */
const tickValues = computed(() => {
  const out = []
  for (let i = 0; i <= props.ticks; i += 1) {
    out.push((yMax.value / props.ticks) * i)
  }
  return out
})

/** 눈금 라벨 포맷: 1000 이상은 'k' 로 줄이고, 정수는 소수점을 붙이지 않는다. */
function tickLabel(v) {
  if (v >= 10000) return `${Math.round(v / 1000)}k`
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k`
  const r = Math.round(v * 10) / 10
  return Number.isInteger(r) ? String(r) : r.toFixed(1)
}

/** 세로 막대 기준: 한 그룹(= x축 한 칸)의 폭 */
const groupWidth = computed(() => (count.value ? plot.value.w / count.value : plot.value.w))

/** 그룹 안쪽 여백을 뺀 실제 막대 폭 */
const barWidth = computed(() => {
  const n = Math.max(1, validSeries.value.length)
  // 그룹 폭의 68% 를 막대들이 나눠 쓰고, 나머지는 간격으로 둔다.
  return (groupWidth.value * 0.68) / n
})

/** 가로 막대 기준: 한 행 높이 */
const rowHeight = computed(() => (count.value ? plot.value.h / count.value : plot.value.h))

/** 가로 막대 높이 */
const barHeight = computed(() => {
  const n = Math.max(1, validSeries.value.length)
  return (rowHeight.value * 0.62) / n
})

/**
 * 그릴 막대 목록을 하나의 배열로 평탄화한다.
 * 각 항목: {x, y, w, h, color, value, label, seriesName}
 */
const bars = computed(() => {
  const out = []
  const p = plot.value
  const nSeries = Math.max(1, validSeries.value.length)

  validSeries.value.forEach((s, si) => {
    s.values.forEach((raw, i) => {
      const v = Number(raw) || 0
      const ratio = yMax.value > 0 ? Math.min(1, Math.max(0, v / yMax.value)) : 0

      if (props.horizontal) {
        const w = ratio * p.w
        const rowTop = p.y + rowHeight.value * i
        const inset = (rowHeight.value - barHeight.value * nSeries) / 2
        out.push({
          key: `${si}-${i}`,
          x: p.x,
          y: rowTop + inset + barHeight.value * si,
          w: Math.max(v > 0 ? 2 : 0, w),
          h: barHeight.value,
          color: s.color || 'var(--c-chart-1)',
          value: v,
          label: props.labels[i] ?? '',
          seriesName: s.name || '',
          // 값 라벨 좌표(막대 오른쪽)
          tx: p.x + Math.max(v > 0 ? 2 : 0, w) + 6,
          ty: rowTop + inset + barHeight.value * si + barHeight.value / 2 + 3.5,
          anchor: 'start',
        })
      } else {
        const h = ratio * p.h
        const groupLeft = p.x + groupWidth.value * i
        const inset = (groupWidth.value - barWidth.value * nSeries) / 2
        const x = groupLeft + inset + barWidth.value * si
        out.push({
          key: `${si}-${i}`,
          x,
          y: p.y + p.h - h,
          w: barWidth.value,
          h: Math.max(v > 0 ? 2 : 0, h),
          color: s.color || 'var(--c-chart-1)',
          value: v,
          label: props.labels[i] ?? '',
          seriesName: s.name || '',
          // 값 라벨 좌표(막대 위)
          tx: x + barWidth.value / 2,
          ty: p.y + p.h - h - 5,
          anchor: 'middle',
        })
      }
    })
  })
  return out
})

/** 눈금선 + 눈금 라벨 좌표 */
const gridLines = computed(() => {
  const p = plot.value
  return tickValues.value.map((v) => {
    const ratio = yMax.value > 0 ? v / yMax.value : 0
    if (props.horizontal) {
      const x = p.x + ratio * p.w
      return { key: `g${v}`, x1: x, y1: p.y, x2: x, y2: p.y + p.h, tx: x, ty: p.y + p.h + 16, value: v, anchor: 'middle' }
    }
    const y = p.y + p.h - ratio * p.h
    return { key: `g${v}`, x1: p.x, y1: y, x2: p.x + p.w, y2: y, tx: p.x - 8, ty: y + 3.5, value: v, anchor: 'end' }
  })
})

/** x축(가로 막대면 y축) 항목 라벨 좌표 */
const axisLabels = computed(() => {
  const p = plot.value
  const out = []
  for (let i = 0; i < count.value; i += 1) {
    // labelEvery 가 있으면 해당 간격만 표기한다(24시간 축 등).
    if (props.labelEvery > 0 && i % props.labelEvery !== 0) continue
    if (props.horizontal) {
      out.push({
        key: `a${i}`,
        x: p.x - 10,
        y: p.y + rowHeight.value * i + rowHeight.value / 2 + 3.5,
        text: props.labels[i] ?? '',
        anchor: 'end',
      })
    } else {
      out.push({
        key: `a${i}`,
        x: p.x + groupWidth.value * i + groupWidth.value / 2,
        y: p.y + p.h + 18,
        text: props.labels[i] ?? '',
        anchor: 'middle',
      })
    }
  }
  return out
})

/** 기준선(0축) 좌표 */
const baseline = computed(() => {
  const p = plot.value
  return props.horizontal
    ? { x1: p.x, y1: p.y, x2: p.x, y2: p.y + p.h }
    : { x1: p.x, y1: p.y + p.h, x2: p.x + p.w, y2: p.y + p.h }
})

/** 범례에 쓸 시리즈 목록(이름이 있는 것만) */
const legendItems = computed(() => validSeries.value.filter((s) => s.name))

/** 데이터가 전혀 없으면 안내 문구를 띄운다. */
const hasData = computed(() => bars.value.length > 0)
</script>

<template>
  <div class="bar-chart">
    <svg
      class="chart"
      :viewBox="`0 0 ${width} ${height}`"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      :aria-label="`막대그래프. ${legendItems.map((s) => s.name).join(', ')}`"
    >
      <!-- 눈금선 -->
      <g>
        <line
          v-for="g in gridLines"
          :key="g.key"
          class="chart__grid-line"
          :x1="g.x1"
          :y1="g.y1"
          :x2="g.x2"
          :y2="g.y2"
        />
      </g>

      <!-- 눈금 라벨 -->
      <g>
        <text
          v-for="g in gridLines"
          :key="`t${g.key}`"
          :x="g.tx"
          :y="g.ty"
          :text-anchor="g.anchor"
        >
          {{ tickLabel(g.value) }}
        </text>
      </g>

      <!-- 0 기준선 -->
      <line
        class="chart__axis-line"
        :x1="baseline.x1"
        :y1="baseline.y1"
        :x2="baseline.x2"
        :y2="baseline.y2"
      />

      <!-- 막대 -->
      <g>
        <rect
          v-for="b in bars"
          :key="b.key"
          class="chart__bar"
          :x="b.x"
          :y="b.y"
          :width="b.w"
          :height="b.h"
          :fill="b.color"
          rx="3"
        >
          <title>{{ b.label }} {{ b.seriesName }} {{ valueFormat(b.value) }}{{ unit }}</title>
        </rect>
      </g>

      <!-- 값 라벨 -->
      <g v-if="showValues">
        <text
          v-for="b in bars"
          :key="`v${b.key}`"
          class="bar-chart__value"
          :x="b.tx"
          :y="b.ty"
          :text-anchor="b.anchor"
        >
          {{ valueFormat(b.value) }}
        </text>
      </g>

      <!-- 항목 라벨 -->
      <g>
        <text
          v-for="a in axisLabels"
          :key="a.key"
          :x="a.x"
          :y="a.y"
          :text-anchor="a.anchor"
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
        class="bar-chart__nodata"
      >
        표시할 데이터가 없습니다
      </text>
    </svg>

    <!-- 범례 -->
    <div v-if="legend && legendItems.length > 1" class="chart-legend">
      <span v-for="s in legendItems" :key="s.name" class="chart-legend__item">
        <span class="chart-legend__swatch" :style="{ background: s.color }" />
        {{ s.name }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.bar-chart {
  width: 100%;
}

/* viewBox 비율을 유지하면서 컨테이너 폭에 맞춰 늘어난다. */
.bar-chart .chart {
  height: auto;
}

.bar-chart__value {
  font-size: 10.5px;
  font-weight: 700;
  fill: var(--c-text-sub);
}

.bar-chart__nodata {
  font-size: 13px;
  fill: var(--c-text-muted);
}
</style>
