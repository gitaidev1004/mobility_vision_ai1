<script setup>
/**
 * 순수 SVG 도넛 차트 (차트 라이브러리 미사용)
 * - stroke-dasharray 기법 대신 arc 패스를 직접 계산해 조각을 그린다.
 *   (조각 사이 간격을 정확히 제어할 수 있어 교재 설명에 적합하다.)
 * - 중앙에 총합/라벨을 표시하고, 우측(또는 하단)에 범례를 둔다.
 *
 * 사용 예)
 *   <DonutChart :slices="[{name:'낮음',value:100,color:'#16a34a'}]" center-label="차량" />
 */
import { computed } from 'vue'

const props = defineProps({
  /** [{name, value, color}] */
  slices: { type: Array, default: () => [] },
  /** SVG 한 변 크기(정사각형) */
  size: { type: Number, default: 220 },
  /** 도넛 두께(px, viewBox 기준) */
  thickness: { type: Number, default: 26 },
  /** 조각 사이 간격(도) */
  gap: { type: Number, default: 1.6 },
  /** 중앙 큰 숫자. 미지정이면 전체 합계를 쓴다. */
  centerValue: { type: [Number, String], default: null },
  /** 중앙 숫자 아래 설명 */
  centerLabel: { type: String, default: '' },
  /** 범례 표시 */
  legend: { type: Boolean, default: true },
  /** 범례에 값/비율 함께 표기 */
  legendValues: { type: Boolean, default: true },
})

/** 유효한(0 이상) 조각만 사용 */
const validSlices = computed(() =>
  props.slices
    .map((s) => ({ ...s, value: Number(s.value) || 0 }))
    .filter((s) => s.value >= 0),
)

/** 전체 합계 */
const total = computed(() => validSlices.value.reduce((a, s) => a + s.value, 0))

/** 중심 좌표 */
const cx = computed(() => props.size / 2)
const cy = computed(() => props.size / 2)
/** 바깥 반지름(여백 2px) */
const rOuter = computed(() => props.size / 2 - 2)
/** 안쪽 반지름 */
const rInner = computed(() => Math.max(4, rOuter.value - props.thickness))

/** 각도(도) → 좌표. 12시 방향에서 시작해 시계방향으로 진행한다. */
function polar(angleDeg, radius) {
  const rad = ((angleDeg - 90) * Math.PI) / 180
  return {
    x: cx.value + radius * Math.cos(rad),
    y: cy.value + radius * Math.sin(rad),
  }
}

/**
 * 도넛 조각 arc 패스를 만든다.
 * 바깥 호 → 안쪽 호 역방향 → 닫기 순서로 그린다.
 */
function arcPath(startDeg, endDeg) {
  const sweep = endDeg - startDeg
  // 360도에 가까우면 하나의 arc 로는 그릴 수 없으므로 두 개로 쪼갠다.
  const largeArc = sweep > 180 ? 1 : 0

  const o1 = polar(startDeg, rOuter.value)
  const o2 = polar(endDeg, rOuter.value)
  const i2 = polar(endDeg, rInner.value)
  const i1 = polar(startDeg, rInner.value)

  return [
    `M${o1.x.toFixed(2)},${o1.y.toFixed(2)}`,
    `A${rOuter.value},${rOuter.value} 0 ${largeArc} 1 ${o2.x.toFixed(2)},${o2.y.toFixed(2)}`,
    `L${i2.x.toFixed(2)},${i2.y.toFixed(2)}`,
    `A${rInner.value},${rInner.value} 0 ${largeArc} 0 ${i1.x.toFixed(2)},${i1.y.toFixed(2)}`,
    'Z',
  ].join(' ')
}

/**
 * 조각별 패스 + 비율을 계산한다.
 * 값이 0인 조각은 그리지 않되 범례에는 남긴다.
 */
const arcs = computed(() => {
  if (total.value <= 0) return []

  const drawable = validSlices.value.filter((s) => s.value > 0)
  // 조각이 하나뿐이면 간격 없이 완전한 고리를 그린다.
  const gap = drawable.length > 1 ? props.gap : 0

  let cursor = 0
  return drawable.map((s, i) => {
    const sweep = (s.value / total.value) * 360
    const start = cursor + gap / 2
    const end = cursor + sweep - gap / 2
    cursor += sweep
    return {
      key: `${s.name || i}`,
      name: s.name || '',
      value: s.value,
      color: s.color || 'var(--c-chart-1)',
      ratio: s.value / total.value,
      // 간격 때문에 end < start 가 되는 아주 작은 조각은 최소 폭을 보장한다.
      d: arcPath(start, Math.max(start + 0.6, end)),
    }
  })
})

/** 조각이 하나도 없을 때 보여줄 회색 고리 */
const emptyRing = computed(() => arcPath(0, 359.99))

/** 중앙 큰 숫자 */
const centerText = computed(() => {
  if (props.centerValue !== null) return props.centerValue
  return total.value.toLocaleString('ko-KR')
})

/** 범례 항목(0 값 포함) */
const legendItems = computed(() =>
  validSlices.value.map((s) => ({
    ...s,
    percent: total.value > 0 ? ((s.value / total.value) * 100).toFixed(1) : '0.0',
  })),
)
</script>

<template>
  <div class="donut-chart">
    <div class="donut-chart__figure">
      <svg
        class="donut-chart__svg"
        :viewBox="`0 0 ${size} ${size}`"
        role="img"
        :aria-label="`도넛 차트. 총 ${total}건`"
      >
        <!-- 데이터가 없을 때의 빈 고리 -->
        <path v-if="!arcs.length" :d="emptyRing" fill="#eef1f6" />

        <!-- 조각 -->
        <path v-for="a in arcs" :key="a.key" :d="a.d" :fill="a.color">
          <title>{{ a.name }} {{ a.value }}건 ({{ (a.ratio * 100).toFixed(1) }}%)</title>
        </path>

        <!-- 중앙 텍스트 -->
        <text
          class="donut-chart__center-value"
          :x="cx"
          :y="cy - 2"
          text-anchor="middle"
          dominant-baseline="middle"
        >
          {{ centerText }}
        </text>
        <text
          v-if="centerLabel"
          class="donut-chart__center-label"
          :x="cx"
          :y="cy + 20"
          text-anchor="middle"
        >
          {{ centerLabel }}
        </text>
      </svg>
    </div>

    <!-- 범례: 색 / 이름 / 값 / 비율 -->
    <ul v-if="legend" class="donut-chart__legend">
      <li v-for="s in legendItems" :key="s.name" class="donut-chart__legend-item">
        <span class="chart-legend__swatch" :style="{ background: s.color }" />
        <span class="donut-chart__legend-name">{{ s.name }}</span>
        <span v-if="legendValues" class="donut-chart__legend-value num">
          {{ s.value.toLocaleString('ko-KR') }}
        </span>
        <span v-if="legendValues" class="donut-chart__legend-pct num">{{ s.percent }}%</span>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.donut-chart {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--sp-4);
  width: 100%;
}

.donut-chart__figure {
  width: 100%;
  max-width: 220px;
}

.donut-chart__svg {
  width: 100%;
  height: auto;
  display: block;
}

.donut-chart__center-value {
  font-family: var(--font-sans);
  font-size: 30px;
  font-weight: 800;
  fill: var(--c-text);
  letter-spacing: -0.03em;
}

.donut-chart__center-label {
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 600;
  fill: var(--c-text-muted);
}

.donut-chart__legend {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.donut-chart__legend-item {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  font-size: var(--fs-base);
}

.donut-chart__legend-name {
  color: var(--c-text-sub);
  font-weight: 600;
}

.donut-chart__legend-value {
  margin-left: auto;
  font-weight: 700;
}

.donut-chart__legend-pct {
  width: 48px;
  text-align: right;
  color: var(--c-text-muted);
  font-weight: 600;
}
</style>
