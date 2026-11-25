<script setup>
/**
 * 순수 SVG 게이지(반원 계기) 차트 (차트 라이브러리 미사용)
 * - 예지보전 '건강 점수(0~100)' 표시에 사용한다.
 * - 배경 호 + 값 호 + 구간 눈금 + 바늘을 직접 계산해 그린다.
 * - 각도 범위는 -210도 ~ +30도(=240도 스윕)로, 자동차 계기판 느낌을 낸다.
 */
import { computed } from 'vue'

const props = defineProps({
  /** 표시 값 */
  value: { type: Number, default: 0 },
  /** 최소/최대 */
  min: { type: Number, default: 0 },
  max: { type: Number, default: 100 },
  /** SVG viewBox 폭 */
  width: { type: Number, default: 260 },
  /** 호 두께 */
  thickness: { type: Number, default: 20 },
  /** 값 색상 톤: success | warn | danger | primary */
  tone: { type: String, default: 'primary' },
  /** 중앙 숫자 아래 설명 */
  label: { type: String, default: '' },
  /** 값 단위(점/% 등) */
  unit: { type: String, default: '' },
  /** 눈금 표시 */
  showTicks: { type: Boolean, default: true },
  /** 눈금 개수 */
  tickCount: { type: Number, default: 5 },
  /** 바늘 표시 */
  showNeedle: { type: Boolean, default: true },
})

/** 게이지 스윕: 시작 -210도, 끝 30도 (총 240도) */
const START = -210
const SWEEP = 240

/** 중심/반지름. 240도 스윕이므로 높이는 폭의 약 0.66 배면 충분하다. */
const cx = computed(() => props.width / 2)
const r = computed(() => props.width / 2 - props.thickness / 2 - 6)
const cy = computed(() => props.thickness / 2 + 6 + r.value)
/** viewBox 높이: 호의 아래쪽 끝(30도)까지 포함해야 잘리지 않는다. */
const height = computed(() => cy.value + r.value * Math.sin((30 * Math.PI) / 180) + props.thickness / 2 + 30)

/** 0~1 로 정규화된 값 */
const ratio = computed(() => {
  const span = props.max - props.min
  if (span <= 0) return 0
  return Math.min(1, Math.max(0, (Number(props.value) - props.min) / span))
})

/** 톤별 색상 */
const COLOR = {
  success: 'var(--c-success)',
  warn: 'var(--c-warn)',
  danger: 'var(--c-danger)',
  primary: 'var(--c-primary)',
  info: 'var(--c-info)',
}

const valueColor = computed(() => COLOR[props.tone] || COLOR.primary)

/** 각도(도) → 좌표 */
function polar(deg, radius) {
  const rad = (deg * Math.PI) / 180
  return {
    x: cx.value + radius * Math.cos(rad),
    y: cy.value + radius * Math.sin(rad),
  }
}

/** 열린 호 패스(stroke 로 그린다). */
function arc(fromRatio, toRatio) {
  const a1 = START + SWEEP * fromRatio
  const a2 = START + SWEEP * toRatio
  const p1 = polar(a1, r.value)
  const p2 = polar(a2, r.value)
  const largeArc = Math.abs(a2 - a1) > 180 ? 1 : 0
  return `M${p1.x.toFixed(2)},${p1.y.toFixed(2)} A${r.value},${r.value} 0 ${largeArc} 1 ${p2.x.toFixed(2)},${p2.y.toFixed(2)}`
}

/** 배경(전체) 호 */
const trackPath = computed(() => arc(0, 1))

/** 값 호. 값이 0 이면 그리지 않는다. */
const valuePath = computed(() => (ratio.value > 0 ? arc(0, ratio.value) : ''))

/**
 * 눈금 선/라벨 좌표.
 * 반지름 배치: 호 안쪽 끝(r-thickness/2) → 눈금선 → 눈금 라벨 → 바늘 끝.
 * 바늘이 라벨 위를 지나가지 않도록 각 구간을 충분히 벌린다.
 */
const ticks = computed(() => {
  if (!props.showTicks) return []
  const out = []
  for (let i = 0; i <= props.tickCount; i += 1) {
    const t = i / props.tickCount
    const deg = START + SWEEP * t
    const inner = polar(deg, r.value - props.thickness / 2 - 1)
    const outer = polar(deg, r.value - props.thickness / 2 - 7)
    const textPos = polar(deg, r.value - props.thickness / 2 - 26)
    out.push({
      key: `tk${i}`,
      x1: inner.x,
      y1: inner.y,
      x2: outer.x,
      y2: outer.y,
      tx: textPos.x,
      ty: textPos.y + 3.5,
      text: String(Math.round(props.min + (props.max - props.min) * t)),
    })
  }
  return out
})

/** 바늘 끝 좌표. 눈금 라벨(r-thickness/2-26)보다 안쪽에서 멈춰 겹치지 않게 한다. */
const needle = computed(() => {
  const deg = START + SWEEP * ratio.value
  const tip = polar(deg, Math.max(14, r.value - props.thickness / 2 - 40))
  // 꼬리를 중심 반대쪽으로 약간 뺀다.
  const tail = polar(deg + 180, 12)
  return { tip, tail }
})

/** 표시 숫자(소수 1자리, 정수면 정수로) */
const displayValue = computed(() => {
  const v = Number(props.value)
  if (Number.isNaN(v)) return '-'
  return Number.isInteger(v) ? String(v) : v.toFixed(1)
})
</script>

<template>
  <div class="gauge-chart">
    <svg
      class="gauge-chart__svg"
      :viewBox="`0 0 ${width} ${height}`"
      role="img"
      :aria-label="`게이지. ${label} ${displayValue}${unit}`"
    >
      <!-- 배경 호 -->
      <path
        :d="trackPath"
        fill="none"
        stroke="#eef1f6"
        :stroke-width="thickness"
        stroke-linecap="round"
      />

      <!-- 값 호 -->
      <path
        v-if="valuePath"
        :d="valuePath"
        fill="none"
        :stroke="valueColor"
        :stroke-width="thickness"
        stroke-linecap="round"
      />

      <!-- 눈금 -->
      <g>
        <line
          v-for="t in ticks"
          :key="t.key"
          :x1="t.x1"
          :y1="t.y1"
          :x2="t.x2"
          :y2="t.y2"
          stroke="var(--c-border-strong)"
          stroke-width="1.4"
          stroke-linecap="round"
        />
      </g>
      <g>
        <text
          v-for="t in ticks"
          :key="`txt-${t.key}`"
          class="gauge-chart__tick-text"
          :x="t.tx"
          :y="t.ty"
          text-anchor="middle"
        >
          {{ t.text }}
        </text>
      </g>

      <!-- 바늘 -->
      <g v-if="showNeedle">
        <line
          :x1="needle.tail.x"
          :y1="needle.tail.y"
          :x2="needle.tip.x"
          :y2="needle.tip.y"
          :stroke="valueColor"
          stroke-width="3"
          stroke-linecap="round"
        />
        <circle :cx="cx" :cy="cy" r="6" :fill="valueColor" />
        <circle :cx="cx" :cy="cy" r="2.4" fill="#fff" />
      </g>

      <!-- 중앙 값 -->
      <text
        class="gauge-chart__value"
        :x="cx"
        :y="cy + 46"
        text-anchor="middle"
      >
        {{ displayValue }}<tspan class="gauge-chart__unit">{{ unit }}</tspan>
      </text>
      <text
        v-if="label"
        class="gauge-chart__label"
        :x="cx"
        :y="cy + 66"
        text-anchor="middle"
      >
        {{ label }}
      </text>
    </svg>
  </div>
</template>

<style scoped>
.gauge-chart {
  width: 100%;
  display: flex;
  justify-content: center;
}

.gauge-chart__svg {
  width: 100%;
  max-width: 260px;
  height: auto;
  display: block;
}

.gauge-chart__value {
  font-family: var(--font-sans);
  font-size: 34px;
  font-weight: 800;
  fill: var(--c-text);
  letter-spacing: -0.03em;
}

.gauge-chart__unit {
  font-size: 15px;
  font-weight: 700;
  fill: var(--c-text-sub);
}

.gauge-chart__label {
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 600;
  fill: var(--c-text-muted);
}

.gauge-chart__tick-text {
  font-family: var(--font-sans);
  font-size: 10px;
  font-weight: 600;
  fill: var(--c-text-muted);
}
</style>
