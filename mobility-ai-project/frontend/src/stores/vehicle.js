/**
 * 차량 스토어 (계약서 5장)
 * - 검색어/페이지 상태를 보관하고 목록/등록/수정/삭제를 담당한다.
 * - 백엔드 페이지 응답 {content,totalElements,totalPages,page,size} 를 그대로 분해해서 쓴다.
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as vehicleApi from '@/api/vehicle'
import { extractMessage } from '@/api/http'

export const useVehicleStore = defineStore('vehicle', () => {
  // ---------------- 상태 ----------------
  const items = ref([])
  const totalElements = ref(0)
  const totalPages = ref(0)
  const page = ref(0)
  const size = ref(20)
  /** 검색어(번호판/차주명 부분 일치) */
  const query = ref('')
  const loading = ref(false)
  const saving = ref(false)
  const error = ref('')

  // ---------------- 계산 속성 ----------------
  /** 조회 결과가 비었는지 */
  const isEmpty = computed(() => !loading.value && items.value.length === 0)
  /** 현재 페이지의 첫 행 번호(1-base, 표의 '번호' 열에 사용) */
  const rowOffset = computed(() => page.value * size.value)
  /** 정기권 차량 수 */
  const passCount = computed(
    () => items.value.filter((v) => v.passType && v.passType !== 'NONE').length,
  )

  // ---------------- 액션 ----------------
  /**
   * 목록 조회. 인자를 주면 상태를 먼저 갱신한 뒤 조회한다.
   * @param {{q?:string, page?:number, size?:number}} params
   */
  async function fetchList(params = {}) {
    if (params.q !== undefined) query.value = params.q
    if (params.page !== undefined) page.value = params.page
    if (params.size !== undefined) size.value = params.size

    loading.value = true
    error.value = ''
    try {
      const data = await vehicleApi.listVehicles({
        q: query.value,
        page: page.value,
        size: size.value,
      })
      items.value = data?.content ?? []
      totalElements.value = data?.totalElements ?? 0
      totalPages.value = data?.totalPages ?? 0
      // 서버가 보정한 page/size 를 신뢰한다.
      page.value = data?.page ?? page.value
      size.value = data?.size ?? size.value
    } catch (e) {
      error.value = extractMessage(e)
      items.value = []
      totalElements.value = 0
      totalPages.value = 0
    } finally {
      loading.value = false
    }
  }

  /** 검색 실행(페이지를 0으로 되돌린다). */
  function search(q) {
    return fetchList({ q, page: 0 })
  }

  /** 페이지 이동. */
  function goPage(p) {
    return fetchList({ page: p })
  }

  /** 등록. 성공 시 목록을 새로 고친다. */
  async function create(payload) {
    saving.value = true
    error.value = ''
    try {
      const created = await vehicleApi.createVehicle(payload)
      await fetchList({ page: 0 })
      return created
    } catch (e) {
      error.value = extractMessage(e)
      throw e
    } finally {
      saving.value = false
    }
  }

  /** 수정. 현재 페이지를 유지한 채 새로 고친다. */
  async function update(id, payload) {
    saving.value = true
    error.value = ''
    try {
      const updated = await vehicleApi.updateVehicle(id, payload)
      await fetchList()
      return updated
    } catch (e) {
      error.value = extractMessage(e)
      throw e
    } finally {
      saving.value = false
    }
  }

  /** 삭제. 마지막 행을 지운 경우 이전 페이지로 이동한다. */
  async function remove(id) {
    saving.value = true
    error.value = ''
    try {
      await vehicleApi.deleteVehicle(id)
      const lastRowOnPage = items.value.length === 1 && page.value > 0
      await fetchList({ page: lastRowOnPage ? page.value - 1 : page.value })
    } catch (e) {
      error.value = extractMessage(e)
      throw e
    } finally {
      saving.value = false
    }
  }

  /** 번호판 단건 조회(인식 결과에서 차량 상세로 이동할 때 사용). */
  function findByPlate(plate) {
    return vehicleApi.getVehicleByPlate(plate)
  }

  /** 스토어 초기화. */
  function reset() {
    items.value = []
    totalElements.value = 0
    totalPages.value = 0
    page.value = 0
    query.value = ''
    error.value = ''
  }

  return {
    items,
    totalElements,
    totalPages,
    page,
    size,
    query,
    loading,
    saving,
    error,
    isEmpty,
    rowOffset,
    passCount,
    fetchList,
    search,
    goPage,
    create,
    update,
    remove,
    findByPlate,
    reset,
  }
})
