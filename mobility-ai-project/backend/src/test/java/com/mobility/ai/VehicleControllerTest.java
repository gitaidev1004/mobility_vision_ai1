package com.mobility.ai;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.mobility.ai.common.BusinessException;
import com.mobility.ai.common.GlobalExceptionHandler;
import com.mobility.ai.common.NotFoundException;
import com.mobility.ai.common.PageResponse;
import com.mobility.ai.dto.VehicleDtos.VehicleRequest;
import com.mobility.ai.dto.VehicleDtos.VehicleResponse;
import com.mobility.ai.service.VehicleService;
import com.mobility.ai.web.VehicleController;
import java.time.LocalDateTime;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

/**
 * {@link VehicleController} 웹 계층 테스트 — <b>경로 · HTTP 메서드 · 응답 JSON 필드명</b>이
 * 계약서 4장과 1:1 인지 DB 없이 검증한다.
 *
 * <h2>왜 {@code @WebMvcTest} 가 아니라 standalone 설정인가</h2>
 * Spring Boot 4 는 자동 구성과 테스트 슬라이스를 기술별 모듈로 분리했고, 그 과정에서
 * {@code @WebMvcTest} 등 슬라이스 애너테이션의 패키지가 이동했다. 교재 부록은
 * 어느 마이너 버전에서도 그대로 컴파일돼야 하므로, 여기서는 Spring Framework 의
 * {@link MockMvcBuilders#standaloneSetup(Object...)} 를 쓴다.
 *
 * <p>얻는 것은 같다 — 컨트롤러와 {@link GlobalExceptionHandler} 만 올리고,
 * 서비스는 Mockito 목으로 대체하며, Jackson 기반 JSON 직렬화와 Bean Validation 은
 * 실제로 동작한다. 스프링 컨텍스트·DataSource·보안 필터는 뜨지 않아 매우 빠르다.
 *
 * <p>프로젝트에서 슬라이스 애너테이션을 쓰고 싶다면 이 클래스를 다음으로 바꾸면 된다
 * (해당 버전의 패키지 경로와 {@code spring-boot-webmvc-test} 의존성 확인 필요):
 * <pre>
 * &#64;WebMvcTest(controllers = VehicleController.class)
 * &#64;AutoConfigureMockMvc(addFilters = false)
 * class VehicleControllerTest {
 *     &#64;Autowired MockMvc mockMvc;
 *     &#64;MockitoBean VehicleService vehicleService;
 *     // ... 아래 테스트 메서드 그대로 재사용
 * }
 * </pre>
 */
@DisplayName("VehicleController 웹 계층")
class VehicleControllerTest {

    private MockMvc mockMvc;

    private VehicleService vehicleService;

    /**
     * 컨트롤러 + 전역 예외 처리기 + Bean Validation 으로 MockMvc 를 구성한다.
     */
    @BeforeEach
    void setUp() {
        this.vehicleService = mock(VehicleService.class);

        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();

        this.mockMvc = MockMvcBuilders
                .standaloneSetup(new VehicleController(vehicleService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    /**
     * 테스트용 차량 응답을 만든다.
     *
     * @return id=1 인 차량 응답
     */
    private VehicleResponse sampleVehicle() {
        return new VehicleResponse(
                1L,
                "12가3456",
                "홍길동",
                "010-1111-2222",
                "SEDAN",
                "아반떼 CN7",
                2021,
                "KMHLM41AAMU000111",
                "REGULAR",
                LocalDateTime.of(2027, 1, 1, 0, 0),
                true,
                LocalDateTime.of(2026, 1, 1, 9, 0)
        );
    }

    @Test
    @DisplayName("GET /api/v1/vehicles 는 계약서 페이지 포맷으로 응답한다")
    void listReturnsContractPageShape() throws Exception {
        PageResponse<VehicleResponse> page =
                new PageResponse<>(List.of(sampleVehicle()), 1L, 1, 0, 20);
        when(vehicleService.search(any(), anyInt(), anyInt())).thenReturn(page);

        mockMvc.perform(get("/api/v1/vehicles")
                        .param("q", "홍길동")
                        .param("page", "0")
                        .param("size", "20"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.content").isArray())
                .andExpect(jsonPath("$.content[0].id").value(1))
                .andExpect(jsonPath("$.content[0].plateNumber").value("12가3456"))
                .andExpect(jsonPath("$.content[0].ownerName").value("홍길동"))
                .andExpect(jsonPath("$.content[0].passType").value("REGULAR"))
                .andExpect(jsonPath("$.content[0].active").value(true))
                .andExpect(jsonPath("$.totalElements").value(1))
                .andExpect(jsonPath("$.totalPages").value(1))
                .andExpect(jsonPath("$.page").value(0))
                .andExpect(jsonPath("$.size").value(20));

        verify(vehicleService).search("홍길동", 0, 20);
    }

    @Test
    @DisplayName("GET /api/v1/vehicles 는 page/size 를 생략하면 0/20 을 기본값으로 쓴다")
    void listUsesDefaultPaging() throws Exception {
        when(vehicleService.search(any(), anyInt(), anyInt()))
                .thenReturn(new PageResponse<>(List.of(), 0L, 0, 0, 20));

        mockMvc.perform(get("/api/v1/vehicles"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.content").isArray());

        verify(vehicleService).search(null, 0, 20);
    }

    @Test
    @DisplayName("GET /api/v1/vehicles/{id} 는 단건을 돌려준다")
    void getReturnsSingleVehicle() throws Exception {
        when(vehicleService.get(1L)).thenReturn(sampleVehicle());

        mockMvc.perform(get("/api/v1/vehicles/{id}", 1L))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.plateNumber").value("12가3456"))
                .andExpect(jsonPath("$.modelName").value("아반떼 CN7"))
                .andExpect(jsonPath("$.modelYear").value(2021))
                .andExpect(jsonPath("$.vin").value("KMHLM41AAMU000111"));
    }

    @Test
    @DisplayName("GET /api/v1/vehicles/by-plate/{plate} 는 번호로 조회한다")
    void getByPlate() throws Exception {
        when(vehicleService.getByPlate("12가3456")).thenReturn(sampleVehicle());

        mockMvc.perform(get("/api/v1/vehicles/by-plate/{plate}", "12가3456"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.plateNumber").value("12가3456"));

        verify(vehicleService).getByPlate("12가3456");
    }

    @Test
    @DisplayName("POST /api/v1/vehicles 는 201 과 생성된 차량을 돌려준다")
    void createReturns201() throws Exception {
        when(vehicleService.create(any(VehicleRequest.class))).thenReturn(sampleVehicle());

        String body = """
                {
                  "plateNumber": "12가3456",
                  "ownerName": "홍길동",
                  "phone": "010-1111-2222",
                  "vehicleType": "SEDAN",
                  "modelName": "아반떼 CN7",
                  "modelYear": 2021,
                  "vin": "KMHLM41AAMU000111",
                  "passType": "REGULAR",
                  "active": true
                }
                """;

        mockMvc.perform(post("/api/v1/vehicles")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.plateNumber").value("12가3456"));
    }

    @Test
    @DisplayName("PUT /api/v1/vehicles/{id} 는 수정된 차량을 돌려준다")
    void updateReturnsVehicle() throws Exception {
        when(vehicleService.update(any(Long.class), any(VehicleRequest.class)))
                .thenReturn(sampleVehicle());

        String body = """
                {"plateNumber":"12가3456","ownerName":"홍길동","passType":"REGULAR"}
                """;

        mockMvc.perform(put("/api/v1/vehicles/{id}", 1L)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1));
    }

    @Test
    @DisplayName("필수 필드가 빠지면 400 과 공통 오류 포맷을 돌려준다")
    void createWithoutRequiredFieldsReturns400() throws Exception {
        String body = """
                {
                  "plateNumber": "",
                  "ownerName": ""
                }
                """;

        mockMvc.perform(post("/api/v1/vehicles")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400))
                .andExpect(jsonPath("$.error").value("Bad Request"))
                .andExpect(jsonPath("$.path").value("/api/v1/vehicles"))
                .andExpect(jsonPath("$.timestamp").exists())
                .andExpect(jsonPath("$.message").exists());
    }

    @Test
    @DisplayName("중복 번호판은 400(BusinessException)으로 매핑된다")
    void duplicatePlateReturns400() throws Exception {
        when(vehicleService.create(any(VehicleRequest.class)))
                .thenThrow(new BusinessException("이미 등록된 차량 번호입니다: 12가3456"));

        String body = """
                {"plateNumber":"12가3456","ownerName":"홍길동"}
                """;

        mockMvc.perform(post("/api/v1/vehicles")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.status").value(400))
                .andExpect(jsonPath("$.message").value("이미 등록된 차량 번호입니다: 12가3456"));
    }

    @Test
    @DisplayName("없는 차량 조회는 404(NotFoundException)으로 매핑된다")
    void missingVehicleReturns404() throws Exception {
        when(vehicleService.get(999L)).thenThrow(NotFoundException.of("차량", 999L));

        mockMvc.perform(get("/api/v1/vehicles/{id}", 999L))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404))
                .andExpect(jsonPath("$.error").value("Not Found"))
                .andExpect(jsonPath("$.path").value("/api/v1/vehicles/999"));
    }

    @Test
    @DisplayName("DELETE /api/v1/vehicles/{id} 는 204 를 돌려준다")
    void deleteReturns204() throws Exception {
        mockMvc.perform(delete("/api/v1/vehicles/{id}", 1L))
                .andExpect(status().isNoContent());

        verify(vehicleService).delete(1L);
    }

    @Test
    @DisplayName("없는 차량 삭제는 404 를 돌려준다")
    void deleteMissingReturns404() throws Exception {
        doThrow(NotFoundException.of("차량", 999L)).when(vehicleService).delete(999L);

        mockMvc.perform(delete("/api/v1/vehicles/{id}", 999L))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404));
    }
}
