package com.mobility.ai.service;

import com.mobility.ai.client.AiInferenceClient;
import com.mobility.ai.client.AiRecognitionResult;
import com.mobility.ai.common.BusinessException;
import com.mobility.ai.common.PageResponse;
import com.mobility.ai.config.AppProperties;
import com.mobility.ai.domain.AccessEvent;
import com.mobility.ai.domain.EventType;
import com.mobility.ai.domain.ParkingSession;
import com.mobility.ai.domain.PlateRecognition;
import com.mobility.ai.domain.RecognitionStatus;
import com.mobility.ai.domain.Vehicle;
import com.mobility.ai.dto.RecognitionDtos.DetectionBox;
import com.mobility.ai.dto.RecognitionDtos.ElapsedMs;
import com.mobility.ai.dto.RecognitionDtos.QualityInfo;
import com.mobility.ai.dto.RecognitionDtos.RecognitionResponse;
import com.mobility.ai.dto.RecognitionDtos.RecognitionResultResponse;
import com.mobility.ai.dto.VehicleDtos.VehicleSummary;
import com.mobility.ai.repository.PlateRecognitionRepository;
import com.mobility.ai.repository.VehicleRepository;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

/**
 * 번호판 인식 파이프라인 서비스 — 이 프로젝트의 핵심 흐름이다
 * (교재 6장 "인식 → 출입 → 과금 연계" 절 인용 대상).
 *
 * <h2>처리 순서</h2>
 * <ol>
 *   <li>multipart 파일을 {@code app.storage.upload-dir}/yyyy-MM-dd/ 아래에 저장</li>
 *   <li>{@link AiInferenceClient} 로 AI 서비스 {@code /api/v1/lpr/recognize} 호출</li>
 *   <li>결과를 {@code plate_recognition} 에 저장(상태/지연/품질 포함)</li>
 *   <li>번호판으로 {@code vehicle} 매칭</li>
 *   <li>ENTRY → {@code parking_session} 생성 + {@code access_event(ENTRY)}<br>
 *       EXIT  → 세션 종료 + 요금 계산 + {@code access_event(EXIT)}</li>
 *   <li>{@code RecognitionResultResponse} 조립</li>
 * </ol>
 *
 * <h2>AI 장애 시 동작(계약서 요구사항)</h2>
 * AI 서비스가 죽어 있으면 <b>503 을 던지지 않는다.</b>
 * {@code status=NO_PLATE} 로 기록하고 HTTP 200 + {@code allowed=false},
 * {@code rejectReason="AI_SERVICE_UNAVAILABLE"} 로 응답한다.
 * 운영 화면은 멈추지 않고 실패 건을 그대로 보여줄 수 있다.
 */
@Service
public class RecognitionService {

    private static final Logger log = LoggerFactory.getLogger(RecognitionService.class);

    /** 거부 사유 코드: AI 서비스 장애 */
    public static final String REJECT_AI_UNAVAILABLE = "AI_SERVICE_UNAVAILABLE";
    /** 거부 사유 코드: 번호판 미검출 */
    public static final String REJECT_NO_PLATE = "NO_PLATE";
    /** 거부 사유 코드: 신뢰도 미달 */
    public static final String REJECT_LOW_CONFIDENCE = "LOW_CONFIDENCE";
    /** 거부 사유 코드: 번호판 형식 불일치 */
    public static final String REJECT_FORMAT_INVALID = "FORMAT_INVALID";
    /** 거부 사유 코드: 미등록 차량 */
    public static final String REJECT_NOT_REGISTERED = "NOT_REGISTERED";
    /** 거부 사유 코드: 비활성 차량 */
    public static final String REJECT_INACTIVE_VEHICLE = "INACTIVE_VEHICLE";

    private static final DateTimeFormatter DATE_DIR = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private final AiInferenceClient aiInferenceClient;
    private final PlateRecognitionRepository plateRecognitionRepository;
    private final VehicleRepository vehicleRepository;
    private final AccessEventService accessEventService;
    private final ParkingService parkingService;
    private final AppProperties appProperties;

    /**
     * @param aiInferenceClient          AI 호출 클라이언트
     * @param plateRecognitionRepository 인식 이력 리포지토리
     * @param vehicleRepository          차량 리포지토리
     * @param accessEventService         출입 이벤트 서비스
     * @param parkingService             주차 세션 서비스
     * @param appProperties              {@code app.*} 설정
     */
    public RecognitionService(AiInferenceClient aiInferenceClient,
                              PlateRecognitionRepository plateRecognitionRepository,
                              VehicleRepository vehicleRepository,
                              AccessEventService accessEventService,
                              ParkingService parkingService,
                              AppProperties appProperties) {
        this.aiInferenceClient = aiInferenceClient;
        this.plateRecognitionRepository = plateRecognitionRepository;
        this.vehicleRepository = vehicleRepository;
        this.accessEventService = accessEventService;
        this.parkingService = parkingService;
        this.appProperties = appProperties;
    }

    // =================================================================
    // 메인 파이프라인
    // =================================================================

    /**
     * 이미지 1장을 받아 인식 → 출입 → 과금까지 처리한다.
     *
     * @param file        업로드 이미지(필수)
     * @param gate        게이트 이름(null 이면 "UNKNOWN")
     * @param eventTypeRaw ENTRY|EXIT (null 이면 ENTRY)
     * @param minConf     최소 신뢰도(null 이면 AI 기본값)
     * @param detector    검출기 선택(null 이면 AI 기본값)
     * @param recognizer  OCR 선택(null 이면 AI 기본값)
     * @return 인식/처리 결과
     * @throws BusinessException 파일이 비었을 때
     */
    @Transactional
    public RecognitionResultResponse recognize(MultipartFile file,
                                               String gate,
                                               String eventTypeRaw,
                                               Double minConf,
                                               String detector,
                                               String recognizer) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException("이미지 파일(file)이 필요합니다.");
        }

        String gateName = (gate == null || gate.isBlank()) ? "UNKNOWN" : gate.trim();
        EventType eventType = parseEventType(eventTypeRaw);

        // 1) 파일 저장 (저장 실패해도 인식은 계속한다 — 경로만 null 로 남긴다)
        String storedPath = storeFile(file);

        // 2) AI 호출
        byte[] bytes = readBytes(file);
        AiRecognitionResult ai = aiInferenceClient.recognize(
                bytes, file.getOriginalFilename(), file.getContentType(), minConf, detector, recognizer);

        // 3) 상태 판정 + 인식 기록 저장
        String normalizedPlate = PlateFormatValidator.normalize(ai.plateNumber());
        boolean validFormat = ai.validFormat() != null
                ? ai.validFormat()
                : PlateFormatValidator.isValid(normalizedPlate);
        RecognitionStatus status = resolveStatus(ai, normalizedPlate, validFormat);

        // 4) 차량 매칭 — 형식이 유효한 번호판만 매칭을 시도한다.
        Vehicle vehicle = null;
        if (normalizedPlate != null && status == RecognitionStatus.SUCCESS) {
            vehicle = vehicleRepository.findByPlateNumber(normalizedPlate).orElse(null);
        }

        PlateRecognition recognition = new PlateRecognition(status);
        recognition.setPlateNumber(normalizedPlate);
        recognition.setConfidence(ai.confidence());
        recognition.setDetectorName(ai.detector());
        recognition.setOcrName(ai.recognizer());
        recognition.setImagePath(storedPath);
        recognition.setDetectMs(ai.detectMs());
        recognition.setOcrMs(ai.ocrMs());
        recognition.setTotalMs(ai.totalMs());
        recognition.setQualityScore(ai.qualityScore());
        recognition.setValidFormat(validFormat);
        recognition.setMatchedVehicle(vehicle);
        recognition = plateRecognitionRepository.save(recognition);

        // 5) 통행 허용 판정
        String rejectReason = resolveRejectReason(ai, status, vehicle);
        boolean allowed = rejectReason == null;

        // 6) 출입 이벤트 기록(중복 제거 적용)
        AccessEventService.EventCreation creation = accessEventService.record(
                eventType, normalizedPlate, gateName, vehicle, recognition,
                ai.confidence(), allowed, rejectReason);
        AccessEvent event = creation.event();

        // 7) 주차 세션 처리 — 허용된 건에 대해서만, 그리고 중복 이벤트가 아닐 때만 수행
        Long sessionId = null;
        Long feeAmount = null;
        Integer parkedMinutes = null;

        if (allowed && !creation.duplicated()) {
            if (eventType == EventType.ENTRY) {
                ParkingSession session = parkingService.openSession(vehicle, normalizedPlate, event);
                sessionId = session == null ? null : session.getId();
            } else {
                ParkingSession session = parkingService.closeSession(vehicle, normalizedPlate, event);
                if (session != null) {
                    sessionId = session.getId();
                    feeAmount = session.getFeeAmount();
                    parkedMinutes = session.getParkedMinutes();
                }
            }
        }

        log.info("인식 처리 완료: plate={}, status={}, type={}, gate={}, allowed={}, reason={}",
                normalizedPlate, status, eventType, gateName, allowed, rejectReason);

        return new RecognitionResultResponse(
                recognition.getId(),
                normalizedPlate,
                ai.confidence(),
                status.name(),
                validFormat,
                VehicleSummary.from(vehicle),
                event.getId(),
                eventType.name(),
                allowed,
                rejectReason,
                sessionId,
                feeAmount,
                parkedMinutes,
                toDetectionBoxes(ai),
                new ElapsedMs(ai.detectMs(), ai.ocrMs(), ai.totalMs()),
                new QualityInfo(ai.qualityScore(), ai.readable())
        );
    }

    // =================================================================
    // 이력 조회
    // =================================================================

    /**
     * 인식 이력 검색.
     *
     * @param from   시작 시각(null 허용)
     * @param to     종료 시각(null 허용)
     * @param status 상태 문자열(null/빈값이면 전체)
     * @param page   페이지 번호
     * @param size   페이지 크기
     * @return 인식 이력 페이지
     */
    @Transactional(readOnly = true)
    public PageResponse<RecognitionResponse> search(LocalDateTime from, LocalDateTime to,
                                                    String status, int page, int size) {
        RecognitionStatus parsed = parseStatus(status);
        PageRequest pageable = PageRequest.of(
                Math.max(0, page),
                normalizeSize(size),
                Sort.by(Sort.Direction.DESC, "createdAt").and(Sort.by(Sort.Direction.DESC, "id"))
        );
        Page<PlateRecognition> result = plateRecognitionRepository.search(from, to, parsed, pageable);
        return PageResponse.from(result, RecognitionResponse::from);
    }

    /**
     * 인식 실패 목록(SUCCESS 가 아닌 모든 건).
     *
     * @param page 페이지 번호
     * @param size 페이지 크기
     * @return 실패 이력 페이지
     */
    @Transactional(readOnly = true)
    public PageResponse<RecognitionResponse> failures(int page, int size) {
        PageRequest pageable = PageRequest.of(
                Math.max(0, page),
                normalizeSize(size),
                Sort.by(Sort.Direction.DESC, "createdAt").and(Sort.by(Sort.Direction.DESC, "id"))
        );
        Page<PlateRecognition> result = plateRecognitionRepository.findFailures(pageable);
        return PageResponse.from(result, RecognitionResponse::from);
    }

    // =================================================================
    // 내부 로직
    // =================================================================

    /**
     * AI 결과와 형식 검증을 합쳐 최종 상태를 결정한다.
     *
     * <p>판정 우선순위
     * <ol>
     *   <li>AI 서비스 장애 → {@code NO_PLATE} (계약서 요구사항)</li>
     *   <li>번호판 문자열 없음 → {@code NO_PLATE}</li>
     *   <li>형식 불일치 → {@code FORMAT_INVALID}</li>
     *   <li>신뢰도 임계값 미만 → {@code LOW_CONFIDENCE}</li>
     *   <li>그 외 → {@code SUCCESS}</li>
     * </ol>
     *
     * @param ai          AI 결과
     * @param plate       정규화된 번호판(null 가능)
     * @param validFormat 형식 적합 여부
     * @return 최종 상태
     */
    private RecognitionStatus resolveStatus(AiRecognitionResult ai, String plate, boolean validFormat) {
        if (!ai.available()) {
            return RecognitionStatus.NO_PLATE;
        }
        if (plate == null || plate.isBlank()) {
            return RecognitionStatus.NO_PLATE;
        }
        if (!validFormat) {
            return RecognitionStatus.FORMAT_INVALID;
        }
        double threshold = appProperties.getRecognition().getMinConfidence();
        double confidence = ai.confidence() == null ? 0.0 : ai.confidence();
        if (confidence < threshold) {
            return RecognitionStatus.LOW_CONFIDENCE;
        }
        return RecognitionStatus.SUCCESS;
    }

    /**
     * 통행 거부 사유를 결정한다. null 이면 허용이다.
     *
     * @param ai      AI 결과
     * @param status  판정된 상태
     * @param vehicle 매칭된 차량(null 가능)
     * @return 거부 사유 코드 또는 null
     */
    private String resolveRejectReason(AiRecognitionResult ai, RecognitionStatus status, Vehicle vehicle) {
        if (!ai.available()) {
            return REJECT_AI_UNAVAILABLE;
        }
        switch (status) {
            case NO_PLATE -> {
                return REJECT_NO_PLATE;
            }
            case FORMAT_INVALID -> {
                return REJECT_FORMAT_INVALID;
            }
            case LOW_CONFIDENCE -> {
                return REJECT_LOW_CONFIDENCE;
            }
            default -> {
                // SUCCESS — 등록 차량인지 확인한다.
            }
        }
        if (vehicle == null) {
            return REJECT_NOT_REGISTERED;
        }
        if (!vehicle.isActive()) {
            return REJECT_INACTIVE_VEHICLE;
        }
        return null;
    }

    /**
     * AI 검출 박스를 응답 DTO 로 바꾼다.
     *
     * @param ai AI 결과
     * @return 검출 박스 목록(없으면 빈 목록)
     */
    private List<DetectionBox> toDetectionBoxes(AiRecognitionResult ai) {
        if (ai.detections() == null || ai.detections().isEmpty()) {
            return List.of();
        }
        List<DetectionBox> boxes = new ArrayList<>(ai.detections().size());
        for (AiRecognitionResult.AiDetection d : ai.detections()) {
            boxes.add(new DetectionBox(d.bbox(), d.score()));
        }
        return boxes;
    }

    /**
     * 업로드 파일을 날짜별 디렉터리에 저장한다.
     *
     * <p>원본 파일명은 신뢰하지 않고 확장자만 취해 UUID 파일명을 새로 만든다
     * (경로 조작 방지).
     *
     * @param file 업로드 파일
     * @return 저장된 절대 경로 문자열. 저장 실패 시 null
     */
    private String storeFile(MultipartFile file) {
        try {
            Path baseDir = Paths.get(appProperties.getStorage().getUploadDir())
                    .toAbsolutePath().normalize();
            Path dayDir = baseDir.resolve(LocalDate.now().format(DATE_DIR));
            Files.createDirectories(dayDir);

            String extension = extensionOf(file.getOriginalFilename(), file.getContentType());
            Path target = dayDir.resolve(UUID.randomUUID() + extension);

            try (var in = file.getInputStream()) {
                Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            }
            return target.toString();
        } catch (IOException | RuntimeException e) {
            log.warn("업로드 파일 저장 실패 — 경로 없이 진행합니다: {}", e.getMessage());
            return null;
        }
    }

    /**
     * 파일 확장자를 결정한다. 원본 파일명에서 못 얻으면 Content-Type 으로 추정한다.
     *
     * @param originalFilename 원본 파일명(null 허용)
     * @param contentType      MIME 타입(null 허용)
     * @return 점을 포함한 확장자(기본 {@code .jpg})
     */
    private String extensionOf(String originalFilename, String contentType) {
        if (originalFilename != null) {
            int dot = originalFilename.lastIndexOf('.');
            if (dot >= 0 && dot < originalFilename.length() - 1) {
                String ext = originalFilename.substring(dot + 1).toLowerCase();
                if (ext.matches("[a-z0-9]{1,5}")) {
                    return "." + ext;
                }
            }
        }
        if (contentType != null) {
            if (contentType.contains("png")) {
                return ".png";
            }
            if (contentType.contains("webp")) {
                return ".webp";
            }
            if (contentType.contains("bmp")) {
                return ".bmp";
            }
        }
        return ".jpg";
    }

    /**
     * multipart 바이트를 읽는다. 실패 시 빈 배열(AI 가 NO_PLATE 로 응답).
     *
     * @param file 업로드 파일
     * @return 바이트 배열
     */
    private byte[] readBytes(MultipartFile file) {
        try {
            return file.getBytes();
        } catch (IOException e) {
            log.warn("업로드 파일 읽기 실패: {}", e.getMessage());
            return new byte[0];
        }
    }

    /**
     * 이벤트 타입 문자열을 변환한다. 비었거나 알 수 없으면 ENTRY 로 본다.
     *
     * @param raw 입력 문자열
     * @return 이벤트 타입
     */
    private EventType parseEventType(String raw) {
        if (raw == null || raw.isBlank()) {
            return EventType.ENTRY;
        }
        for (EventType type : EventType.values()) {
            if (type.name().equalsIgnoreCase(raw.trim())) {
                return type;
            }
        }
        throw new BusinessException("eventType 은 ENTRY 또는 EXIT 이어야 합니다: " + raw);
    }

    /**
     * 상태 문자열을 변환한다. 비었거나 알 수 없으면 null(전체 조회).
     *
     * @param raw 입력 문자열
     * @return 상태 또는 null
     */
    private RecognitionStatus parseStatus(String raw) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        for (RecognitionStatus status : RecognitionStatus.values()) {
            if (status.name().equalsIgnoreCase(raw.trim())) {
                return status;
            }
        }
        return null;
    }

    /**
     * 페이지 크기를 1~200 으로 제한한다.
     *
     * @param size 요청 크기
     * @return 보정된 크기
     */
    private int normalizeSize(int size) {
        if (size <= 0) {
            return 20;
        }
        return Math.min(size, 200);
    }
}
