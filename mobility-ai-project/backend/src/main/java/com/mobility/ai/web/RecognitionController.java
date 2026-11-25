package com.mobility.ai.web;

import com.mobility.ai.common.PageResponse;
import com.mobility.ai.dto.RecognitionDtos.RecognitionResponse;
import com.mobility.ai.dto.RecognitionDtos.RecognitionResultResponse;
import com.mobility.ai.service.RecognitionService;
import java.time.LocalDateTime;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * 번호판 인식 컨트롤러. 계약서 4장의 경로와 1:1 이다.
 *
 * <pre>
 * POST /api/v1/recognitions                 (multipart: file, gate, eventType)
 * GET  /api/v1/recognitions?from=&to=&status=&page=&size=
 * GET  /api/v1/recognitions/failures?page=&size=
 * </pre>
 *
 * <p>AI 서비스가 내려가 있어도 이 엔드포인트는 <b>200</b> 을 반환하고
 * 본문의 {@code allowed=false}, {@code rejectReason="AI_SERVICE_UNAVAILABLE"} 로
 * 상황을 알린다(계약서 요구사항).
 */
@RestController
@RequestMapping("/api/v1/recognitions")
public class RecognitionController {

    private final RecognitionService recognitionService;

    /**
     * @param recognitionService 인식 서비스
     */
    public RecognitionController(RecognitionService recognitionService) {
        this.recognitionService = recognitionService;
    }

    /**
     * 이미지 업로드 → 인식 → 출입/주차 처리.
     *
     * @param file       이미지 파일(필수)
     * @param gate       게이트 이름(기본 UNKNOWN)
     * @param eventType  ENTRY|EXIT (기본 ENTRY)
     * @param minConf    최소 신뢰도(AI 로 전달, 선택)
     * @param detector   검출기 선택(yolo|mobilenet, 선택)
     * @param recognizer OCR 선택(crnn|attention|paddle, 선택)
     * @return 인식/처리 결과
     */
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public RecognitionResultResponse recognize(
            @RequestPart("file") MultipartFile file,
            @RequestParam(name = "gate", required = false) String gate,
            @RequestParam(name = "eventType", required = false) String eventType,
            @RequestParam(name = "minConf", required = false) Double minConf,
            @RequestParam(name = "detector", required = false) String detector,
            @RequestParam(name = "recognizer", required = false) String recognizer) {
        return recognitionService.recognize(file, gate, eventType, minConf, detector, recognizer);
    }

    /**
     * 인식 이력 조회.
     *
     * @param from   시작 시각(ISO-8601, 선택)
     * @param to     종료 시각(ISO-8601, 선택)
     * @param status SUCCESS|LOW_CONFIDENCE|NO_PLATE|FORMAT_INVALID (선택)
     * @param page   페이지 번호
     * @param size   페이지 크기
     * @return 인식 이력 페이지
     */
    @GetMapping
    public PageResponse<RecognitionResponse> list(
            @RequestParam(name = "from", required = false)
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime from,
            @RequestParam(name = "to", required = false)
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime to,
            @RequestParam(name = "status", required = false) String status,
            @RequestParam(name = "page", defaultValue = "0") int page,
            @RequestParam(name = "size", defaultValue = "20") int size) {
        return recognitionService.search(from, to, status, page, size);
    }

    /**
     * 인식 실패 목록.
     *
     * @param page 페이지 번호
     * @param size 페이지 크기
     * @return 실패 이력 페이지
     */
    @GetMapping("/failures")
    public PageResponse<RecognitionResponse> failures(
            @RequestParam(name = "page", defaultValue = "0") int page,
            @RequestParam(name = "size", defaultValue = "20") int size) {
        return recognitionService.failures(page, size);
    }
}
