"use client";

import { useEffect, useRef, useState } from "react";
import type { Camera as MediaPipeCamera } from "@mediapipe/camera_utils";
import type { FaceMesh as MediaPipeFaceMesh, NormalizedLandmark, Results } from "@mediapipe/face_mesh";

const READER_DB_NAME = "motion-pdf-reader";
const READER_STORE_NAME = "reader-state";
const SESSION_STATE_KEY = "session-state";
const SAVED_SCORES_KEY = "saved-scores";
const WINK_MIRROR_KEY = "wink-mirror";
const MOTION_DETECTION_ENABLED_KEY = "motion-detection-enabled";
const CALIBRATION_SAMPLE_TARGET = 18;
const WINK_HOLD_DURATION_MS = 260;
const PAGE_TURN_COOLDOWN_MS = 1200;
const EYE_RATIO_SMOOTHING_ALPHA = 0.7;
const WINK_START_CLOSED_RATIO_THRESHOLD = 0.78;
const WINK_START_OTHER_EYE_OPEN_RATIO_THRESHOLD = 0.82;
const WINK_START_CLOSURE_GAP_THRESHOLD = 0.16;
const WINK_CONTINUE_CLOSED_RATIO_THRESHOLD = 0.84;
const WINK_CONTINUE_OTHER_EYE_OPEN_RATIO_THRESHOLD = 0.72;
const WINK_CONTINUE_CLOSURE_GAP_THRESHOLD = 0.12;
const WINK_MIN_PEAK_STRENGTH = 0.18;
const BOTH_EYES_CLOSED_RATIO_THRESHOLD = 0.86;
const BOTH_EYES_CLOSED_COMBINED_THRESHOLD = 1.72;
const RECOVERY_OPEN_RATIO_THRESHOLD = 0.84;
const RECOVERY_OPEN_RATIO_FAST_THRESHOLD = 0.78;
const EYE_VISIBILITY_BALANCE_THRESHOLD = 0.5;
const EYE_CENTER_BALANCE_THRESHOLD = 0.46;
const CALIBRATION_FORWARD_TILT_MAX = 0.1;
const CALIBRATION_FORWARD_NOSE_X_MAX = 0.16;
const CALIBRATION_FORWARD_NOSE_Y_MAX = 0.38;
const CALIBRATION_EYE_DISTANCE_MIN = 0.035;
const ACTIVE_FORWARD_TILT_MAX = 0.11;
const ACTIVE_FORWARD_NOSE_X_MAX = 0.14;
const ACTIVE_FORWARD_NOSE_Y_MAX = 0.4;
const ACTIVE_FORWARD_EYE_DISTANCE_MIN = 0.04;
const ACTIVE_FORWARD_EYE_WIDTH_BALANCE_MIN = 0.62;
const ACTIVE_FORWARD_EYE_CENTER_BALANCE_MIN = 0.58;
const FORWARD_STABLE_FRAME_TARGET = 2;
const REARM_OPEN_FRAME_TARGET = 3;
const BOTH_EYES_SIMILARITY_GAP_THRESHOLD = 0.025;
const BOTH_EYES_PARTIAL_CLOSED_THRESHOLD = 0.82;
const BOTH_EYES_DOMINANCE_STRENGTH_MAX = 0.1;
const BOTH_EYES_BLINK_RATIO_THRESHOLD = 0.9;
const BOTH_EYES_BLINK_SIMILARITY_GAP_MAX = 0.12;
const BOTH_EYES_BLINK_DOMINANCE_MAX = 0.2;
const WINK_STABLE_FRAME_TARGET = 2;
const WINK_MISSING_FRAME_TOLERANCE = 1;

type SavedScore = {
  id: string;
  name: string;
  pdfBlob: Blob;
  currentPage: number;
  updatedAt: number;
};

type SessionState = {
  pdfBlob: Blob | null;
  fileName: string | null;
  currentPage: number;
  savedScoreId: string | null;
};

type WinkEye = "left" | "right" | "none";
type WinkDirection = "left" | "right" | "none";
type WinkPhase = "idle" | "tracking" | "waiting-open";

type WinkDetectionState = {
  phase: WinkPhase;
  eye: WinkEye;
  direction: WinkDirection;
  startedAt: number | null;
  peakStrength: number;
  stableFrames: number;
  missingFrames: number;
  openFrames: number;
};

type WinkDebugState = {
  status: string;
  detail: string;
  leftNormalized: number | null;
  rightNormalized: number | null;
  leftStrength: number;
  rightStrength: number;
  forwardStableFrames: number;
  trackingPhase: WinkPhase;
  trackingEye: WinkEye;
  stableFrames: number;
  openFrames: number;
};

declare global {
  interface Window {
    pdfjsLib: any;
  }
}

const openReaderDb = () =>
  new Promise<IDBDatabase>((resolve, reject) => {
    const request = window.indexedDB.open(READER_DB_NAME, 1);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(READER_STORE_NAME)) {
        db.createObjectStore(READER_STORE_NAME);
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

const readReaderValue = async <T,>(key: string): Promise<T | null> => {
  const db = await openReaderDb();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(READER_STORE_NAME, "readonly");
    const store = transaction.objectStore(READER_STORE_NAME);
    const request = store.get(key);

    transaction.oncomplete = () => {
      resolve((request.result as T | undefined) ?? null);
      db.close();
    };

    transaction.onerror = () => {
      reject(transaction.error);
      db.close();
    };
  });
};

const writeReaderValue = async (key: string, value: unknown) => {
  const db = await openReaderDb();

  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(READER_STORE_NAME, "readwrite");
    transaction.objectStore(READER_STORE_NAME).put(value, key);

    transaction.oncomplete = () => {
      resolve();
      db.close();
    };

    transaction.onerror = () => {
      reject(transaction.error);
      db.close();
    };
  });
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pageText, setPageText] = useState("페이지: 0 / 0");
  const [gestureText, setGestureText] = useState("PDF를 먼저 불러오면 다음 단계가 쉬워집니다.");
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [pageCount, setPageCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [currentFileName, setCurrentFileName] = useState<string | null>(null);
  const [savedScores, setSavedScores] = useState<SavedScore[]>([]);
  const [isLibraryOpen, setIsLibraryOpen] = useState(false);
  const [isMotionGuideExpanded, setIsMotionGuideExpanded] = useState(true);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [isFocusControlsVisible, setIsFocusControlsVisible] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [isWinkMirrored, setIsWinkMirrored] = useState(true);
  const [cameraPermission, setCameraPermission] = useState<"unknown" | "prompt" | "granted" | "denied">(
    "unknown"
  );
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [hasCalibration, setHasCalibration] = useState(false);
  const [winkDebug, setWinkDebug] = useState<WinkDebugState>({
    status: "대기 중",
    detail: "모션 감지를 켜면 실패 이유가 여기에 표시됩니다.",
    leftNormalized: null,
    rightNormalized: null,
    leftStrength: 0,
    rightStrength: 0,
    forwardStableFrames: 0,
    trackingPhase: "idle",
    trackingEye: "none",
    stableFrames: 0,
    openFrames: 0,
  });

  const lastSwitchTimeRef = useRef<number>(0);
  const isRenderingRef = useRef(false);
  const pendingPageRef = useRef<number | null>(null);
  const cameraInstanceRef = useRef<MediaPipeCamera | null>(null);
  const faceMeshRef = useRef<MediaPipeFaceMesh | null>(null);
  const pdfDocRef = useRef<any>(null);
  const pageCountRef = useRef(0);
  const currentPageRef = useRef(1);
  const currentFileNameRef = useRef<string | null>(null);
  const currentPdfBlobRef = useRef<Blob | null>(null);
  const currentSavedScoreIdRef = useRef<string | null>(null);
  const winkStateRef = useRef<WinkDetectionState>({
    phase: "idle",
    eye: "none",
    direction: "none",
    startedAt: null,
    peakStrength: 0,
    stableFrames: 0,
    missingFrames: 0,
    openFrames: 0,
  });
  const gestureTextRef = useRef("PDF를 먼저 불러오면 다음 단계가 쉬워집니다.");
  const calibrationActiveRef = useRef(false);
  const calibrationFramesRef = useRef(0);
  const calibrationLeftSumRef = useRef(0);
  const calibrationRightSumRef = useRef(0);
  const forwardStableFramesRef = useRef(0);
  const leftEyeOpenRef = useRef(0.24);
  const rightEyeOpenRef = useRef(0.24);
  const smoothedLeftEyeRef = useRef<number | null>(null);
  const smoothedRightEyeRef = useRef<number | null>(null);
  const focusControlsTimeoutRef = useRef<number | null>(null);

  const loadPdfJs = async () => {
    if (window.pdfjsLib) {
      return window.pdfjsLib;
    }

    const pdfjsLib = await import("pdfjs-dist/build/pdf.mjs");
    window.pdfjsLib = pdfjsLib;
    pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
    return pdfjsLib;
  };

  const updateGestureText = (nextText: string) => {
    if (gestureTextRef.current === nextText) {
      return;
    }

    gestureTextRef.current = nextText;
    setGestureText(nextText);
  };

  const resetWinkState = (phase: WinkPhase = "idle") => {
    winkStateRef.current = {
      phase,
      eye: "none",
      direction: "none",
      startedAt: null,
      peakStrength: 0,
      stableFrames: 0,
      missingFrames: 0,
      openFrames: 0,
    };
  };

  const updateWinkDebug = (next: Partial<WinkDebugState>) => {
    setWinkDebug((prev) => ({
      ...prev,
      ...next,
    }));
  };

  const saveSessionState = async (overrides?: Partial<SessionState>) => {
    const sessionState: SessionState = {
      pdfBlob: currentPdfBlobRef.current,
      fileName: currentFileNameRef.current,
      currentPage: currentPageRef.current,
      savedScoreId: currentSavedScoreIdRef.current,
      ...overrides,
    };

    await writeReaderValue(SESSION_STATE_KEY, sessionState);
  };

  const saveScoresToDb = async (scores: SavedScore[]) => {
    setSavedScores(scores);
    await writeReaderValue(SAVED_SCORES_KEY, scores);
  };

  const loadPdfDocument = async ({
    blob,
    fileName,
    initialPage = 1,
    savedScoreId = null,
  }: {
    blob: Blob;
    fileName: string | null;
    initialPage?: number;
    savedScoreId?: string | null;
  }) => {
    await loadPdfJs();
    if (!window.pdfjsLib) {
      throw new Error("PDF 스크립트가 아직 준비되지 않았습니다.");
    }

    const data = await blob.arrayBuffer();
    const pdf = await window.pdfjsLib.getDocument({ data }).promise;
    const safeInitialPage = Math.max(1, Math.min(pdf.numPages, initialPage));

    setPdfDoc(pdf);
    setPageCount(pdf.numPages);
    setCurrentPage(safeInitialPage);
    setCurrentFileName(fileName);
    currentFileNameRef.current = fileName;
    pdfDocRef.current = pdf;
    pageCountRef.current = pdf.numPages;
    currentPageRef.current = safeInitialPage;
    currentPdfBlobRef.current = blob;
    currentSavedScoreIdRef.current = savedScoreId;

    await renderPage(pdf, safeInitialPage, pdf.numPages);
    await saveSessionState({
      pdfBlob: blob,
      fileName,
      currentPage: safeInitialPage,
      savedScoreId,
    });
  };

  useEffect(() => {
    let mounted = true;
    let permissionStatus: PermissionStatus | null = null;

    const syncPermissionState = async () => {
      if (!("permissions" in navigator) || !navigator.permissions?.query) {
        return;
      }

      try {
        permissionStatus = await navigator.permissions.query({ name: "camera" as PermissionName });
        if (!mounted) {
          return;
        }

        setCameraPermission(permissionStatus.state);
        permissionStatus.onchange = () => {
          if (mounted) {
            setCameraPermission(permissionStatus!.state);
          }
        };
      } catch (error) {
        console.warn("camera permission query unsupported", error);
      }
    };

    void syncPermissionState();

    return () => {
      mounted = false;
      if (permissionStatus) {
        permissionStatus.onchange = null;
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    const restoreReaderState = async () => {
      try {
        const [storedScores, sessionState, storedWinkMirror, storedMotionDetectionEnabled] = await Promise.all([
          readReaderValue<SavedScore[]>(SAVED_SCORES_KEY),
          readReaderValue<SessionState>(SESSION_STATE_KEY),
          readReaderValue<boolean>(WINK_MIRROR_KEY),
          readReaderValue<boolean>(MOTION_DETECTION_ENABLED_KEY),
        ]);

        if (cancelled) {
          return;
        }

        setSavedScores(storedScores ?? []);
        setIsWinkMirrored(storedWinkMirror ?? true);
        setCameraEnabled(storedMotionDetectionEnabled ?? false);

        if (!sessionState?.pdfBlob) {
          return;
        }

        await loadPdfDocument({
          blob: sessionState.pdfBlob,
          fileName: sessionState.fileName,
          initialPage: sessionState.currentPage,
          savedScoreId: sessionState.savedScoreId,
        });
      } catch (error) {
        console.error("pdf restore error", error);
      }
    };

    void restoreReaderState();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsFocusMode(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (!isFocusMode) {
      setIsFocusControlsVisible(false);
    }

    return () => {
      if (focusControlsTimeoutRef.current !== null) {
        window.clearTimeout(focusControlsTimeoutRef.current);
        focusControlsTimeoutRef.current = null;
      }
    };
  }, [isFocusMode]);

  useEffect(() => {
    let cancelled = false;

    const resetCalibration = () => {
      calibrationActiveRef.current = false;
      calibrationFramesRef.current = 0;
      calibrationLeftSumRef.current = 0;
      calibrationRightSumRef.current = 0;
      forwardStableFramesRef.current = 0;
      leftEyeOpenRef.current = 0.24;
      rightEyeOpenRef.current = 0.24;
      smoothedLeftEyeRef.current = null;
      smoothedRightEyeRef.current = null;
      resetWinkState();
      setCalibrationProgress(0);
      setHasCalibration(false);
    };

    const stopCamera = async () => {
      resetWinkState();
      updateWinkDebug({
        status: "모션 감지 꺼짐",
        detail: "카메라가 멈춰 있어 윙크를 검사하지 않습니다.",
        leftNormalized: null,
        rightNormalized: null,
        leftStrength: 0,
        rightStrength: 0,
        forwardStableFrames: 0,
        trackingPhase: "idle",
        trackingEye: "none",
        stableFrames: 0,
        openFrames: 0,
      });
      setIsCalibrating(false);
      resetCalibration();

      if (cameraInstanceRef.current) {
        await cameraInstanceRef.current.stop();
        cameraInstanceRef.current = null;
      }

      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }

      faceMeshRef.current = null;
    };

    const initFaceMesh = async () => {
      if (!cameraEnabled) {
        await stopCamera();
        if (cameraPermission === "granted") {
          updateGestureText("모션 감지 대기 중");
        } else {
          updateGestureText("모션 감지 꺼짐");
        }
        return;
      }

      if (!videoRef.current) {
        return;
      }

      try {
        const [{ FaceMesh }, { Camera }] = await Promise.all([
          import("@mediapipe/face_mesh"),
          import("@mediapipe/camera_utils"),
        ]);

        if (cancelled || !videoRef.current) {
          return;
        }

        const faceMesh = new FaceMesh({
          locateFile: (file: string) => `/mediapipe/face_mesh/${file}`,
        });
        faceMeshRef.current = faceMesh;

        faceMesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6,
        });

        resetCalibration();
        setIsCalibrating(true);
        calibrationActiveRef.current = true;
        updateWinkDebug({
          status: "보정 시작",
          detail: "정면을 보고 눈 기준값을 수집하는 중입니다.",
        });
        updateGestureText("정면을 1초만 봐 주세요");

        faceMesh.onResults((results: Results) => {
          if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            updateWinkDebug({
              status: "실패: 얼굴 없음",
              detail: "얼굴 랜드마크를 찾지 못해서 윙크를 검사할 수 없습니다.",
              leftNormalized: null,
              rightNormalized: null,
              leftStrength: 0,
              rightStrength: 0,
              forwardStableFrames: 0,
              trackingPhase: "idle",
              trackingEye: "none",
              stableFrames: 0,
              openFrames: 0,
            });
            updateGestureText("얼굴을 인식할 수 없음");
            forwardStableFramesRef.current = 0;
            resetWinkState();
            return;
          }

          const landmarks = results.multiFaceLandmarks[0];
          const rawLeftEyeRatio = computeEyeRatio(landmarks, 33, 133, [
            [159, 145],
            [158, 153],
            [160, 144],
          ]);
          const rawRightEyeRatio = computeEyeRatio(landmarks, 362, 263, [
            [386, 374],
            [387, 373],
            [385, 380],
          ]);
          const leftEyeRatio = getSmoothedEyeRatio(smoothedLeftEyeRef, rawLeftEyeRatio);
          const rightEyeRatio = getSmoothedEyeRatio(smoothedRightEyeRef, rawRightEyeRatio);
          const isForwardEnoughForWink = isFaceForwardEnoughForWink(landmarks);
          const isForwardEnoughForCalibration = isFaceForwardEnoughForCalibration(landmarks);
          const leftOpenBaseline = Math.max(leftEyeOpenRef.current, 0.0001);
          const rightOpenBaseline = Math.max(rightEyeOpenRef.current, 0.0001);
          const leftNormalized = leftEyeRatio / leftOpenBaseline;
          const rightNormalized = rightEyeRatio / rightOpenBaseline;
          const leftClosure = 1 - leftNormalized;
          const rightClosure = 1 - rightNormalized;
          const leftStrength = leftClosure - rightClosure;
          const rightStrength = rightClosure - leftClosure;
          const normalizedGap = Math.abs(leftNormalized - rightNormalized);
          const dominantWinkStrength = Math.max(leftStrength, rightStrength);
          const isOneEyeClearlyDominant = dominantWinkStrength > BOTH_EYES_DOMINANCE_STRENGTH_MAX;
          const bothEyesSimilarClosure = normalizedGap <= BOTH_EYES_SIMILARITY_GAP_THRESHOLD;
          const areBothEyesPartiallyClosed =
            leftNormalized < BOTH_EYES_PARTIAL_CLOSED_THRESHOLD &&
            rightNormalized < BOTH_EYES_PARTIAL_CLOSED_THRESHOLD &&
            bothEyesSimilarClosure &&
            !isOneEyeClearlyDominant;
          const areBothEyesBlinking =
            leftNormalized < BOTH_EYES_BLINK_RATIO_THRESHOLD &&
            rightNormalized < BOTH_EYES_BLINK_RATIO_THRESHOLD &&
            normalizedGap <= BOTH_EYES_BLINK_SIMILARITY_GAP_MAX &&
            dominantWinkStrength <= BOTH_EYES_BLINK_DOMINANCE_MAX;
          const winkState = winkStateRef.current;
          const winkSignal = computeWink(leftEyeRatio, rightEyeRatio, winkState.eye);
          const hasStrongWinkSignal = winkSignal.eye !== "none";

          updateWinkDebug({
            leftNormalized,
            rightNormalized,
            leftStrength,
              rightStrength,
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: winkState.phase,
              trackingEye: winkState.eye,
              stableFrames: winkState.stableFrames,
            openFrames: winkState.openFrames,
          });

          if (calibrationActiveRef.current) {
            if (!isForwardEnoughForCalibration) {
              updateWinkDebug({
                status: "보정 실패",
                detail: "보정 중에는 얼굴을 더 정면으로 맞춰야 합니다.",
              });
              updateGestureText("얼굴을 조금만 카메라 쪽으로");
              return;
            }

            calibrationFramesRef.current += 1;
            calibrationLeftSumRef.current += rawLeftEyeRatio;
            calibrationRightSumRef.current += rawRightEyeRatio;

            const nextProgress = Math.min(
              100,
              Math.round((calibrationFramesRef.current / CALIBRATION_SAMPLE_TARGET) * 100)
            );
            setCalibrationProgress(nextProgress);
            updateWinkDebug({
              status: `보정 중 ${nextProgress}%`,
              detail: "이 단계에서는 윙크 감지를 하지 않고 기본 눈 뜸 기준값만 잡습니다.",
            });
            updateGestureText(`보정 중 ${nextProgress}%`);

            if (calibrationFramesRef.current >= CALIBRATION_SAMPLE_TARGET) {
              leftEyeOpenRef.current = calibrationLeftSumRef.current / calibrationFramesRef.current;
              rightEyeOpenRef.current = calibrationRightSumRef.current / calibrationFramesRef.current;
              calibrationActiveRef.current = false;
              setIsCalibrating(false);
              setHasCalibration(true);
              updateWinkDebug({
                status: "보정 완료",
                detail: "이제 한쪽 눈 감기 조건을 검사할 준비가 됐습니다.",
              });
              updateGestureText("보정 완료");
            }
            return;
          }

          if (isForwardEnoughForWink) {
            forwardStableFramesRef.current = Math.min(forwardStableFramesRef.current + 1, FORWARD_STABLE_FRAME_TARGET);
          } else {
            forwardStableFramesRef.current = 0;
          }

          if (
            (!hasStrongWinkSignal && areBothEyesBlinking) ||
            (!hasStrongWinkSignal && areBothEyesClosed(leftEyeRatio, rightEyeRatio, dominantWinkStrength, normalizedGap)) ||
            (!hasStrongWinkSignal && areBothEyesPartiallyClosed)
          ) {
            updateWinkDebug({
              status: "실패: 양쪽 눈 감김",
              detail: hasStrongWinkSignal
                ? "두 눈이 함께 닫힌 상태로 보여 안전하게 양쪽 눈 감김으로 막았습니다."
                : `두 눈이 모두 닫힘 범위에 들어왔고 비율 차이(${normalizedGap.toFixed(2)})와 한쪽 눈 우세 강도(${dominantWinkStrength.toFixed(2)})가 모두 양눈 깜빡임 쪽에 가깝습니다.`,
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: winkState.phase,
              trackingEye: winkState.eye,
              stableFrames: winkState.stableFrames,
              openFrames: winkState.openFrames,
            });
            updateGestureText(winkState.phase === "waiting-open" ? "눈을 다시 떠 주세요" : "양쪽 눈 감김으로 인식 중");
            forwardStableFramesRef.current = 0;
            if (winkState.phase !== "waiting-open") {
              resetWinkState();
            }
            return;
          }

          if (!areBothEyesClearlyVisible(landmarks)) {
            updateWinkDebug({
              status: "실패: 눈 가시성 부족",
              detail: "양쪽 눈 크기나 위치 균형이 많이 달라 보여 정면 윙크로 확신하지 못했습니다.",
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: winkState.phase,
              trackingEye: winkState.eye,
              stableFrames: winkState.stableFrames,
              openFrames: winkState.openFrames,
            });
            updateGestureText("양쪽 눈이 보이게 정면을 봐 주세요");
            forwardStableFramesRef.current = 0;
            if (winkStateRef.current.phase !== "waiting-open") {
              resetWinkState();
            }
            return;
          }

          if (!isForwardEnoughForWink) {
            updateWinkDebug({
              status: "실패: 정면 아님",
              detail: "고개 회전이나 좌우 비대칭이 커서, 정면 윙크로 보기엔 너무 불안정합니다.",
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: winkState.phase,
              trackingEye: winkState.eye,
              stableFrames: winkState.stableFrames,
              openFrames: winkState.openFrames,
            });
            updateGestureText("고개를 정면으로 맞춰 주세요");
            if (winkStateRef.current.phase !== "waiting-open") {
              resetWinkState();
            }
            return;
          }

          if (forwardStableFramesRef.current < FORWARD_STABLE_FRAME_TARGET) {
            updateWinkDebug({
              status: "실패: 정면 유지 부족",
              detail: "정면 상태가 아직 충분한 프레임 수만큼 이어지지 않았습니다.",
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: winkState.phase,
              trackingEye: winkState.eye,
              stableFrames: winkState.stableFrames,
              openFrames: winkState.openFrames,
            });
            updateGestureText("정면을 잠깐 유지해 주세요");
            if (winkStateRef.current.phase !== "waiting-open") {
              resetWinkState();
            }
            return;
          }

          const eyesRecovered = areEyesRecovered(leftEyeRatio, rightEyeRatio);
          const eyesRecoveredFast = areEyesRecoveredFast(rawLeftEyeRatio, rawRightEyeRatio);
          if (winkState.phase === "waiting-open") {
            if (eyesRecovered || eyesRecoveredFast) {
              winkState.openFrames = Math.min(winkState.openFrames + 1, REARM_OPEN_FRAME_TARGET);
            } else {
              winkState.openFrames = 0;
            }

            if (winkState.openFrames >= REARM_OPEN_FRAME_TARGET) {
              resetWinkState();
              updateWinkDebug({
                status: "재무장 완료",
                detail: "눈이 다시 열린 것으로 확인되어 다음 윙크를 받을 수 있습니다.",
                trackingPhase: "idle",
                trackingEye: "none",
                stableFrames: 0,
                openFrames: 0,
              });
              updateGestureText("다시 윙크할 수 있습니다");
            } else {
              updateWinkDebug({
                status: "실패: 아직 재무장 중",
                detail: `재무장 확인 중입니다. 열린 눈 기준은 부드러운 값 ${RECOVERY_OPEN_RATIO_THRESHOLD.toFixed(2)} 또는 즉시 값 ${RECOVERY_OPEN_RATIO_FAST_THRESHOLD.toFixed(2)} 이상입니다.`,
                trackingPhase: winkState.phase,
                trackingEye: winkState.eye,
                stableFrames: winkState.stableFrames,
                openFrames: winkState.openFrames,
              });
              updateGestureText("눈을 다시 떠 주세요");
            }
            return;
          }

          const now = performance.now();

          if (winkSignal.eye === "none") {
            if (winkState.phase === "tracking") {
              winkState.missingFrames += 1;
              if (winkState.missingFrames > WINK_MISSING_FRAME_TOLERANCE) {
                resetWinkState();
                updateWinkDebug({
                  status: "실패: 유지 시간 부족",
                  detail: "한쪽 눈 감기가 잠깐 보였지만 충분히 오래 유지되지 않았습니다.",
                  trackingPhase: "idle",
                  trackingEye: "none",
                  stableFrames: 0,
                  openFrames: 0,
                });
                updateGestureText("짧게 한쪽 윙크");
              } else {
                const holdDuration = winkState.startedAt ? now - winkState.startedAt : 0;
                updateWinkDebug({
                  status: "확인 중",
                  detail: "한쪽 눈 감기를 추적 중이지만 아직 프레임이 더 필요합니다.",
                  trackingPhase: winkState.phase,
                  trackingEye: winkState.eye,
                  stableFrames: winkState.stableFrames,
                  openFrames: winkState.openFrames,
                });
                updateGestureText(
                  winkState.direction === "none"
                    ? "윙크 확인 중"
                    : getPendingGestureText(winkState.direction, holdDuration)
                );
              }
            } else {
              updateWinkDebug({
                status: "실패: 시작 조건 미달",
                detail: "닫힌 눈과 뜬 눈의 차이가 임계값보다 작아서 한쪽 눈 감기로 시작하지 못했습니다.",
                trackingPhase: "idle",
                trackingEye: "none",
                stableFrames: 0,
                openFrames: 0,
              });
              updateGestureText("짧게 한쪽 윙크");
            }
            return;
          }

          if (winkState.phase !== "tracking" || winkState.eye !== winkSignal.eye) {
            winkStateRef.current = {
              phase: "tracking",
              eye: winkSignal.eye,
              direction: winkSignal.direction,
              startedAt: now,
              peakStrength: winkSignal.strength,
              stableFrames: 1,
              missingFrames: 0,
              openFrames: 0,
            };
            updateWinkDebug({
              status: "추적 시작",
              detail: `${winkSignal.eye === "left" ? "왼쪽" : "오른쪽"} 눈 감기를 후보로 잡았습니다.`,
              trackingPhase: "tracking",
              trackingEye: winkSignal.eye,
              stableFrames: 1,
              openFrames: 0,
            });
            updateGestureText("윙크 확인 중");
            return;
          }

          winkState.missingFrames = 0;
          winkState.stableFrames = Math.min(winkState.stableFrames + 1, WINK_STABLE_FRAME_TARGET);
          winkState.peakStrength = Math.max(winkState.peakStrength, winkSignal.strength);
          winkState.direction = winkSignal.direction;

          const holdDuration = winkState.startedAt ? now - winkState.startedAt : 0;
          updateWinkDebug({
            status: "성공 직전",
            detail: "한쪽 눈 감기가 유지되고 있어 페이지 넘김 조건을 채우는 중입니다.",
            trackingPhase: winkState.phase,
            trackingEye: winkState.eye,
            stableFrames: winkState.stableFrames,
            openFrames: winkState.openFrames,
          });
          updateGestureText(
            winkState.direction === "none"
              ? "윙크 확인 중"
              : getPendingGestureText(winkState.direction, holdDuration)
          );

          if (
            winkState.stableFrames >= WINK_STABLE_FRAME_TARGET &&
            holdDuration >= WINK_HOLD_DURATION_MS &&
            winkState.peakStrength >= WINK_MIN_PEAK_STRENGTH &&
            winkState.direction !== "none"
          ) {
            const didTurnPage = handleGesture(winkState.direction);
            if (didTurnPage) {
              resetWinkState("waiting-open");
              updateWinkDebug({
                status: "성공",
                detail: "윙크가 확정되어 페이지를 넘겼습니다. 다음 감지를 위해 눈을 다시 떠 주세요.",
                trackingPhase: "waiting-open",
                trackingEye: "none",
                stableFrames: 0,
                openFrames: 0,
              });
            } else {
              resetWinkState();
              updateWinkDebug({
                status: "실패: 페이지 이동 없음",
                detail: "쿨다운 중이거나 첫/마지막 페이지라 실제 페이지 이동이 일어나지 않았습니다.",
                trackingPhase: "idle",
                trackingEye: "none",
                stableFrames: 0,
                openFrames: 0,
              });
            }
          }
        });

        const camera = new Camera(videoRef.current, {
          onFrame: async () => {
            if (!videoRef.current || cancelled) {
              return;
            }

            await faceMesh.send({ image: videoRef.current });
          },
          width: 640,
          height: 480,
          facingMode: "user",
        });

        cameraInstanceRef.current = camera;
        await camera.start();
        setCameraPermission("granted");

        if (!cancelled) {
          updateGestureText("정면을 1초만 봐 주세요");
        }
      } catch (error) {
        console.error("mediapipe init error", error);
        await stopCamera();
        if (!cancelled) {
          if (error instanceof DOMException && error.name === "NotAllowedError") {
            setCameraPermission("denied");
          }
          setCameraEnabled(false);
          void writeReaderValue(MOTION_DETECTION_ENABLED_KEY, false);
          updateGestureText("카메라 권한 허용이 필요합니다.");
        }
      }
    };

    void initFaceMesh();

    return () => {
      cancelled = true;
      void stopCamera();
    };
  }, [cameraEnabled, cameraPermission]);

  const onFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await loadPdfDocument({
        blob: file,
        fileName: file.name,
        initialPage: 1,
        savedScoreId: null,
      });
      const saved = await persistCurrentPdf();
      updateGestureText(saved ? "PDF를 불러오고 자동 저장했습니다." : "PDF를 불러왔습니다.");
    } catch (error) {
      updateGestureText("PDF 라이브러리 로드 실패");
      console.error("pdfjs load error", error);
    } finally {
      event.target.value = "";
    }
  };

  const renderPage = async (pdf: any, pageNumber: number, totalPages: number) => {
    if (!canvasRef.current) return;
    if (isRenderingRef.current) {
      pendingPageRef.current = pageNumber;
      return;
    }

    isRenderingRef.current = true;

    const page = await pdf.getPage(pageNumber);
    const viewport = page.getViewport({ scale: 1.4 });
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      isRenderingRef.current = false;
      return;
    }

    canvas.width = viewport.width;
    canvas.height = viewport.height;

    await page.render({ canvasContext: ctx, viewport }).promise;
    isRenderingRef.current = false;

    setPageText(`페이지: ${pageNumber} / ${totalPages}`);

    if (pendingPageRef.current !== null) {
      const next = pendingPageRef.current;
      pendingPageRef.current = null;
      renderPage(pdf, next, totalPages);
    }
  };

  const queuePage = (pageNumber: number) => {
    const pdf = pdfDocRef.current;
    const totalPages = pageCountRef.current;
    const activePage = currentPageRef.current;

    if (!pdf || totalPages === 0) return;

    const safePage = Math.max(1, Math.min(totalPages, pageNumber));
    if (safePage === activePage) return;

    currentPageRef.current = safePage;
    setCurrentPage(safePage);
    void renderPage(pdf, safePage, totalPages);
    void saveSessionState({ currentPage: safePage });

    if (currentSavedScoreIdRef.current) {
      const nextScores = savedScores.map((score) =>
        score.id === currentSavedScoreIdRef.current ? { ...score, currentPage: safePage, updatedAt: Date.now() } : score
      );
      void saveScoresToDb(nextScores);
    }
  };

  const persistCurrentPdf = async () => {
    if (!currentPdfBlobRef.current) {
      return false;
    }

    const nextScore: SavedScore = {
      id: currentSavedScoreIdRef.current ?? crypto.randomUUID(),
      name: currentFileNameRef.current ?? `악보 ${savedScores.length + 1}`,
      pdfBlob: currentPdfBlobRef.current,
      currentPage: currentPageRef.current,
      updatedAt: Date.now(),
    };

    const nextScores = currentSavedScoreIdRef.current
      ? savedScores.map((score) => (score.id === nextScore.id ? nextScore : score))
      : [nextScore, ...savedScores];

    currentSavedScoreIdRef.current = nextScore.id;
    await saveScoresToDb(nextScores);
    await saveSessionState({ savedScoreId: nextScore.id });
    return true;
  };

  const openSavedScore = async (score: SavedScore) => {
    try {
      await loadPdfDocument({
        blob: score.pdfBlob,
        fileName: score.name,
        initialPage: score.currentPage,
        savedScoreId: score.id,
      });

      const nextScores = savedScores.map((item) =>
        item.id === score.id ? { ...item, updatedAt: Date.now() } : item
      );
      await saveScoresToDb(nextScores);
      updateGestureText("저장된 악보를 불러왔습니다.");
      setIsLibraryOpen(false);
    } catch (error) {
      console.error("saved score open error", error);
      updateGestureText("저장된 악보를 불러오지 못했습니다.");
    }
  };

  const deleteSavedScore = async (scoreId: string) => {
    const nextScores = savedScores.filter((score) => score.id !== scoreId);
    await saveScoresToDb(nextScores);

    if (currentSavedScoreIdRef.current === scoreId) {
      currentSavedScoreIdRef.current = null;
      await saveSessionState({ savedScoreId: null });
    }

    updateGestureText("저장된 악보를 삭제했습니다.");
  };

  const handleCameraAction = () => {
    if (cameraEnabled) {
      setCameraEnabled(false);
      void writeReaderValue(MOTION_DETECTION_ENABLED_KEY, false);
      return;
    }

    setCameraEnabled(true);
    void writeReaderValue(MOTION_DETECTION_ENABLED_KEY, true);
    updateGestureText("모션 감지 권한 창이 뜨면 허용해 주세요.");
  };

  const handleCalibrationReset = () => {
    calibrationActiveRef.current = true;
    calibrationFramesRef.current = 0;
    calibrationLeftSumRef.current = 0;
    calibrationRightSumRef.current = 0;
    forwardStableFramesRef.current = 0;
    smoothedLeftEyeRef.current = null;
    smoothedRightEyeRef.current = null;
    resetWinkState();
    setCalibrationProgress(0);
    setHasCalibration(false);
    setIsCalibrating(true);
    updateGestureText("정면을 1초만 봐 주세요");
  };

  const revealFocusControls = () => {
    if (!isFocusMode) {
      return;
    }

    setIsFocusControlsVisible(true);

    if (focusControlsTimeoutRef.current !== null) {
      window.clearTimeout(focusControlsTimeoutRef.current);
    }

    focusControlsTimeoutRef.current = window.setTimeout(() => {
      setIsFocusControlsVisible(false);
      focusControlsTimeoutRef.current = null;
    }, 2200);
  };

  const getPendingGestureText = (direction: "left" | "right", holdDuration: number) => {
    const totalPages = pageCountRef.current;
    const currentPage = currentPageRef.current;

    if (direction === "right" && currentPage >= totalPages) {
      return "마지막 페이지";
    }

    if (direction === "left" && currentPage <= 1) {
      return "첫 페이지";
    }

    const remainingSeconds = Math.max(0, (WINK_HOLD_DURATION_MS - holdDuration) / 1000).toFixed(1);
    const directionLabel = direction === "right" ? "다음" : "이전";
    return `인식됨 ${remainingSeconds}초 후 ${directionLabel} 페이지 이동`;
  };

  const handleGesture = (direction: "left" | "right") => {
    const now = performance.now();
    if (now - lastSwitchTimeRef.current < PAGE_TURN_COOLDOWN_MS) {
      return false;
    }

    const totalPages = pageCountRef.current;
    const currentPage = currentPageRef.current;

    if (direction === "right") {
      if (currentPage >= totalPages) {
        updateGestureText("마지막 페이지");
        return false;
      } else {
        queuePage(currentPage + 1);
        updateGestureText("다음 페이지");
        lastSwitchTimeRef.current = now;
        return true;
      }
    } else {
      if (currentPage <= 1) {
        updateGestureText("첫 페이지");
        return false;
      } else {
        queuePage(currentPage - 1);
        updateGestureText("이전 페이지");
        lastSwitchTimeRef.current = now;
        return true;
      }
    }
  };

  const computeEyeRatio = (
    landmarks: NormalizedLandmark[],
    outerIndex: number,
    innerIndex: number,
    verticalPairs: Array<[number, number]>
  ) => {
    const outer = landmarks[outerIndex];
    const inner = landmarks[innerIndex];

    const horizontal = Math.hypot(outer.x - inner.x, outer.y - inner.y);
    const vertical =
      verticalPairs.reduce((sum, [upperIndex, lowerIndex]) => {
        const upper = landmarks[upperIndex];
        const lower = landmarks[lowerIndex];
        return sum + Math.hypot(upper.x - lower.x, upper.y - lower.y);
      }, 0) / verticalPairs.length;

    if (horizontal === 0) {
      return 0;
    }

    return vertical / horizontal;
  };

  const getSmoothedEyeRatio = (targetRef: React.MutableRefObject<number | null>, nextValue: number) => {
    if (targetRef.current === null) {
      targetRef.current = nextValue;
      return nextValue;
    }

    targetRef.current =
      targetRef.current * (1 - EYE_RATIO_SMOOTHING_ALPHA) + nextValue * EYE_RATIO_SMOOTHING_ALPHA;
    return targetRef.current;
  };

  const getEyeVisibilityMetrics = (landmarks: NormalizedLandmark[]) => {
    const leftEyeOuter = landmarks[33];
    const leftEyeInner = landmarks[133];
    const rightEyeInner = landmarks[362];
    const rightEyeOuter = landmarks[263];
    const noseTip = landmarks[1];

    const leftEyeWidth = Math.hypot(leftEyeOuter.x - leftEyeInner.x, leftEyeOuter.y - leftEyeInner.y);
    const rightEyeWidth = Math.hypot(rightEyeOuter.x - rightEyeInner.x, rightEyeOuter.y - rightEyeInner.y);
    const smallerEyeWidth = Math.min(leftEyeWidth, rightEyeWidth);
    const largerEyeWidth = Math.max(leftEyeWidth, rightEyeWidth, 0.0001);
    const eyeWidthBalance = smallerEyeWidth / largerEyeWidth;

    const leftEyeCenterX = (leftEyeOuter.x + leftEyeInner.x) / 2;
    const leftEyeCenterY = (leftEyeOuter.y + leftEyeInner.y) / 2;
    const rightEyeCenterX = (rightEyeOuter.x + rightEyeInner.x) / 2;
    const rightEyeCenterY = (rightEyeOuter.y + rightEyeInner.y) / 2;

    const noseToLeftEye = Math.hypot(noseTip.x - leftEyeCenterX, noseTip.y - leftEyeCenterY);
    const noseToRightEye = Math.hypot(noseTip.x - rightEyeCenterX, noseTip.y - rightEyeCenterY);
    const smallerEyeCenterDistance = Math.min(noseToLeftEye, noseToRightEye);
    const largerEyeCenterDistance = Math.max(noseToLeftEye, noseToRightEye, 0.0001);
    const eyeCenterBalance = smallerEyeCenterDistance / largerEyeCenterDistance;

    return {
      eyeWidthBalance,
      eyeCenterBalance,
    };
  };

  const isFaceFacingForward = (landmarks: NormalizedLandmark[]) => {
    const leftEyeOuter = landmarks[33];
    const rightEyeOuter = landmarks[263];
    const noseTip = landmarks[1];
    const eyeCenterX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
    const eyeCenterY = (leftEyeOuter.y + rightEyeOuter.y) / 2;
    const eyeLineTilt = Math.abs(leftEyeOuter.y - rightEyeOuter.y);
    const noseOffsetX = Math.abs(noseTip.x - eyeCenterX);
    const noseOffsetY = Math.abs(noseTip.y - eyeCenterY);
    const eyeDistanceX = Math.abs(rightEyeOuter.x - leftEyeOuter.x);

    return (
      eyeLineTilt < 0.18 &&
      noseOffsetX < 0.26 &&
      noseOffsetY < 0.5 &&
      eyeDistanceX > 0.03
    );
  };

  const isFaceForwardEnoughForWink = (landmarks: NormalizedLandmark[]) => {
    const leftEyeOuter = landmarks[33];
    const rightEyeOuter = landmarks[263];
    const noseTip = landmarks[1];
    const eyeCenterX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
    const eyeCenterY = (leftEyeOuter.y + rightEyeOuter.y) / 2;
    const eyeLineTilt = Math.abs(leftEyeOuter.y - rightEyeOuter.y);
    const noseOffsetX = Math.abs(noseTip.x - eyeCenterX);
    const noseOffsetY = Math.abs(noseTip.y - eyeCenterY);
    const eyeDistanceX = Math.abs(rightEyeOuter.x - leftEyeOuter.x);
    const { eyeWidthBalance, eyeCenterBalance } = getEyeVisibilityMetrics(landmarks);

    return (
      eyeLineTilt < ACTIVE_FORWARD_TILT_MAX &&
      noseOffsetX < ACTIVE_FORWARD_NOSE_X_MAX &&
      noseOffsetY < ACTIVE_FORWARD_NOSE_Y_MAX &&
      eyeDistanceX > ACTIVE_FORWARD_EYE_DISTANCE_MIN &&
      eyeWidthBalance >= ACTIVE_FORWARD_EYE_WIDTH_BALANCE_MIN &&
      eyeCenterBalance >= ACTIVE_FORWARD_EYE_CENTER_BALANCE_MIN
    );
  };

  const isFaceForwardEnoughForCalibration = (landmarks: NormalizedLandmark[]) => {
    const leftEyeOuter = landmarks[33];
    const rightEyeOuter = landmarks[263];
    const noseTip = landmarks[1];
    const eyeCenterX = (leftEyeOuter.x + rightEyeOuter.x) / 2;
    const eyeCenterY = (leftEyeOuter.y + rightEyeOuter.y) / 2;
    const eyeLineTilt = Math.abs(leftEyeOuter.y - rightEyeOuter.y);
    const noseOffsetX = Math.abs(noseTip.x - eyeCenterX);
    const noseOffsetY = Math.abs(noseTip.y - eyeCenterY);
    const eyeDistanceX = Math.abs(rightEyeOuter.x - leftEyeOuter.x);
    return (
      eyeLineTilt < CALIBRATION_FORWARD_TILT_MAX &&
      noseOffsetX < CALIBRATION_FORWARD_NOSE_X_MAX &&
      noseOffsetY < CALIBRATION_FORWARD_NOSE_Y_MAX &&
      eyeDistanceX > CALIBRATION_EYE_DISTANCE_MIN
    );
  };

  const computeWink = (
    leftEyeRatio: number,
    rightEyeRatio: number,
    activeCandidate: "left" | "right" | "none"
  ): { eye: "left" | "right" | "none"; direction: "left" | "right" | "none"; strength: number } => {
    const leftOpenBaseline = Math.max(leftEyeOpenRef.current, 0.0001);
    const rightOpenBaseline = Math.max(rightEyeOpenRef.current, 0.0001);
    const leftNormalized = leftEyeRatio / leftOpenBaseline;
    const rightNormalized = rightEyeRatio / rightOpenBaseline;
    const leftClosure = 1 - leftNormalized;
    const rightClosure = 1 - rightNormalized;
    const leftStrength = leftClosure - rightClosure;
    const rightStrength = rightClosure - leftClosure;

    const isLeftStart =
      leftNormalized < WINK_START_CLOSED_RATIO_THRESHOLD &&
      rightNormalized > WINK_START_OTHER_EYE_OPEN_RATIO_THRESHOLD &&
      leftStrength > WINK_START_CLOSURE_GAP_THRESHOLD;
    const isRightStart =
      rightNormalized < WINK_START_CLOSED_RATIO_THRESHOLD &&
      leftNormalized > WINK_START_OTHER_EYE_OPEN_RATIO_THRESHOLD &&
      rightStrength > WINK_START_CLOSURE_GAP_THRESHOLD;

    const isLeftContinue =
      leftNormalized < WINK_CONTINUE_CLOSED_RATIO_THRESHOLD &&
      rightNormalized > WINK_CONTINUE_OTHER_EYE_OPEN_RATIO_THRESHOLD &&
      leftStrength > WINK_CONTINUE_CLOSURE_GAP_THRESHOLD;
    const isRightContinue =
      rightNormalized < WINK_CONTINUE_CLOSED_RATIO_THRESHOLD &&
      leftNormalized > WINK_CONTINUE_OTHER_EYE_OPEN_RATIO_THRESHOLD &&
      rightStrength > WINK_CONTINUE_CLOSURE_GAP_THRESHOLD;

    if (activeCandidate === "left" && isLeftContinue) {
      return { eye: "left", direction: isWinkMirrored ? "right" : "left", strength: leftStrength };
    }

    if (activeCandidate === "right" && isRightContinue) {
      return { eye: "right", direction: isWinkMirrored ? "left" : "right", strength: rightStrength };
    }

    if (isRightStart) {
      return { eye: "right", direction: isWinkMirrored ? "left" : "right", strength: rightStrength };
    }

    if (isLeftStart) {
      return { eye: "left", direction: isWinkMirrored ? "right" : "left", strength: leftStrength };
    }

    return { eye: "none", direction: "none", strength: 0 };
  };

  const areEyesRecovered = (leftEyeRatio: number, rightEyeRatio: number) => {
    const leftOpenBaseline = Math.max(leftEyeOpenRef.current, 0.0001);
    const rightOpenBaseline = Math.max(rightEyeOpenRef.current, 0.0001);
    const leftNormalized = leftEyeRatio / leftOpenBaseline;
    const rightNormalized = rightEyeRatio / rightOpenBaseline;

    return leftNormalized >= RECOVERY_OPEN_RATIO_THRESHOLD && rightNormalized >= RECOVERY_OPEN_RATIO_THRESHOLD;
  };

  const areEyesRecoveredFast = (leftEyeRatio: number, rightEyeRatio: number) => {
    const leftOpenBaseline = Math.max(leftEyeOpenRef.current, 0.0001);
    const rightOpenBaseline = Math.max(rightEyeOpenRef.current, 0.0001);
    const leftNormalized = leftEyeRatio / leftOpenBaseline;
    const rightNormalized = rightEyeRatio / rightOpenBaseline;

    return (
      leftNormalized >= RECOVERY_OPEN_RATIO_FAST_THRESHOLD &&
      rightNormalized >= RECOVERY_OPEN_RATIO_FAST_THRESHOLD
    );
  };

  const areBothEyesClosed = (
    leftEyeRatio: number,
    rightEyeRatio: number,
    dominantWinkStrength: number,
    normalizedGap: number
  ) => {
    const leftOpenBaseline = Math.max(leftEyeOpenRef.current, 0.0001);
    const rightOpenBaseline = Math.max(rightEyeOpenRef.current, 0.0001);
    const leftNormalized = leftEyeRatio / leftOpenBaseline;
    const rightNormalized = rightEyeRatio / rightOpenBaseline;

    return (
      leftNormalized < BOTH_EYES_CLOSED_RATIO_THRESHOLD &&
      rightNormalized < BOTH_EYES_CLOSED_RATIO_THRESHOLD &&
      leftNormalized + rightNormalized < BOTH_EYES_CLOSED_COMBINED_THRESHOLD &&
      normalizedGap <= BOTH_EYES_SIMILARITY_GAP_THRESHOLD &&
      dominantWinkStrength <= BOTH_EYES_DOMINANCE_STRENGTH_MAX
    );
  };

  const areBothEyesClearlyVisible = (landmarks: NormalizedLandmark[]) => {
    const { eyeWidthBalance, eyeCenterBalance } = getEyeVisibilityMetrics(landmarks);

    return (
      eyeWidthBalance >= EYE_VISIBILITY_BALANCE_THRESHOLD &&
      eyeCenterBalance >= EYE_CENTER_BALANCE_THRESHOLD
    );
  };

  const canGoPrevious = currentPage > 1;
  const canGoNext = pageCount > 0 && currentPage < pageCount;
  const hasLoadedPdf = pdfDoc !== null;
  const cameraPermissionOn = cameraPermission === "granted";
  const isCameraReady = cameraEnabled && cameraPermissionOn;
  const cameraStatusText = isCalibrating
    ? `${calibrationProgress}%`
    : isCameraReady
      ? hasCalibration
        ? "준비됨"
        : "보정 필요"
      : cameraPermission === "denied"
        ? "권한 필요"
        : "대기 중";
  const cameraButtonText = cameraEnabled
    ? "모션 감지 끄기"
    : cameraPermission === "denied"
      ? "모션 감지 권한 확인"
      : "모션 감지 켜기";
  const mirrorModeTitle = isWinkMirrored ? "전면카메라 기준" : "일반 카메라 기준";
  const mirrorModeHint = isWinkMirrored ? "거울처럼 좌우를 해석" : "실제 좌우 그대로 해석";
  const viewerTitle = hasLoadedPdf ? `${currentFileName ?? "불러온 악보"} (${currentPage}/${pageCount})` : "악보";
  const formatDebugNumber = (value: number | null) => (value === null ? "-" : value.toFixed(2));
  const gestureOverlayText = (() => {
    if (gestureText.includes("인식됨") || gestureText === "윙크 인식 중") {
      return gestureText;
    }

    if (gestureText === "다음 페이지" || gestureText === "이전 페이지") {
      return gestureText;
    }

    if (gestureText === "마지막 페이지" || gestureText === "첫 페이지") {
      return gestureText;
    }

    return null;
  })();
  const motionGuideSection = (
    <section className="motion-guide-card">
      <button
        type="button"
        className="motion-guide-toggle"
        onClick={() => setIsMotionGuideExpanded((prev) => !prev)}
        aria-expanded={isMotionGuideExpanded}
      >
        <h3>사용법</h3>
        <span className="motion-guide-toggle-text">{isMotionGuideExpanded ? "접기" : "펼치기"}</span>
      </button>
      {isMotionGuideExpanded ? (
        <div className="motion-guide-content">
          <ol className="motion-guide-list">
            <li>
              <span className="motion-guide-step">1.</span>
              <div>
                <strong>모션 감지 켜기</strong>
                <span>상단 버튼을 눌러 카메라를 시작합니다.</span>
              </div>
            </li>
            <li>
              <span className="motion-guide-step">2.</span>
              <div>
                <strong>정면 보고 기본 보정</strong>
                <span>처음 켜면 잠깐 정면을 보고 기본 보정을 진행합니다.</span>
              </div>
            </li>
            <li>
              <span className="motion-guide-step">3.</span>
              <div>
                <strong>오른쪽 윙크 0.3초 이상</strong>
                <span className="motion-guide-result">다음 페이지</span>
              </div>
            </li>
            <li>
              <span className="motion-guide-step">4.</span>
              <div>
                <strong>왼쪽 윙크 0.3초 이상</strong>
                <span className="motion-guide-result">이전 페이지</span>
              </div>
            </li>
          </ol>
        </div>
      ) : null}
    </section>
  );

  return (
    <main className={`app-shell ${isFocusMode ? "is-focus-mode" : ""} ${isFocusControlsVisible ? "show-focus-controls" : ""}`}>
      <header className="topbar">
        <div className="topbar-actions">
          <div className="topbar-primary-actions">
            <label className="file-label primary-upload-action">
              <strong>PDF 불러오기</strong>
              <input type="file" accept="application/pdf" onChange={onFileChange} />
            </label>
          </div>

          <button
            type="button"
            className={`camera-toggle compact-action ${isCameraReady ? "is-on" : "is-off"}`}
            onClick={handleCameraAction}
            aria-pressed={cameraEnabled}
          >
            {cameraButtonText}
            <span>{cameraStatusText}</span>
          </button>
        </div>
      </header>

      {cameraPermission === "denied" ? <p className="permission-help">카메라 권한이 차단됨. 주소창에서 허용해 주세요.</p> : null}

      <section className="panel-card compact-panel mobile-motion-guide">{motionGuideSection}</section>

      <section className="workspace-grid">
        <div className="viewer-card" onMouseMove={isFocusMode ? revealFocusControls : undefined}>
          {gestureOverlayText ? <div className="gesture-overlay-text">{gestureOverlayText}</div> : null}
          <div className="viewer-toolbar">
            <div className="viewer-heading">
              <h2>{viewerTitle}</h2>
            </div>
            <div className="page-buttons" aria-label="페이지 이동 버튼">
              <button
                type="button"
                className="library-toggle-button"
                onClick={() => setIsLibraryOpen((prev) => !prev)}
                aria-expanded={isLibraryOpen}
              >
                저장된 악보 {isLibraryOpen ? "숨기기" : "보기"}
              </button>
              <button type="button" onClick={() => queuePage(currentPage - 1)} disabled={!canGoPrevious}>
                <span aria-hidden="true">←</span>
                <span className="sr-only">이전 페이지</span>
              </button>
              <button type="button" onClick={() => queuePage(currentPage + 1)} disabled={!canGoNext}>
                <span aria-hidden="true">→</span>
                <span className="sr-only">다음 페이지</span>
              </button>
              <button type="button" onClick={() => setIsFocusMode((prev) => !prev)}>
                <span aria-hidden="true">{isFocusMode ? "⤡" : "⤢"}</span>
                <span className="sr-only">{isFocusMode ? "전체보기 종료" : "전체보기"}</span>
              </button>
            </div>
          </div>

          <div className="pdf-viewer">
            <button
              type="button"
              className="focus-controls-hit-area"
              onClick={revealFocusControls}
              aria-label="전체화면 컨트롤 보기"
            />
            <canvas ref={canvasRef} />
            {!hasLoadedPdf ? (
              <div className="empty-viewer">
                <strong>PDF를 불러오세요</strong>
              </div>
            ) : null}
          </div>
        </div>

        <aside className="side-panel">
          <div className="panel-card compact-panel">
            <h2>옵션</h2>
            <div className="option-buttons">
              <button
                type="button"
                className="secondary-button"
                onClick={handleCalibrationReset}
                disabled={!isCameraReady}
              >
                윙크 보정 다시하기
              </button>
              <button
                type="button"
                className={`secondary-button ${isWinkMirrored ? "is-active" : ""}`}
                onClick={() => {
                  const next = !isWinkMirrored;
                  setIsWinkMirrored(next);
                  void writeReaderValue(WINK_MIRROR_KEY, next);
                }}
                aria-pressed={isWinkMirrored}
              >
                인식 방향: {mirrorModeTitle} ({mirrorModeHint})
              </button>
            </div>

            <div className="desktop-motion-guide">{motionGuideSection}</div>
          </div>

          <section className="panel-card compact-panel wink-debug-panel" aria-label="윙크 감지 디버그">
            <h2>감지 디버그</h2>
            <p className="wink-debug-status">{winkDebug.status}</p>
            <p className="wink-debug-detail">{winkDebug.detail}</p>
            <dl className="wink-debug-grid">
              <div>
                <dt>왼눈 비율</dt>
                <dd>{formatDebugNumber(winkDebug.leftNormalized)}</dd>
              </div>
              <div>
                <dt>오른눈 비율</dt>
                <dd>{formatDebugNumber(winkDebug.rightNormalized)}</dd>
              </div>
              <div>
                <dt>왼눈 강도</dt>
                <dd>{winkDebug.leftStrength.toFixed(2)}</dd>
              </div>
              <div>
                <dt>오른눈 강도</dt>
                <dd>{winkDebug.rightStrength.toFixed(2)}</dd>
              </div>
              <div>
                <dt>정면 프레임</dt>
                <dd>{winkDebug.forwardStableFrames}</dd>
              </div>
              <div>
                <dt>추적 단계</dt>
                <dd>{winkDebug.trackingPhase}</dd>
              </div>
              <div>
                <dt>후보 눈</dt>
                <dd>{winkDebug.trackingEye}</dd>
              </div>
              <div>
                <dt>안정 프레임</dt>
                <dd>{winkDebug.stableFrames}</dd>
              </div>
              <div>
                <dt>재무장 프레임</dt>
                <dd>{winkDebug.openFrames}</dd>
              </div>
            </dl>
          </section>

          <video ref={videoRef} autoPlay muted playsInline className="camera-feed-hidden" />

          {isLibraryOpen ? (
            <section className="saved-library" aria-label="저장된 악보 목록">
              {savedScores.length === 0 ? (
                <p className="library-empty">저장된 악보가 아직 없습니다.</p>
              ) : (
                <ul className="saved-score-list">
                  {savedScores.map((score) => (
                    <li key={score.id} className="saved-score-item">
                      <button type="button" className="saved-score-open" onClick={() => openSavedScore(score)}>
                        <strong>{score.name}</strong>
                        <span>마지막 페이지 {score.currentPage}</span>
                      </button>
                      <button type="button" className="saved-score-delete" onClick={() => void deleteSavedScore(score.id)}>
                        삭제
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          ) : null}
        </aside>
      </section>
    </main>
  );
}
