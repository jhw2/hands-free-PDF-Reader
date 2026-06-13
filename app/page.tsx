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
const WINK_CALIBRATION_PROFILE_KEY = "wink-calibration-profile";
const OPEN_CALIBRATION_SAMPLE_TARGET = 18;
const WINK_CALIBRATION_SAMPLE_TARGET = 12;
const OPEN_CALIBRATION_NOISE_MAX = 0.075;
const WINK_CALIBRATION_CLOSED_RATIO_MAX = 0.78;
const WINK_CALIBRATION_OTHER_EYE_OPEN_MIN = 0.72;
const WINK_CALIBRATION_STRENGTH_MIN = 0.12;
const WINK_HOLD_DURATION_MS = 250;
const PAGE_TURN_COOLDOWN_MS = 1200;
const EYE_RATIO_SMOOTHING_ALPHA = 0.55;
const WINK_CANDIDATE_CLOSED_RATIO_THRESHOLD = 0.88;
const WINK_CANDIDATE_OTHER_EYE_OPEN_RATIO_THRESHOLD = 0.72;
const WINK_CANDIDATE_CLOSURE_GAP_THRESHOLD = 0.08;
const WINK_CONFIRM_CLOSED_RATIO_THRESHOLD = 0.82;
const WINK_CONFIRM_OTHER_EYE_OPEN_RATIO_THRESHOLD = 0.74;
const WINK_CONFIRM_CLOSURE_GAP_THRESHOLD = 0.12;
const WINK_CONTINUE_CLOSED_RATIO_THRESHOLD = 0.84;
const WINK_CONTINUE_OTHER_EYE_OPEN_RATIO_THRESHOLD = 0.72;
const WINK_CONTINUE_CLOSURE_GAP_THRESHOLD = 0.12;
const WINK_MIN_PEAK_STRENGTH = 0.12;
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
const ACTIVE_FORWARD_NOSE_X_MAX = 0.20;
const ACTIVE_FORWARD_NOSE_Y_MAX = 0.4;
const ACTIVE_FORWARD_EYE_DISTANCE_MIN = 0.04;
const HEAD_TURN_EYE_BALANCE_THRESHOLD = 0.55;
const FORWARD_STABLE_FRAME_TARGET = 2;
const REARM_OPEN_FRAME_TARGET = 3;
const BOTH_EYES_SIMILARITY_GAP_THRESHOLD = 0.025;
const BOTH_EYES_PARTIAL_CLOSED_THRESHOLD = 0.82;
const BOTH_EYES_DOMINANCE_STRENGTH_MAX = 0.1;
const BOTH_EYES_BLINK_RATIO_THRESHOLD = 0.84;
const BOTH_EYES_BLINK_SIMILARITY_GAP_MAX = 0.16;
const BOTH_EYES_BLINK_DOMINANCE_MAX = 0.18;
const BOTH_EYES_ASYMMETRIC_BLINK_CLOSED_RATIO_THRESHOLD = 0.82;
const BOTH_EYES_ASYMMETRIC_BLINK_OPEN_RATIO_THRESHOLD = 0.80;
const BOTH_EYES_ASYMMETRIC_BLINK_COMBINED_THRESHOLD = 1.58;
const WINK_STABLE_FRAME_TARGET = 2;
const WINK_CONFIRM_FRAME_TARGET = 2;
const WINK_MISSING_FRAME_TOLERANCE = 4;
const MOUTH_OPEN_RATIO_THRESHOLD = 0.36;
const MOUTH_CLOSE_RATIO_THRESHOLD = 0.18;
const MOUTH_HOLD_DURATION_MS = 500;
const MOUTH_MIN_OPEN_MS = 120;
const MOUTH_DOUBLE_TAP_WINDOW_MS = 600;
const MOUTH_STABLE_FRAME_TARGET = 2;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

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
type WinkPhase = "idle" | "candidate" | "tracking" | "waiting-open";
type WinkSignalStage = "none" | "candidate" | "confirmed";
type CalibrationStep = "idle" | "open" | "left-wink" | "right-wink";
type MouthPhase = "idle" | "open" | "first-closed" | "waiting-close";

type MouthDetectionState = {
  phase: MouthPhase;
  startedAt: number | null;
  closedAt: number | null;
  stableFrames: number;
};

type WinkDetectionState = {
  phase: WinkPhase;
  eye: WinkEye;
  direction: WinkDirection;
  startedAt: number | null;
  peakStrength: number;
  stableFrames: number;
  confirmedFrames: number;
  missingFrames: number;
  openFrames: number;
};

type WinkSignal = {
  eye: WinkEye;
  direction: WinkDirection;
  strength: number;
  stage: WinkSignalStage;
};

type WinkCalibrationProfile = {
  closedRatio: number;
  strength: number;
  otherEyeOpenRatio: number;
  targetNoise: number;
};

type SavedWinkCalibrationProfile = {
  version: 1;
  updatedAt: number;
  leftEyeOpen: number;
  rightEyeOpen: number;
  leftEyeNoise: number;
  rightEyeNoise: number;
  leftWinkProfile: WinkCalibrationProfile;
  rightWinkProfile: WinkCalibrationProfile;
};

type WinkDebugState = {
  status: string;
  detail: string;
  leftRawRatio: number | null;
  rightRawRatio: number | null;
  leftNormalized: number | null;
  rightNormalized: number | null;
  leftBaseline: number;
  rightBaseline: number;
  leftNoise: number;
  rightNoise: number;
  leftStrength: number;
  rightStrength: number;
  signalStage: WinkSignalStage;
  forwardStableFrames: number;
  trackingPhase: WinkPhase;
  trackingEye: WinkEye;
  stableFrames: number;
  confirmedFrames: number;
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
  const [calibrationStep, setCalibrationStep] = useState<CalibrationStep>("idle");
  const [calibrationFeedback, setCalibrationFeedback] = useState("카메라를 보고 편하게 준비해 주세요.");
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [hasCalibration, setHasCalibration] = useState(false);
  const [winkDebug, setWinkDebug] = useState<WinkDebugState>({
    status: "대기 중",
    detail: "모션 감지를 켜면 실패 이유가 여기에 표시됩니다.",
    leftRawRatio: null,
    rightRawRatio: null,
    leftNormalized: null,
    rightNormalized: null,
    leftBaseline: 0.24,
    rightBaseline: 0.24,
    leftNoise: 0,
    rightNoise: 0,
    leftStrength: 0,
    rightStrength: 0,
    signalStage: "none",
    forwardStableFrames: 0,
    trackingPhase: "idle",
    trackingEye: "none",
    stableFrames: 0,
    confirmedFrames: 0,
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
  const isWinkMirroredRef = useRef(true);
  const winkStateRef = useRef<WinkDetectionState>({
    phase: "idle",
    eye: "none",
    direction: "none",
    startedAt: null,
    peakStrength: 0,
    stableFrames: 0,
    confirmedFrames: 0,
    missingFrames: 0,
    openFrames: 0,
  });
  const gestureTextRef = useRef("PDF를 먼저 불러오면 다음 단계가 쉬워집니다.");
  const calibrationActiveRef = useRef(false);
  const calibrationStepRef = useRef<CalibrationStep>("idle");
  const calibrationFramesRef = useRef(0);
  const calibrationLeftSumRef = useRef(0);
  const calibrationRightSumRef = useRef(0);
  const calibrationLeftSquareSumRef = useRef(0);
  const calibrationRightSquareSumRef = useRef(0);
  const winkCalibrationTargetSumRef = useRef(0);
  const winkCalibrationTargetSquareSumRef = useRef(0);
  const winkCalibrationStrengthSumRef = useRef(0);
  const winkCalibrationOtherOpenSumRef = useRef(0);
  const forwardStableFramesRef = useRef(0);
  const leftEyeOpenRef = useRef(0.24);
  const rightEyeOpenRef = useRef(0.24);
  const leftEyeNoiseRef = useRef(0);
  const rightEyeNoiseRef = useRef(0);
  const leftWinkProfileRef = useRef<WinkCalibrationProfile | null>(null);
  const rightWinkProfileRef = useRef<WinkCalibrationProfile | null>(null);
  const savedCalibrationProfileRef = useRef<SavedWinkCalibrationProfile | null>(null);
  const smoothedLeftEyeRef = useRef<number | null>(null);
  const smoothedRightEyeRef = useRef<number | null>(null);
  const smoothedMouthRef = useRef<number | null>(null);
  const mouthStateRef = useRef<MouthDetectionState>({
    phase: "idle",
    startedAt: null,
    closedAt: null,
    stableFrames: 0,
  });
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
      confirmedFrames: 0,
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

  const resetOpenCalibrationSamples = () => {
    calibrationFramesRef.current = 0;
    calibrationLeftSumRef.current = 0;
    calibrationRightSumRef.current = 0;
    calibrationLeftSquareSumRef.current = 0;
    calibrationRightSquareSumRef.current = 0;
  };

  const resetWinkCalibrationSamples = () => {
    calibrationFramesRef.current = 0;
    winkCalibrationTargetSumRef.current = 0;
    winkCalibrationTargetSquareSumRef.current = 0;
    winkCalibrationStrengthSumRef.current = 0;
    winkCalibrationOtherOpenSumRef.current = 0;
  };

  const setActiveCalibrationStep = (nextStep: CalibrationStep) => {
    calibrationStepRef.current = nextStep;
    setCalibrationStep(nextStep);
  };

  const updateCalibrationProgress = (step: CalibrationStep, frames: number) => {
    const stepIndex = step === "left-wink" ? 1 : step === "right-wink" ? 2 : 0;
    const target = step === "open" ? OPEN_CALIBRATION_SAMPLE_TARGET : WINK_CALIBRATION_SAMPLE_TARGET;
    const stageProgress = clamp(frames / target, 0, 1);
    setCalibrationProgress(Math.round(((stepIndex + stageProgress) / 3) * 100));
  };

  const resetCalibrationState = () => {
    calibrationActiveRef.current = false;
    setActiveCalibrationStep("idle");
    resetOpenCalibrationSamples();
    resetWinkCalibrationSamples();
    forwardStableFramesRef.current = 0;
    leftEyeOpenRef.current = 0.24;
    rightEyeOpenRef.current = 0.24;
    leftEyeNoiseRef.current = 0;
    rightEyeNoiseRef.current = 0;
    leftWinkProfileRef.current = null;
    rightWinkProfileRef.current = null;
    smoothedLeftEyeRef.current = null;
    smoothedRightEyeRef.current = null;
    resetWinkState();
    setCalibrationFeedback("카메라를 보고 편하게 준비해 주세요.");
    setCalibrationProgress(0);
    setHasCalibration(false);
  };

  const startPersonalCalibration = () => {
    resetCalibrationState();
    calibrationActiveRef.current = true;
    setActiveCalibrationStep("open");
    setIsCalibrating(true);
    setCalibrationFeedback("정면을 보고 양쪽 눈을 편하게 떠 주세요.");
    updateGestureText("정면 보정 중");
  };

  const getCurrentCalibrationProfile = (): SavedWinkCalibrationProfile | null => {
    if (!leftWinkProfileRef.current || !rightWinkProfileRef.current) {
      return null;
    }

    return {
      version: 1,
      updatedAt: Date.now(),
      leftEyeOpen: leftEyeOpenRef.current,
      rightEyeOpen: rightEyeOpenRef.current,
      leftEyeNoise: leftEyeNoiseRef.current,
      rightEyeNoise: rightEyeNoiseRef.current,
      leftWinkProfile: leftWinkProfileRef.current,
      rightWinkProfile: rightWinkProfileRef.current,
    };
  };

  const applySavedCalibrationProfile = (profile: SavedWinkCalibrationProfile) => {
    savedCalibrationProfileRef.current = profile;
    leftEyeOpenRef.current = profile.leftEyeOpen;
    rightEyeOpenRef.current = profile.rightEyeOpen;
    leftEyeNoiseRef.current = profile.leftEyeNoise;
    rightEyeNoiseRef.current = profile.rightEyeNoise;
    leftWinkProfileRef.current = profile.leftWinkProfile;
    rightWinkProfileRef.current = profile.rightWinkProfile;
    calibrationActiveRef.current = false;
    setActiveCalibrationStep("idle");
    setCalibrationProgress(100);
    setHasCalibration(true);
  };

  const saveCurrentCalibrationProfile = async () => {
    const profile = getCurrentCalibrationProfile();

    if (!profile) {
      return;
    }

    savedCalibrationProfileRef.current = profile;
    await writeReaderValue(WINK_CALIBRATION_PROFILE_KEY, profile);
  };

  const moveToWinkCalibrationStep = (nextStep: "left-wink" | "right-wink") => {
    resetWinkCalibrationSamples();
    setActiveCalibrationStep(nextStep);
    updateCalibrationProgress(nextStep, 0);
    resetWinkState();

    const eyeLabel = nextStep === "left-wink" ? "왼쪽" : "오른쪽";
    setCalibrationFeedback(`${eyeLabel} 눈만 연주 중처럼 편하게 감아 주세요.`);
    updateGestureText(`${eyeLabel} 눈 보정 중`);
    updateWinkDebug({
      status: `${eyeLabel} 눈 보정`,
      detail: "한쪽 눈만 편하게 감은 프레임이 충분히 모이면 자동으로 다음 단계로 넘어갑니다.",
      trackingPhase: "idle",
      trackingEye: "none",
      stableFrames: 0,
      confirmedFrames: 0,
      openFrames: 0,
    });
  };

  const getCalibrationStepLabel = (step: "left-wink" | "right-wink") => (step === "left-wink" ? "왼쪽" : "오른쪽");

  const getOppositeCalibrationStepLabel = (step: "left-wink" | "right-wink") =>
    step === "left-wink" ? "오른쪽" : "왼쪽";

  const getCalibrationTargetEye = (step: "left-wink" | "right-wink"): "left" | "right" => {
    const requestedEye = step === "left-wink" ? "left" : "right";

    if (!isWinkMirroredRef.current) {
      return requestedEye;
    }

    return requestedEye === "left" ? "right" : "left";
  };

  const completePersonalCalibration = () => {
    calibrationActiveRef.current = false;
    setActiveCalibrationStep("idle");
    setIsCalibrating(false);
    setHasCalibration(true);
    setCalibrationProgress(100);
    setCalibrationFeedback("개인 윙크 보정이 완료됐습니다.");
    resetWinkState();
    updateGestureText("개인 윙크 보정 완료");
    updateWinkDebug({
      status: "보정 완료",
      detail: "정면, 왼쪽 윙크, 오른쪽 윙크 기준값을 모두 저장했습니다.",
      trackingPhase: "idle",
      trackingEye: "none",
      stableFrames: 0,
      confirmedFrames: 0,
      openFrames: 0,
    });
    void saveCurrentCalibrationProfile().catch((error) => {
      console.error("wink calibration save error", error);
    });
  };

  const getCalibrationWinkSample = ({
    step,
    leftNormalized,
    rightNormalized,
    leftStrength,
    rightStrength,
    areBothEyesBlinking,
    areBothEyesAsymmetricallyBlinking,
    areBothEyesPartiallyClosed,
  }: {
    step: "left-wink" | "right-wink";
    leftNormalized: number;
    rightNormalized: number;
    leftStrength: number;
    rightStrength: number;
    areBothEyesBlinking: boolean;
    areBothEyesAsymmetricallyBlinking: boolean;
    areBothEyesPartiallyClosed: boolean;
  }) => {
    const targetEye = getCalibrationTargetEye(step);
    const eyeLabel = getCalibrationStepLabel(step);
    const otherEyeLabel = getOppositeCalibrationStepLabel(step);
    const targetRatio = targetEye === "left" ? leftNormalized : rightNormalized;
    const otherOpenRatio = targetEye === "left" ? rightNormalized : leftNormalized;
    const targetStrength = targetEye === "left" ? leftStrength : rightStrength;
    const otherStrength = targetEye === "left" ? rightStrength : leftStrength;

    if (areBothEyesBlinking || areBothEyesAsymmetricallyBlinking || areBothEyesPartiallyClosed) {
      return {
        isValid: false,
        feedback: "양쪽 눈이 같이 감겼어요. 한쪽 눈만 편하게 다시 해볼게요.",
        targetRatio,
        targetStrength,
        otherOpenRatio,
      };
    }

    if (otherStrength > targetStrength && otherStrength > WINK_CALIBRATION_STRENGTH_MIN) {
      return {
        isValid: false,
        feedback: `${otherEyeLabel} 눈이 더 많이 움직였어요. ${eyeLabel} 눈만 감아 주세요.`,
        targetRatio,
        targetStrength,
        otherOpenRatio,
      };
    }

    if (targetRatio >= WINK_CALIBRATION_CLOSED_RATIO_MAX) {
      return {
        isValid: false,
        feedback: `${eyeLabel} 눈이 아직 충분히 감긴 상태로 보이지 않아요. 연주 중처럼 편하게 감아 주세요.`,
        targetRatio,
        targetStrength,
        otherOpenRatio,
      };
    }

    if (otherOpenRatio <= WINK_CALIBRATION_OTHER_EYE_OPEN_MIN) {
      return {
        isValid: false,
        feedback: `${otherEyeLabel} 눈도 같이 감겼어요. 반대쪽 눈은 편하게 떠 주세요.`,
        targetRatio,
        targetStrength,
        otherOpenRatio,
      };
    }

    if (
      targetStrength <= WINK_CALIBRATION_STRENGTH_MIN ||
      targetStrength <= otherStrength + WINK_CALIBRATION_STRENGTH_MIN
    ) {
      return {
        isValid: false,
        feedback: `${eyeLabel} 눈과 반대쪽 눈의 차이가 조금 더 필요해요.`,
        targetRatio,
        targetStrength,
        otherOpenRatio,
      };
    }

    return {
      isValid: true,
      feedback: `${eyeLabel} 눈 보정 중입니다. 그대로 잠깐 유지해 주세요.`,
      targetRatio,
      targetStrength,
      otherOpenRatio,
    };
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
    isWinkMirroredRef.current = isWinkMirrored;
  }, [isWinkMirrored]);

  useEffect(() => {
    let cancelled = false;

    const restoreReaderState = async () => {
      try {
        const [
          storedScores,
          sessionState,
          storedWinkMirror,
          storedMotionDetectionEnabled,
          storedCalibrationProfile,
        ] = await Promise.all([
          readReaderValue<SavedScore[]>(SAVED_SCORES_KEY),
          readReaderValue<SessionState>(SESSION_STATE_KEY),
          readReaderValue<boolean>(WINK_MIRROR_KEY),
          readReaderValue<boolean>(MOTION_DETECTION_ENABLED_KEY),
          readReaderValue<SavedWinkCalibrationProfile>(WINK_CALIBRATION_PROFILE_KEY),
        ]);

        if (cancelled) {
          return;
        }

        setSavedScores(storedScores ?? []);
        const nextWinkMirror = storedWinkMirror ?? true;
        isWinkMirroredRef.current = nextWinkMirror;
        setIsWinkMirrored(nextWinkMirror);
        if (storedCalibrationProfile) {
          applySavedCalibrationProfile(storedCalibrationProfile);
        }
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

    const stopCamera = async () => {
      resetWinkState();
      updateWinkDebug({
        status: "모션 감지 꺼짐",
        detail: "카메라가 멈춰 있어 윙크를 검사하지 않습니다.",
        leftRawRatio: null,
        rightRawRatio: null,
        leftNormalized: null,
        rightNormalized: null,
        leftBaseline: 0.24,
        rightBaseline: 0.24,
        leftNoise: 0,
        rightNoise: 0,
        leftStrength: 0,
        rightStrength: 0,
        signalStage: "none",
        forwardStableFrames: 0,
        trackingPhase: "idle",
        trackingEye: "none",
        stableFrames: 0,
        confirmedFrames: 0,
        openFrames: 0,
      });
      setIsCalibrating(false);
      resetCalibrationState();

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
      smoothedMouthRef.current = null;
      mouthStateRef.current = { phase: "idle", startedAt: null, closedAt: null, stableFrames: 0 };
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

        if (savedCalibrationProfileRef.current) {
          applySavedCalibrationProfile(savedCalibrationProfileRef.current);
          setIsCalibrating(false);
          updateWinkDebug({
            status: "저장된 보정값 사용 중",
            detail: "이전에 완료한 개인 윙크 보정값을 불러왔습니다.",
          });
          updateGestureText("저장된 윙크 보정값 사용 중");
        } else {
          startPersonalCalibration();
          updateWinkDebug({
            status: "보정 시작",
            detail: "정면, 왼쪽 윙크, 오른쪽 윙크를 순서대로 보정합니다.",
          });
        }

        faceMesh.onResults((results: Results) => {
          if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            updateWinkDebug({
              status: "실패: 얼굴 없음",
              detail: "얼굴 랜드마크를 찾지 못해서 윙크를 검사할 수 없습니다.",
              leftRawRatio: null,
              rightRawRatio: null,
              leftNormalized: null,
              rightNormalized: null,
              leftStrength: 0,
              rightStrength: 0,
              signalStage: "none",
              forwardStableFrames: 0,
              trackingPhase: "idle",
              trackingEye: "none",
              stableFrames: 0,
              confirmedFrames: 0,
              openFrames: 0,
            });
            updateGestureText("얼굴을 인식할 수 없음");
            if (calibrationActiveRef.current) {
              setCalibrationFeedback("얼굴을 화면 중앙에 맞춰 주세요.");
            }
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
          const moreClosedEyeNormalized = Math.min(leftNormalized, rightNormalized);
          const moreOpenEyeNormalized = Math.max(leftNormalized, rightNormalized);
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
          const calibratedOtherOpenBaseline = Math.min(
            leftWinkProfileRef.current?.otherEyeOpenRatio ?? 1,
            rightWinkProfileRef.current?.otherEyeOpenRatio ?? 1
          );
          const personalAsymOpenThreshold = Math.min(
            BOTH_EYES_ASYMMETRIC_BLINK_OPEN_RATIO_THRESHOLD,
            calibratedOtherOpenBaseline - 0.07
          );
          const areBothEyesAsymmetricallyBlinking =
            moreClosedEyeNormalized < BOTH_EYES_ASYMMETRIC_BLINK_CLOSED_RATIO_THRESHOLD &&
            moreOpenEyeNormalized < personalAsymOpenThreshold &&
            leftNormalized + rightNormalized < BOTH_EYES_ASYMMETRIC_BLINK_COMBINED_THRESHOLD;
          const blinkMoreOpenEyeNormalized = moreOpenEyeNormalized;
          const blinkCombinedNormalized = leftNormalized + rightNormalized;
          const winkState = winkStateRef.current;
          const winkSignal = computeWink(leftEyeRatio, rightEyeRatio, winkState.eye);
          const hasStrongWinkSignal = winkSignal.stage === "confirmed";
          const isTrackingWink = winkState.phase === "candidate" || winkState.phase === "tracking";

          updateWinkDebug({
            leftRawRatio: rawLeftEyeRatio,
            rightRawRatio: rawRightEyeRatio,
            leftNormalized,
            rightNormalized,
            leftBaseline: leftOpenBaseline,
            rightBaseline: rightOpenBaseline,
            leftNoise: leftEyeNoiseRef.current,
            rightNoise: rightEyeNoiseRef.current,
            leftStrength,
            rightStrength,
            signalStage: winkSignal.stage,
            forwardStableFrames: forwardStableFramesRef.current,
            trackingPhase: winkState.phase,
            trackingEye: winkState.eye,
            stableFrames: winkState.stableFrames,
            confirmedFrames: winkState.confirmedFrames,
            openFrames: winkState.openFrames,
          });

          if (calibrationActiveRef.current) {
            const activeCalibrationStep = calibrationStepRef.current;

            if (!isForwardEnoughForCalibration) {
              setCalibrationFeedback("얼굴이 조금 돌아갔어요. 정면으로 다시 봐 주세요.");
              updateWinkDebug({
                status: "보정 실패",
                detail: "보정 중에는 얼굴을 더 정면으로 맞춰야 합니다.",
              });
              updateGestureText("얼굴을 조금만 카메라 쪽으로");
              return;
            }

            if (activeCalibrationStep === "open") {
              calibrationFramesRef.current += 1;
              calibrationLeftSumRef.current += rawLeftEyeRatio;
              calibrationRightSumRef.current += rawRightEyeRatio;
              calibrationLeftSquareSumRef.current += rawLeftEyeRatio * rawLeftEyeRatio;
              calibrationRightSquareSumRef.current += rawRightEyeRatio * rawRightEyeRatio;
              updateCalibrationProgress("open", calibrationFramesRef.current);
              setCalibrationFeedback("정면을 보고 양쪽 눈을 편하게 떠 주세요.");
              updateWinkDebug({
                status: "정면 보정",
                detail: "양쪽 눈을 편하게 뜬 기준값이 안정적인지 확인하고 있습니다.",
              });
              updateGestureText("정면 보정 중");

              if (calibrationFramesRef.current >= OPEN_CALIBRATION_SAMPLE_TARGET) {
                const leftMean = calibrationLeftSumRef.current / calibrationFramesRef.current;
                const rightMean = calibrationRightSumRef.current / calibrationFramesRef.current;
                const leftVariance = Math.max(
                  calibrationLeftSquareSumRef.current / calibrationFramesRef.current - leftMean * leftMean,
                  0
                );
                const rightVariance = Math.max(
                  calibrationRightSquareSumRef.current / calibrationFramesRef.current - rightMean * rightMean,
                  0
                );
                const leftNoise = clamp(Math.sqrt(leftVariance) / Math.max(leftMean, 0.0001), 0, 0.16);
                const rightNoise = clamp(Math.sqrt(rightVariance) / Math.max(rightMean, 0.0001), 0, 0.16);

                if (leftNoise > OPEN_CALIBRATION_NOISE_MAX || rightNoise > OPEN_CALIBRATION_NOISE_MAX) {
                  resetOpenCalibrationSamples();
                  updateCalibrationProgress("open", 0);
                  setCalibrationFeedback("눈 깜빡임이나 흔들림이 섞였어요. 정면으로 편하게 다시 봐 주세요.");
                  updateWinkDebug({
                    status: "정면 보정 재시도",
                    detail: "눈 뜬 기준값의 흔들림이 커서 정면 보정 샘플을 다시 모읍니다.",
                  });
                  updateGestureText("정면 보정 다시 시도");
                  return;
                }

                leftEyeOpenRef.current = leftMean;
                rightEyeOpenRef.current = rightMean;
                leftEyeNoiseRef.current = clamp(leftNoise, 0, 0.08);
                rightEyeNoiseRef.current = clamp(rightNoise, 0, 0.08);
                resetOpenCalibrationSamples();
                moveToWinkCalibrationStep("left-wink");
              }

              return;
            }

            if (activeCalibrationStep === "left-wink" || activeCalibrationStep === "right-wink") {
              const winkSample = getCalibrationWinkSample({
                step: activeCalibrationStep,
                leftNormalized,
                rightNormalized,
                leftStrength,
                rightStrength,
                areBothEyesBlinking,
                areBothEyesAsymmetricallyBlinking,
                areBothEyesPartiallyClosed,
              });
              const eyeLabel = activeCalibrationStep === "left-wink" ? "왼쪽" : "오른쪽";

              if (!winkSample.isValid) {
                setCalibrationFeedback(winkSample.feedback);
                updateWinkDebug({
                  status: `${eyeLabel} 눈 보정 대기`,
                  detail: winkSample.feedback,
                });
                updateGestureText(`${eyeLabel} 눈 보정 중`);
                return;
              }

              calibrationFramesRef.current += 1;
              winkCalibrationTargetSumRef.current += winkSample.targetRatio;
              winkCalibrationTargetSquareSumRef.current += winkSample.targetRatio * winkSample.targetRatio;
              winkCalibrationStrengthSumRef.current += winkSample.targetStrength;
              winkCalibrationOtherOpenSumRef.current += winkSample.otherOpenRatio;
              updateCalibrationProgress(activeCalibrationStep, calibrationFramesRef.current);
              setCalibrationFeedback(winkSample.feedback);
              updateWinkDebug({
                status: `${eyeLabel} 눈 보정`,
                detail: `확실한 ${eyeLabel} 눈 감김 샘플을 모으는 중입니다. (${calibrationFramesRef.current}/${WINK_CALIBRATION_SAMPLE_TARGET})`,
              });
              updateGestureText(`${eyeLabel} 눈 보정 중`);

              if (calibrationFramesRef.current >= WINK_CALIBRATION_SAMPLE_TARGET) {
                const targetMean = winkCalibrationTargetSumRef.current / calibrationFramesRef.current;
                const targetVariance = Math.max(
                  winkCalibrationTargetSquareSumRef.current / calibrationFramesRef.current - targetMean * targetMean,
                  0
                );
                const nextProfile: WinkCalibrationProfile = {
                  closedRatio: targetMean,
                  strength: winkCalibrationStrengthSumRef.current / calibrationFramesRef.current,
                  otherEyeOpenRatio: winkCalibrationOtherOpenSumRef.current / calibrationFramesRef.current,
                  targetNoise: clamp(Math.sqrt(targetVariance), 0, 0.16),
                };

                const calibrationTargetEye = getCalibrationTargetEye(activeCalibrationStep);
                if (calibrationTargetEye === "left") {
                  leftWinkProfileRef.current = nextProfile;
                } else {
                  rightWinkProfileRef.current = nextProfile;
                }

                if (activeCalibrationStep === "left-wink") {
                  moveToWinkCalibrationStep("right-wink");
                } else {
                  resetWinkCalibrationSamples();
                  completePersonalCalibration();
                }
              }

              return;
            }

            return;
          }

          {
            const rawMouth = computeMouthOpenRatio(landmarks);
            const mouthRatio = getSmoothedEyeRatio(smoothedMouthRef, rawMouth);
            const mouthNow = performance.now();
            const ms = mouthStateRef.current;
            const idleState: MouthDetectionState = { phase: "idle", startedAt: null, closedAt: null, stableFrames: 0 };
            if (ms.phase === "waiting-close") {
              if (mouthRatio < MOUTH_CLOSE_RATIO_THRESHOLD) {
                mouthStateRef.current = idleState;
              }
            } else if (ms.phase === "first-closed") {
              if (mouthRatio > MOUTH_OPEN_RATIO_THRESHOLD) {
                const nextFrames = ms.stableFrames + 1;
                if (nextFrames >= MOUTH_STABLE_FRAME_TARGET) {
                  handleGesture("left");
                  mouthStateRef.current = { phase: "waiting-close", startedAt: null, closedAt: null, stableFrames: 0 };
                } else {
                  mouthStateRef.current = { ...ms, stableFrames: nextFrames };
                }
              } else if (ms.closedAt !== null && mouthNow - ms.closedAt > MOUTH_DOUBLE_TAP_WINDOW_MS) {
                mouthStateRef.current = idleState;
              } else {
                if (ms.stableFrames > 0) mouthStateRef.current = { ...ms, stableFrames: 0 };
              }
            } else if (ms.phase === "open") {
              if (mouthRatio < MOUTH_CLOSE_RATIO_THRESHOLD) {
                const openDuration = ms.startedAt !== null ? mouthNow - ms.startedAt : 0;
                if (openDuration >= MOUTH_MIN_OPEN_MS) {
                  mouthStateRef.current = { phase: "first-closed", startedAt: ms.startedAt, closedAt: mouthNow, stableFrames: 0 };
                } else {
                  mouthStateRef.current = idleState;
                }
              } else if (ms.startedAt !== null && mouthNow - ms.startedAt >= MOUTH_HOLD_DURATION_MS) {
                handleGesture("right");
                mouthStateRef.current = { phase: "waiting-close", startedAt: null, closedAt: null, stableFrames: 0 };
              }
            } else {
              if (mouthRatio > MOUTH_OPEN_RATIO_THRESHOLD) {
                const nextFrames = ms.stableFrames + 1;
                mouthStateRef.current = nextFrames >= MOUTH_STABLE_FRAME_TARGET
                  ? { phase: "open", startedAt: mouthNow, closedAt: null, stableFrames: nextFrames }
                  : { ...ms, stableFrames: nextFrames };
              } else {
                if (ms.stableFrames > 0) mouthStateRef.current = { ...ms, stableFrames: 0 };
              }
            }
          }

          if (isForwardEnoughForWink) {
            forwardStableFramesRef.current = Math.min(forwardStableFramesRef.current + 1, FORWARD_STABLE_FRAME_TARGET);
          } else {
            forwardStableFramesRef.current = 0;
          }

          if (
            areBothEyesBlinking ||
            (!hasStrongWinkSignal && areBothEyesAsymmetricallyBlinking) ||
            (!hasStrongWinkSignal && areBothEyesClosed(leftEyeRatio, rightEyeRatio, dominantWinkStrength, normalizedGap)) ||
            (!hasStrongWinkSignal && areBothEyesPartiallyClosed)
          ) {
            updateWinkDebug({
              status: "실패: 양쪽 눈 감김",
              detail: areBothEyesAsymmetricallyBlinking
                ? `양쪽 눈이 함께 내려간 깜빡임 패턴입니다. 더 열린 눈 비율(${blinkMoreOpenEyeNormalized.toFixed(2)}), 합산 비율(${blinkCombinedNormalized.toFixed(2)})이라 페이지 이동을 막았습니다.`
                : hasStrongWinkSignal
                  ? "두 눈이 함께 닫힌 상태로 보여 안전하게 양쪽 눈 감김으로 막았습니다."
                  : `두 눈이 모두 닫힘 범위에 들어왔고 비율 차이(${normalizedGap.toFixed(2)})와 한쪽 눈 우세 강도(${dominantWinkStrength.toFixed(2)})가 모두 양눈 깜빡임 쪽에 가깝습니다.`,
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: "waiting-open",
              trackingEye: "none",
              stableFrames: 0,
              confirmedFrames: 0,
              openFrames: winkState.phase === "waiting-open" ? winkState.openFrames : 0,
            });
            updateGestureText(winkState.phase === "waiting-open" ? "눈을 다시 떠 주세요" : "양쪽 눈 감김으로 인식 중");
            forwardStableFramesRef.current = 0;
            if (winkState.phase !== "waiting-open") {
              resetWinkState("waiting-open");
            }
            return;
          }

          if (!areBothEyesClearlyVisible(landmarks)) {
            if (isTrackingWink) {
              winkState.missingFrames += 1;
              if (winkState.missingFrames <= WINK_MISSING_FRAME_TOLERANCE) {
                updateWinkDebug({
                  status: "확인 중",
                  detail: "눈 위치가 잠깐 흔들렸지만 이미 잡힌 윙크 후보를 조금 더 확인합니다.",
                  forwardStableFrames: forwardStableFramesRef.current,
                  trackingPhase: winkState.phase,
                  trackingEye: winkState.eye,
                  stableFrames: winkState.stableFrames,
                  confirmedFrames: winkState.confirmedFrames,
                  openFrames: winkState.openFrames,
                });
                updateGestureText("윙크 확인 중");
                return;
              }
            }

            updateWinkDebug({
              status: "실패: 눈 가시성 부족",
              detail: "양쪽 눈 크기나 위치 균형이 많이 달라 보여 정면 윙크로 확신하지 못했습니다.",
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: winkState.phase,
              trackingEye: winkState.eye,
              stableFrames: winkState.stableFrames,
              confirmedFrames: winkState.confirmedFrames,
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
            if (isTrackingWink) {
              winkState.missingFrames += 1;
              if (winkState.missingFrames <= WINK_MISSING_FRAME_TOLERANCE) {
                updateWinkDebug({
                  status: "확인 중",
                  detail: "정면 조건이 잠깐 흔들렸지만 이미 잡힌 윙크 후보를 조금 더 확인합니다.",
                  forwardStableFrames: forwardStableFramesRef.current,
                  trackingPhase: winkState.phase,
                  trackingEye: winkState.eye,
                  stableFrames: winkState.stableFrames,
                  confirmedFrames: winkState.confirmedFrames,
                  openFrames: winkState.openFrames,
                });
                updateGestureText("윙크 확인 중");
                return;
              }
            }

            updateWinkDebug({
              status: "실패: 정면 아님",
              detail: "고개 회전이나 좌우 비대칭이 커서, 정면 윙크로 보기엔 너무 불안정합니다.",
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: winkState.phase,
              trackingEye: winkState.eye,
              stableFrames: winkState.stableFrames,
              confirmedFrames: winkState.confirmedFrames,
              openFrames: winkState.openFrames,
            });
            updateGestureText("고개를 정면으로 맞춰 주세요");
            if (winkStateRef.current.phase !== "waiting-open") {
              resetWinkState();
            }
            return;
          }

          if (forwardStableFramesRef.current < FORWARD_STABLE_FRAME_TARGET) {
            if (isTrackingWink) {
              updateWinkDebug({
                status: "확인 중",
                detail: "정면 상태가 다시 안정되는지 확인하면서 윙크 후보를 유지합니다.",
                forwardStableFrames: forwardStableFramesRef.current,
                trackingPhase: winkState.phase,
                trackingEye: winkState.eye,
                stableFrames: winkState.stableFrames,
                confirmedFrames: winkState.confirmedFrames,
                openFrames: winkState.openFrames,
              });
              updateGestureText("윙크 확인 중");
              return;
            }

            updateWinkDebug({
              status: "실패: 정면 유지 부족",
              detail: "정면 상태가 아직 충분한 프레임 수만큼 이어지지 않았습니다.",
              forwardStableFrames: forwardStableFramesRef.current,
              trackingPhase: winkState.phase,
              trackingEye: winkState.eye,
              stableFrames: winkState.stableFrames,
              confirmedFrames: winkState.confirmedFrames,
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
                confirmedFrames: 0,
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
                confirmedFrames: winkState.confirmedFrames,
                openFrames: winkState.openFrames,
              });
              updateGestureText("눈을 다시 떠 주세요");
            }
            return;
          }

          const now = performance.now();

          if (winkSignal.eye === "none") {
            if (isTrackingWink) {
              winkState.missingFrames += 1;
              if (winkState.missingFrames > WINK_MISSING_FRAME_TOLERANCE) {
                resetWinkState();
                updateWinkDebug({
                  status: "실패: 유지 시간 부족",
                  detail: "한쪽 눈 감기가 잠깐 보였지만 충분히 오래 유지되지 않았습니다.",
                  trackingPhase: "idle",
                  trackingEye: "none",
                  stableFrames: 0,
                  confirmedFrames: 0,
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
                  confirmedFrames: winkState.confirmedFrames,
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
                confirmedFrames: 0,
                openFrames: 0,
              });
              updateGestureText("짧게 한쪽 윙크");
            }
            return;
          }

          if (!isTrackingWink || winkState.eye !== winkSignal.eye) {
            const { isSuspicious, eyeWidthBalance } = isWinkEyeSuspicious(winkSignal.eye as "left" | "right", landmarks);
            if (isSuspicious) {
              resetWinkState();
              updateWinkDebug({
                status: "실패: 고개 회전 의심",
                detail: `${winkSignal.eye === "left" ? "왼쪽" : "오른쪽"} 눈이 더 좁은 눈(균형 ${eyeWidthBalance.toFixed(2)})과 일치해 고개 회전으로 인한 오인식으로 판단했습니다.`,
                trackingPhase: "idle",
                trackingEye: "none",
                stableFrames: 0,
                confirmedFrames: 0,
                openFrames: 0,
              });
              updateGestureText("고개를 정면으로 맞춰 주세요");
              return;
            }
            const nextPhase: WinkPhase = winkSignal.stage === "confirmed" ? "tracking" : "candidate";
            winkStateRef.current = {
              phase: nextPhase,
              eye: winkSignal.eye,
              direction: winkSignal.direction,
              startedAt: now,
              peakStrength: winkSignal.strength,
              stableFrames: 1,
              confirmedFrames: winkSignal.stage === "confirmed" ? 1 : 0,
              missingFrames: 0,
              openFrames: 0,
            };
            updateWinkDebug({
              status: winkSignal.stage === "confirmed" ? "강한 윙크 추적" : "후보 포착",
              detail:
                winkSignal.stage === "confirmed"
                  ? `${winkSignal.eye === "left" ? "왼쪽" : "오른쪽"} 눈 감기가 강하게 잡혔고, 확정 프레임을 더 확인합니다.`
                  : `${winkSignal.eye === "left" ? "왼쪽" : "오른쪽"} 눈 감기 후보를 잡았습니다. 아직 페이지를 넘기지는 않습니다.`,
              trackingPhase: nextPhase,
              trackingEye: winkSignal.eye,
              stableFrames: 1,
              confirmedFrames: winkSignal.stage === "confirmed" ? 1 : 0,
              openFrames: 0,
            });
            updateGestureText("윙크 확인 중");
            return;
          }

          winkState.missingFrames = 0;
          winkState.stableFrames = Math.min(winkState.stableFrames + 1, WINK_STABLE_FRAME_TARGET);
          winkState.peakStrength = Math.max(winkState.peakStrength, winkSignal.strength);
          winkState.direction = winkSignal.direction;
          if (winkSignal.stage === "confirmed") {
            winkState.phase = "tracking";
            winkState.confirmedFrames = Math.min(winkState.confirmedFrames + 1, WINK_CONFIRM_FRAME_TARGET);
          } else {
            winkState.phase = "candidate";
          }

          const holdDuration = winkState.startedAt ? now - winkState.startedAt : 0;
          const isConfirmedEnough = winkState.confirmedFrames >= WINK_CONFIRM_FRAME_TARGET;
          updateWinkDebug({
            status: isConfirmedEnough ? "성공 직전" : "후보 확인 중",
            detail: isConfirmedEnough
              ? "강한 한쪽 눈 감기가 유지되고 있어 페이지 넘김 조건을 채우는 중입니다."
              : `후보는 유지 중이지만 강한 윙크 확정 프레임이 더 필요합니다. (${winkState.confirmedFrames}/${WINK_CONFIRM_FRAME_TARGET})`,
            trackingPhase: winkState.phase,
            trackingEye: winkState.eye,
            stableFrames: winkState.stableFrames,
            confirmedFrames: winkState.confirmedFrames,
            openFrames: winkState.openFrames,
          });
          if (winkState.direction === "none") {
            updateGestureText("윙크 확인 중");
          } else if (!isConfirmedEnough) {
            updateGestureText(`윙크 확정 중 ${winkState.confirmedFrames}/${WINK_CONFIRM_FRAME_TARGET}`);
          } else {
            updateGestureText(getPendingGestureText(winkState.direction, holdDuration));
          }

          if (
            winkState.stableFrames >= WINK_STABLE_FRAME_TARGET &&
            winkState.confirmedFrames >= WINK_CONFIRM_FRAME_TARGET &&
            holdDuration >= WINK_HOLD_DURATION_MS &&
            winkState.peakStrength >= WINK_MIN_PEAK_STRENGTH &&
            winkState.phase === "tracking" &&
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
                confirmedFrames: 0,
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
                confirmedFrames: 0,
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
    startPersonalCalibration();
    updateWinkDebug({
      status: "보정 시작",
      detail: "정면, 왼쪽 윙크, 오른쪽 윙크를 순서대로 보정합니다.",
    });
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

  const computeMouthOpenRatio = (landmarks: NormalizedLandmark[]) => {
    const leftCorner = landmarks[61];
    const rightCorner = landmarks[291];
    const width = Math.hypot(rightCorner.x - leftCorner.x, rightCorner.y - leftCorner.y);
    if (width === 0) return 0;
    const upper = landmarks[13];
    const lower = landmarks[14];
    return Math.hypot(upper.x - lower.x, upper.y - lower.y) / width;
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
      leftEyeWidth,
      rightEyeWidth,
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

    return (
      eyeLineTilt < ACTIVE_FORWARD_TILT_MAX &&
      noseOffsetX < ACTIVE_FORWARD_NOSE_X_MAX &&
      noseOffsetY < ACTIVE_FORWARD_NOSE_Y_MAX &&
      eyeDistanceX > ACTIVE_FORWARD_EYE_DISTANCE_MIN
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

  const getAdaptiveWinkThresholds = (targetEye: "left" | "right") => {
    const targetNoise = targetEye === "left" ? leftEyeNoiseRef.current : rightEyeNoiseRef.current;
    const otherNoise = targetEye === "left" ? rightEyeNoiseRef.current : leftEyeNoiseRef.current;
    const combinedNoise = Math.max(targetNoise, otherNoise);
    const profile = targetEye === "left" ? leftWinkProfileRef.current : rightWinkProfileRef.current;

    if (profile) {
      const profileNoise = Math.max(profile.targetNoise, targetNoise);

      return {
        candidateClosed: clamp(profile.closedRatio + Math.max(0.12, profileNoise * 2), 0.72, 0.94),
        candidateOtherOpen: clamp(profile.otherEyeOpenRatio - 0.1 - otherNoise * 0.5, 0.74, 0.9),
        candidateGap: clamp(profile.strength * 0.35, 0.05, 0.18),
        confirmClosed: clamp(profile.closedRatio + Math.max(0.06, profileNoise * 1.4), 0.66, 0.9),
        confirmOtherOpen: clamp(profile.otherEyeOpenRatio - 0.12 - otherNoise * 0.4, 0.72, 0.90),
        confirmGap: clamp(profile.strength * 0.55, 0.07, 0.24),
        continueClosed: clamp(profile.closedRatio + Math.max(0.18, profileNoise * 2.0), 0.7, 0.93),
        continueOtherOpen: clamp(profile.otherEyeOpenRatio - 0.20 - otherNoise * 0.5, 0.62, 0.9),
        continueGap: clamp(profile.strength * 0.20, 0.03, 0.14),
      };
    }

    return {
      candidateClosed: clamp(WINK_CANDIDATE_CLOSED_RATIO_THRESHOLD - targetNoise * 0.9, 0.8, 0.9),
      candidateOtherOpen: clamp(WINK_CANDIDATE_OTHER_EYE_OPEN_RATIO_THRESHOLD - otherNoise * 0.8, 0.74, 0.8),
      candidateGap: clamp(WINK_CANDIDATE_CLOSURE_GAP_THRESHOLD + combinedNoise * 1.1, 0.08, 0.17),
      confirmClosed: clamp(WINK_CONFIRM_CLOSED_RATIO_THRESHOLD - targetNoise * 0.4, 0.74, 0.84),
      confirmOtherOpen: clamp(WINK_CONFIRM_OTHER_EYE_OPEN_RATIO_THRESHOLD - otherNoise * 0.6, 0.84, 0.88),
      confirmGap: clamp(WINK_CONFIRM_CLOSURE_GAP_THRESHOLD + combinedNoise * 0.8, 0.18, 0.28),
      continueClosed: clamp(WINK_CONTINUE_CLOSED_RATIO_THRESHOLD - targetNoise * 0.7, 0.76, 0.88),
      continueOtherOpen: clamp(WINK_CONTINUE_OTHER_EYE_OPEN_RATIO_THRESHOLD - otherNoise * 0.8, 0.64, 0.8),
      continueGap: clamp(WINK_CONTINUE_CLOSURE_GAP_THRESHOLD + combinedNoise * 0.8, 0.06, 0.17),
    };
  };

  const getWinkDirection = (eye: "left" | "right"): WinkDirection => {
    if (eye === "left") {
      return isWinkMirroredRef.current ? "right" : "left";
    }

    return isWinkMirroredRef.current ? "left" : "right";
  };

  const computeWink = (leftEyeRatio: number, rightEyeRatio: number, activeCandidate: WinkEye): WinkSignal => {
    const leftOpenBaseline = Math.max(leftEyeOpenRef.current, 0.0001);
    const rightOpenBaseline = Math.max(rightEyeOpenRef.current, 0.0001);
    const leftNormalized = leftEyeRatio / leftOpenBaseline;
    const rightNormalized = rightEyeRatio / rightOpenBaseline;
    const leftClosure = 1 - leftNormalized;
    const rightClosure = 1 - rightNormalized;
    const leftStrength = leftClosure - rightClosure;
    const rightStrength = rightClosure - leftClosure;
    const leftThresholds = getAdaptiveWinkThresholds("left");
    const rightThresholds = getAdaptiveWinkThresholds("right");

    const isLeftCandidate =
      leftNormalized < leftThresholds.candidateClosed &&
      rightNormalized > leftThresholds.candidateOtherOpen &&
      leftStrength > leftThresholds.candidateGap;
    const isRightCandidate =
      rightNormalized < rightThresholds.candidateClosed &&
      leftNormalized > rightThresholds.candidateOtherOpen &&
      rightStrength > rightThresholds.candidateGap;

    const isLeftConfirmed =
      leftNormalized < leftThresholds.confirmClosed &&
      rightNormalized > leftThresholds.confirmOtherOpen &&
      leftStrength > leftThresholds.confirmGap;
    const isRightConfirmed =
      rightNormalized < rightThresholds.confirmClosed &&
      leftNormalized > rightThresholds.confirmOtherOpen &&
      rightStrength > rightThresholds.confirmGap;

    const isLeftContinue =
      leftNormalized < leftThresholds.continueClosed &&
      rightNormalized > leftThresholds.continueOtherOpen &&
      leftStrength > leftThresholds.continueGap;
    const isRightContinue =
      rightNormalized < rightThresholds.continueClosed &&
      leftNormalized > rightThresholds.continueOtherOpen &&
      rightStrength > rightThresholds.continueGap;

    const leftSignal = (stage: WinkSignalStage): WinkSignal => ({
      eye: "left",
      direction: getWinkDirection("left"),
      strength: leftStrength,
      stage,
    });

    const rightSignal = (stage: WinkSignalStage): WinkSignal => ({
      eye: "right",
      direction: getWinkDirection("right"),
      strength: rightStrength,
      stage,
    });

    if (activeCandidate === "left" && isLeftContinue) {
      return leftSignal(isLeftConfirmed ? "confirmed" : "candidate");
    }

    if (activeCandidate === "right" && isRightContinue) {
      return rightSignal(isRightConfirmed ? "confirmed" : "candidate");
    }

    if (isRightConfirmed && rightStrength >= leftStrength) {
      return rightSignal("confirmed");
    }

    if (isLeftConfirmed) {
      return leftSignal("confirmed");
    }

    if (isRightCandidate && rightStrength >= leftStrength) {
      return rightSignal("candidate");
    }

    if (isLeftCandidate) {
      return leftSignal("candidate");
    }

    return { eye: "none", direction: "none", strength: 0, stage: "none" };
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

  // 고개 회전으로 인한 오인식 방지: 감지된 눈이 더 좁은 눈이고 균형 비율이 낮으면 오인식으로 판단
  const isWinkEyeSuspicious = (winkEye: "left" | "right", landmarks: NormalizedLandmark[]) => {
    const { leftEyeWidth, rightEyeWidth, eyeWidthBalance } = getEyeVisibilityMetrics(landmarks);
    if (eyeWidthBalance >= HEAD_TURN_EYE_BALANCE_THRESHOLD) {
      return { isSuspicious: false, eyeWidthBalance };
    }
    const narrowerEye: "left" | "right" = leftEyeWidth < rightEyeWidth ? "left" : "right";
    return { isSuspicious: winkEye === narrowerEye, eyeWidthBalance };
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
  const formatDebugSmallNumber = (value: number | null) => (value === null ? "-" : value.toFixed(3));
  const calibrationStepInfo = (() => {
    if (calibrationStep === "left-wink") {
      return {
        step: "2/3",
        title: "왼쪽 눈 보정",
        instruction: "왼쪽 눈만 연주 중처럼 편하게 감아 주세요.",
      };
    }

    if (calibrationStep === "right-wink") {
      return {
        step: "3/3",
        title: "오른쪽 눈 보정",
        instruction: "오른쪽 눈만 연주 중처럼 편하게 감아 주세요.",
      };
    }

    return {
      step: "1/3",
      title: "정면 보정",
      instruction: "정면을 보고 양쪽 눈을 편하게 떠 주세요.",
    };
  })();
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
                <strong>정면, 왼쪽, 오른쪽 순서로 보정</strong>
                <span>큰 안내 문구에 맞춰 확실히 잡히면 자동으로 다음 단계로 넘어갑니다.</span>
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

      {isCalibrating ? (
        <section className="calibration-overlay" aria-live="polite" aria-label="윙크 보정 안내">
          <div className="calibration-card">
            <div className="calibration-step-pill">보정 {calibrationStepInfo.step}</div>
            <h2>{calibrationStepInfo.title}</h2>
            <p className="calibration-instruction">{calibrationStepInfo.instruction}</p>
            <p className="calibration-feedback">{calibrationFeedback}</p>
            <div className="calibration-progress-track" aria-hidden="true">
              <div className="calibration-progress-bar" style={{ width: `${calibrationProgress}%` }} />
            </div>
            <span className="calibration-progress-label">{calibrationProgress}%</span>
          </div>
        </section>
      ) : null}

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
                  isWinkMirroredRef.current = next;
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
                <dt>왼눈 원본</dt>
                <dd>{formatDebugSmallNumber(winkDebug.leftRawRatio)}</dd>
              </div>
              <div>
                <dt>오른눈 원본</dt>
                <dd>{formatDebugSmallNumber(winkDebug.rightRawRatio)}</dd>
              </div>
              <div>
                <dt>왼눈 비율</dt>
                <dd>{formatDebugNumber(winkDebug.leftNormalized)}</dd>
              </div>
              <div>
                <dt>오른눈 비율</dt>
                <dd>{formatDebugNumber(winkDebug.rightNormalized)}</dd>
              </div>
              <div>
                <dt>왼눈 기준</dt>
                <dd>{formatDebugSmallNumber(winkDebug.leftBaseline)}</dd>
              </div>
              <div>
                <dt>오른눈 기준</dt>
                <dd>{formatDebugSmallNumber(winkDebug.rightBaseline)}</dd>
              </div>
              <div>
                <dt>왼눈 노이즈</dt>
                <dd>{winkDebug.leftNoise.toFixed(3)}</dd>
              </div>
              <div>
                <dt>오른눈 노이즈</dt>
                <dd>{winkDebug.rightNoise.toFixed(3)}</dd>
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
                <dt>신호 단계</dt>
                <dd>{winkDebug.signalStage}</dd>
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
                <dt>확정 프레임</dt>
                <dd>{winkDebug.confirmedFrames}</dd>
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
