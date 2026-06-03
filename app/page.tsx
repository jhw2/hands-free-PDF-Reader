"use client";

import { useEffect, useRef, useState } from "react";
import type { Camera as MediaPipeCamera } from "@mediapipe/camera_utils";
import type { FaceMesh as MediaPipeFaceMesh, NormalizedLandmark, Results } from "@mediapipe/face_mesh";

const READER_DB_NAME = "motion-pdf-reader";
const READER_STORE_NAME = "reader-state";
const SESSION_STATE_KEY = "session-state";
const SAVED_SCORES_KEY = "saved-scores";
const WINK_MIRROR_KEY = "wink-mirror";
const CALIBRATION_SAMPLE_TARGET = 18;

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
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [isWinkMirrored, setIsWinkMirrored] = useState(true);
  const [cameraPermission, setCameraPermission] = useState<"unknown" | "prompt" | "granted" | "denied">(
    "unknown"
  );
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [hasCalibration, setHasCalibration] = useState(false);

  const lastSwitchTimeRef = useRef<number>(0);
  const isRenderingRef = useRef(false);
  const pendingPageRef = useRef<number | null>(null);
  const cameraInstanceRef = useRef<MediaPipeCamera | null>(null);
  const faceMeshRef = useRef<MediaPipeFaceMesh | null>(null);
  const pdfDocRef = useRef<any>(null);
  const pageCountRef = useRef(0);
  const currentPageRef = useRef(1);
  const currentPdfBlobRef = useRef<Blob | null>(null);
  const currentSavedScoreIdRef = useRef<string | null>(null);
  const winkCandidateRef = useRef<"left" | "right" | "none">("none");
  const winkFrameCountRef = useRef(0);
  const gestureArmedRef = useRef(true);
  const calibrationActiveRef = useRef(false);
  const calibrationFramesRef = useRef(0);
  const calibrationLeftSumRef = useRef(0);
  const calibrationRightSumRef = useRef(0);
  const leftEyeOpenRef = useRef(0.24);
  const rightEyeOpenRef = useRef(0.24);

  const loadPdfJs = async () => {
    if (window.pdfjsLib) {
      return window.pdfjsLib;
    }

    const pdfjsLib = await import("pdfjs-dist/build/pdf.mjs");
    window.pdfjsLib = pdfjsLib;
    pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
    return pdfjsLib;
  };

  const saveSessionState = async (overrides?: Partial<SessionState>) => {
    const sessionState: SessionState = {
      pdfBlob: currentPdfBlobRef.current,
      fileName: currentFileName,
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
        const [storedScores, sessionState, storedWinkMirror] = await Promise.all([
          readReaderValue<SavedScore[]>(SAVED_SCORES_KEY),
          readReaderValue<SessionState>(SESSION_STATE_KEY),
          readReaderValue<boolean>(WINK_MIRROR_KEY),
        ]);

        if (cancelled) {
          return;
        }

        setSavedScores(storedScores ?? []);
        setIsWinkMirrored(storedWinkMirror ?? true);

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
    let cancelled = false;

    const resetCalibration = () => {
      calibrationActiveRef.current = false;
      calibrationFramesRef.current = 0;
      calibrationLeftSumRef.current = 0;
      calibrationRightSumRef.current = 0;
      leftEyeOpenRef.current = 0.24;
      rightEyeOpenRef.current = 0.24;
      setCalibrationProgress(0);
      setHasCalibration(false);
    };

    const stopCamera = async () => {
      winkCandidateRef.current = "none";
      winkFrameCountRef.current = 0;
      gestureArmedRef.current = true;
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
          setGestureText("카메라 대기 중");
        } else {
          setGestureText("카메라 꺼짐");
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
        setGestureText("정면을 1초만 봐 주세요");

        faceMesh.onResults((results: Results) => {
          if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            setGestureText("얼굴을 인식할 수 없음");
            winkCandidateRef.current = "none";
            winkFrameCountRef.current = 0;
            gestureArmedRef.current = true;
            return;
          }

          const landmarks = results.multiFaceLandmarks[0];
          const leftEyeRatio = computeEyeRatio(landmarks, 33, 133, 159, 145);
          const rightEyeRatio = computeEyeRatio(landmarks, 362, 263, 386, 374);
          const isFacingForward = isFaceFacingForward(landmarks);

          if (calibrationActiveRef.current) {
            if (!isFacingForward) {
              setGestureText("얼굴을 정면으로 맞춰 주세요");
              return;
            }

            calibrationFramesRef.current += 1;
            calibrationLeftSumRef.current += leftEyeRatio;
            calibrationRightSumRef.current += rightEyeRatio;

            const nextProgress = Math.min(
              100,
              Math.round((calibrationFramesRef.current / CALIBRATION_SAMPLE_TARGET) * 100)
            );
            setCalibrationProgress(nextProgress);
            setGestureText(`보정 중 ${nextProgress}%`);

            if (calibrationFramesRef.current >= CALIBRATION_SAMPLE_TARGET) {
              leftEyeOpenRef.current = calibrationLeftSumRef.current / calibrationFramesRef.current;
              rightEyeOpenRef.current = calibrationRightSumRef.current / calibrationFramesRef.current;
              calibrationActiveRef.current = false;
              setIsCalibrating(false);
              setHasCalibration(true);
              setGestureText("보정 완료");
            }
            return;
          }

          const eyesRecovered = areEyesRecovered(leftEyeRatio, rightEyeRatio);
          if (eyesRecovered) {
            gestureArmedRef.current = true;
          }

          const wink = computeWink(leftEyeRatio, rightEyeRatio);
          const quickWink = isQuickWink(leftEyeRatio, rightEyeRatio, wink);

          if (wink === "none") {
            setGestureText(isFacingForward ? "윙크 대기 중" : "정면이면 더 잘 인식돼요");
            winkCandidateRef.current = "none";
            winkFrameCountRef.current = 0;
            return;
          }

          if (!gestureArmedRef.current) {
            return;
          }

          if (winkCandidateRef.current === wink) {
            winkFrameCountRef.current += 1;
          } else {
            winkCandidateRef.current = wink;
            winkFrameCountRef.current = 1;
          }

          const requiredFrames = quickWink ? 1 : 2;

          if (winkFrameCountRef.current >= requiredFrames) {
            handleGesture(wink);
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
          setGestureText("정면을 1초만 봐 주세요");
        }
      } catch (error) {
        console.error("mediapipe init error", error);
        await stopCamera();
        if (!cancelled) {
          if (error instanceof DOMException && error.name === "NotAllowedError") {
            setCameraPermission("denied");
          }
          setCameraEnabled(false);
          setGestureText("카메라 권한 허용이 필요합니다.");
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
      setGestureText("PDF를 불러왔습니다. 저장 버튼을 누르면 목록에 추가됩니다.");
    } catch (error) {
      setGestureText("PDF 라이브러리 로드에 실패했습니다.");
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

  const saveCurrentPdf = async () => {
    if (!currentPdfBlobRef.current) {
      setGestureText("먼저 PDF를 불러와 주세요.");
      return;
    }

    const nextScore: SavedScore = {
      id: currentSavedScoreIdRef.current ?? crypto.randomUUID(),
      name: currentFileName ?? `악보 ${savedScores.length + 1}`,
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
    setGestureText("현재 PDF를 목록에 저장했습니다.");
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
      setGestureText("저장된 악보를 불러왔습니다.");
      setIsLibraryOpen(false);
    } catch (error) {
      console.error("saved score open error", error);
      setGestureText("저장된 악보를 불러오지 못했습니다.");
    }
  };

  const deleteSavedScore = async (scoreId: string) => {
    const nextScores = savedScores.filter((score) => score.id !== scoreId);
    await saveScoresToDb(nextScores);

    if (currentSavedScoreIdRef.current === scoreId) {
      currentSavedScoreIdRef.current = null;
      await saveSessionState({ savedScoreId: null });
    }

    setGestureText("저장된 악보를 목록에서 삭제했습니다.");
  };

  const handleCameraAction = () => {
    if (cameraEnabled) {
      setCameraEnabled(false);
      return;
    }

    setCameraEnabled(true);
    setGestureText("카메라 권한 창이 뜨면 허용해 주세요.");
  };

  const handleCalibrationReset = () => {
    calibrationActiveRef.current = true;
    calibrationFramesRef.current = 0;
    calibrationLeftSumRef.current = 0;
    calibrationRightSumRef.current = 0;
    setCalibrationProgress(0);
    setHasCalibration(false);
    setIsCalibrating(true);
    setGestureText("정면을 1초만 봐 주세요");
  };

  const handleGesture = (direction: "left" | "right") => {
    const now = performance.now();
    if (now - lastSwitchTimeRef.current < 1500) return;

    if (direction === "right") {
      queuePage(currentPageRef.current + 1);
      setGestureText("다음 페이지");
    } else {
      queuePage(currentPageRef.current - 1);
      setGestureText("이전 페이지");
    }

    lastSwitchTimeRef.current = now;
    gestureArmedRef.current = false;
    winkCandidateRef.current = "none";
    winkFrameCountRef.current = 0;
  };

  const computeEyeRatio = (
    landmarks: NormalizedLandmark[],
    outerIndex: number,
    innerIndex: number,
    upperIndex: number,
    lowerIndex: number
  ) => {
    const outer = landmarks[outerIndex];
    const inner = landmarks[innerIndex];
    const upper = landmarks[upperIndex];
    const lower = landmarks[lowerIndex];

    const horizontal = Math.hypot(outer.x - inner.x, outer.y - inner.y);
    const vertical = Math.hypot(upper.x - lower.x, upper.y - lower.y);

    if (horizontal === 0) {
      return 0;
    }

    return vertical / horizontal;
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

    return eyeLineTilt < 0.035 && noseOffsetX < 0.045 && noseOffsetY < 0.18;
  };

  const computeWink = (leftEyeRatio: number, rightEyeRatio: number) => {
    const leftClosedThreshold = leftEyeOpenRef.current * 0.82;
    const rightClosedThreshold = rightEyeOpenRef.current * 0.82;
    const leftOpenThreshold = leftEyeOpenRef.current * 0.72;
    const rightOpenThreshold = rightEyeOpenRef.current * 0.72;
    const leftMuchSmallerThanRight = leftEyeRatio < rightEyeRatio * 0.82;
    const rightMuchSmallerThanLeft = rightEyeRatio < leftEyeRatio * 0.82;

    const leftClosed = leftEyeRatio < leftClosedThreshold && leftMuchSmallerThanRight;
    const rightClosed = rightEyeRatio < rightClosedThreshold && rightMuchSmallerThanLeft;
    const leftOpen = leftEyeRatio > leftOpenThreshold;
    const rightOpen = rightEyeRatio > rightOpenThreshold;

    if (rightClosed && leftOpen) return isWinkMirrored ? "left" : "right";
    if (leftClosed && rightOpen) return isWinkMirrored ? "right" : "left";
    return "none";
  };

  const isQuickWink = (leftEyeRatio: number, rightEyeRatio: number, wink: "left" | "right" | "none") => {
    if (wink === "none") {
      return false;
    }

    const strongerClosedRatio = 0.68;
    const strongerOpenRatio = 0.72;
    const targetLeftEye = (wink === "right" && isWinkMirrored) || (wink === "left" && !isWinkMirrored);
    const activeEyeRatio = targetLeftEye ? leftEyeRatio : rightEyeRatio;
    const supportEyeRatio = targetLeftEye ? rightEyeRatio : leftEyeRatio;
    const activeEyeOpen = targetLeftEye ? leftEyeOpenRef.current : rightEyeOpenRef.current;
    const supportEyeOpen = targetLeftEye ? rightEyeOpenRef.current : leftEyeOpenRef.current;

    return (
      activeEyeRatio < activeEyeOpen * strongerClosedRatio &&
      supportEyeRatio > supportEyeOpen * strongerOpenRatio &&
      activeEyeRatio < supportEyeRatio * 0.7
    );
  };

  const areEyesRecovered = (leftEyeRatio: number, rightEyeRatio: number) => {
    const leftRecoverThreshold = leftEyeOpenRef.current * 0.72;
    const rightRecoverThreshold = rightEyeOpenRef.current * 0.72;
    return leftEyeRatio >= leftRecoverThreshold && rightEyeRatio >= rightRecoverThreshold;
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
    ? "카메라 중지"
    : cameraPermission === "denied"
      ? "권한 다시 확인하기"
      : "카메라 시작하기";

  return (
    <main className={`app-shell ${isFocusMode ? "is-focus-mode" : ""}`}>
      <header className="topbar">
        <div className="brand-block">
          <h1>Hands Free PDF Reader</h1>
        </div>

        <div className="topbar-actions">
          <label className="file-label compact-action">
            PDF 불러오기
            <input type="file" accept="application/pdf" onChange={onFileChange} />
          </label>

          <button
            type="button"
            className={`camera-toggle compact-action ${isCameraReady ? "is-on" : "is-off"}`}
            onClick={handleCameraAction}
            aria-pressed={cameraEnabled}
          >
            {cameraButtonText}
            <span>{cameraStatusText}</span>
          </button>
          <button
            type="button"
            className={`secondary-button compact-action ${isFocusMode ? "is-active" : ""}`}
            onClick={() => setIsFocusMode((prev) => !prev)}
          >
            {isFocusMode ? "전체보기 종료" : "악보 전체보기"}
          </button>
        </div>
      </header>

      <section className="status-strip" aria-label="현재 상태">
        <div className="status-chip">
          <span>악보</span>
          <strong>{currentFileName ?? "없음"}</strong>
        </div>
        <div className="status-chip">
          <span>페이지</span>
          <strong>{pageText}</strong>
        </div>
        <div className="status-chip">
          <span>안내</span>
          <strong>{gestureText}</strong>
        </div>
      </section>

      {cameraPermission === "denied" ? <p className="permission-help">카메라 권한이 차단됨. 주소창에서 허용해 주세요.</p> : null}

      <section className="workspace-grid">
        <div className="viewer-card">
          <div className="viewer-toolbar">
            <h2>악보</h2>
            <div className="page-buttons" aria-label="페이지 이동 버튼">
              <button type="button" onClick={() => queuePage(currentPage - 1)} disabled={!canGoPrevious}>
                이전 페이지
              </button>
              <button type="button" onClick={() => queuePage(currentPage + 1)} disabled={!canGoNext}>
                다음 페이지
              </button>
              <button type="button" onClick={() => setIsFocusMode((prev) => !prev)}>
                {isFocusMode ? "전체보기 종료" : "전체보기"}
              </button>
            </div>
          </div>

          <div className="pdf-viewer">
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
          <button type="button" className="secondary-button" onClick={saveCurrentPdf} disabled={!hasLoadedPdf}>
            현재 PDF 저장
          </button>
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
            className="secondary-button"
                onClick={() => setIsLibraryOpen((prev) => !prev)}
                aria-expanded={isLibraryOpen}
              >
                저장된 악보 {isLibraryOpen ? "숨기기" : "보기"}
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
                인식 방향 바꾸기 {isWinkMirrored ? "ON" : "OFF"}
              </button>
            </div>
          </div>

          <div className="camera-preview compact-panel">
            <div className="camera-preview-header">
              <strong>카메라</strong>
              <span>{isCameraReady ? "인식 중" : "대기 중"}</span>
            </div>
            <div className="camera-preview-frame">
              <video ref={videoRef} autoPlay muted playsInline className="camera-feed" />
              {!isCameraReady ? (
                <div className="camera-placeholder">
                  <strong>대기 중</strong>
                </div>
              ) : null}
            </div>
          </div>

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
