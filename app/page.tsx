"use client";

import { useEffect, useRef, useState } from "react";
import type { Camera as MediaPipeCamera } from "@mediapipe/camera_utils";
import type { FaceMesh as MediaPipeFaceMesh, NormalizedLandmark, Results } from "@mediapipe/face_mesh";

const READER_DB_NAME = "motion-pdf-reader";
const READER_STORE_NAME = "reader-state";
const SESSION_STATE_KEY = "session-state";
const SAVED_SCORES_KEY = "saved-scores";
const MOTION_DETECTION_ENABLED_KEY = "motion-detection-enabled";
const PAGE_TURN_COOLDOWN_MS = 1200;
const MOUTH_RATIO_SMOOTHING_ALPHA = 0.55;
const MOUTH_OPEN_RATIO_THRESHOLD = 0.36;
const MOUTH_CLOSE_RATIO_THRESHOLD = 0.18;
const MOUTH_HOLD_DURATION_MS = 500;
const MOUTH_MIN_OPEN_MS = 120;
const MOUTH_DOUBLE_TAP_WINDOW_MS = 600;
const MOUTH_STABLE_FRAME_TARGET = 2;

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

type MouthPhase = "idle" | "open" | "first-closed" | "waiting-close";

type MouthDetectionState = {
  phase: MouthPhase;
  startedAt: number | null;
  closedAt: number | null;
  stableFrames: number;
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
  const [cameraPermission, setCameraPermission] = useState<"unknown" | "prompt" | "granted" | "denied">(
    "unknown"
  );
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
  const gestureTextRef = useRef("PDF를 먼저 불러오면 다음 단계가 쉬워집니다.");
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
        const [
          storedScores,
          sessionState,
          storedMotionDetectionEnabled,
        ] = await Promise.all([
          readReaderValue<SavedScore[]>(SAVED_SCORES_KEY),
          readReaderValue<SessionState>(SESSION_STATE_KEY),
          readReaderValue<boolean>(MOTION_DETECTION_ENABLED_KEY),
        ]);

        if (cancelled) {
          return;
        }

        setSavedScores(storedScores ?? []);
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

        faceMesh.onResults((results: Results) => {
          if (cancelled) return;
          if (!results.multiFaceLandmarks?.length) {
            smoothedMouthRef.current = null;
            mouthStateRef.current = { phase: "idle", startedAt: null, closedAt: null, stableFrames: 0 };
            updateGestureText("얼굴이 카메라에 보이게 해 주세요");
            return;
          }
          const landmarks = results.multiFaceLandmarks[0];

          {
            const rawMouth = computeMouthOpenRatio(landmarks);
            const mouthRatio = getSmoothedMouthRatio(smoothedMouthRef, rawMouth);
            const mouthNow = performance.now();
            const ms = mouthStateRef.current;
            const idleState: MouthDetectionState = { phase: "idle", startedAt: null, closedAt: null, stableFrames: 0 };
            if (ms.phase === "waiting-close") {
              if (mouthRatio < MOUTH_CLOSE_RATIO_THRESHOLD) {
                mouthStateRef.current = idleState;
              }
            } else if (ms.phase === "first-closed") {
              if (ms.closedAt !== null && mouthNow - ms.closedAt > MOUTH_DOUBLE_TAP_WINDOW_MS) {
                mouthStateRef.current = { ...idleState, stableFrames: mouthRatio > MOUTH_OPEN_RATIO_THRESHOLD ? 1 : 0 };
              } else if (mouthRatio > MOUTH_OPEN_RATIO_THRESHOLD) {
                const nextFrames = ms.stableFrames + 1;
                if (nextFrames >= MOUTH_STABLE_FRAME_TARGET) {
                  handleGesture("left");
                  mouthStateRef.current = { phase: "waiting-close", startedAt: null, closedAt: null, stableFrames: 0 };
                } else {
                  mouthStateRef.current = { ...ms, stableFrames: nextFrames };
                }
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
          updateGestureText("입모양 감지 준비됨");
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

  const handleGesture = (direction: "left" | "right") => {
    const now = performance.now();
    if (now - lastSwitchTimeRef.current < PAGE_TURN_COOLDOWN_MS) {
      return false;
    }

    const totalPages = pageCountRef.current;
    const currentPage = currentPageRef.current;
    if (totalPages === 0) {
      updateGestureText("PDF를 먼저 불러와 주세요");
      return false;
    }

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

  const computeMouthOpenRatio = (landmarks: NormalizedLandmark[]) => {
    const leftCorner = landmarks[61];
    const rightCorner = landmarks[291];
    const width = Math.hypot(rightCorner.x - leftCorner.x, rightCorner.y - leftCorner.y);
    if (width === 0) return 0;
    const upper = landmarks[13];
    const lower = landmarks[14];
    return Math.hypot(upper.x - lower.x, upper.y - lower.y) / width;
  };

  const getSmoothedMouthRatio = (targetRef: React.MutableRefObject<number | null>, nextValue: number) => {
    if (targetRef.current === null) {
      targetRef.current = nextValue;
      return nextValue;
    }

    targetRef.current =
      targetRef.current * (1 - MOUTH_RATIO_SMOOTHING_ALPHA) + nextValue * MOUTH_RATIO_SMOOTHING_ALPHA;
    return targetRef.current;
  };

  const canGoPrevious = currentPage > 1;
  const canGoNext = pageCount > 0 && currentPage < pageCount;
  const hasLoadedPdf = pdfDoc !== null;
  const cameraPermissionOn = cameraPermission === "granted";
  const isCameraReady = cameraEnabled && cameraPermissionOn;
  const cameraStatusText = isCameraReady ? "준비됨" : cameraPermission === "denied" ? "권한 필요" : "대기 중";
  const cameraButtonText = cameraEnabled
    ? "모션 감지 끄기"
    : cameraPermission === "denied"
      ? "모션 감지 권한 확인"
      : "모션 감지 켜기";
  const viewerTitle = hasLoadedPdf ? `${currentFileName ?? "불러온 악보"} (${currentPage}/${pageCount})` : "악보";
  const gestureOverlayText = (() => {
    if (gestureText.includes("인식됨")) {
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
                <strong>입을 0.5초 벌려 유지</strong>
                <span>다음 페이지로 이동합니다. 이동 후 입을 닫아 주세요.</span>
              </div>
            </li>
            <li>
              <span className="motion-guide-step">3.</span>
              <div>
                <strong>입을 짧게 두 번 벌리기</strong>
                <span className="motion-guide-result">이전 페이지 · 한 번 닫고 바로 다시 벌려 주세요.</span>
              </div>
            </li>
            <li>
              <span className="motion-guide-step">4.</span>
              <div>
                <strong>이동 후 입 닫기</strong>
                <span>입을 닫으면 다음 동작을 받을 수 있습니다.</span>
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
          <div className="panel-card compact-panel desktop-motion-guide">{motionGuideSection}</div>
          <section className="panel-card compact-panel" aria-label="입모양 감지 상태">
            <h2>입모양 감지</h2>
            <p role="status">{gestureText}</p>
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
