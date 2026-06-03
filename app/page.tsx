"use client";

import { useEffect, useRef, useState } from "react";
import type { Camera as MediaPipeCamera } from "@mediapipe/camera_utils";
import type { FaceMesh as MediaPipeFaceMesh, NormalizedLandmark, Results } from "@mediapipe/face_mesh";

const READER_DB_NAME = "motion-pdf-reader";
const READER_STORE_NAME = "reader-state";
const SESSION_STATE_KEY = "session-state";
const SAVED_SCORES_KEY = "saved-scores";

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
  const [gestureText, setGestureText] = useState("카메라 권한을 켜면 윙크 인식이 시작됩니다.");
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [pageCount, setPageCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [currentFileName, setCurrentFileName] = useState<string | null>(null);
  const [savedScores, setSavedScores] = useState<SavedScore[]>([]);
  const [isLibraryOpen, setIsLibraryOpen] = useState(false);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraPermission, setCameraPermission] = useState<"unknown" | "prompt" | "granted" | "denied">(
    "unknown"
  );

  const lastGestureRef = useRef<"left" | "right" | "none">("none");
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
        const [storedScores, sessionState] = await Promise.all([
          readReaderValue<SavedScore[]>(SAVED_SCORES_KEY),
          readReaderValue<SessionState>(SESSION_STATE_KEY),
        ]);

        if (cancelled) {
          return;
        }

        setSavedScores(storedScores ?? []);

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
    let cancelled = false;

    const stopCamera = async () => {
      winkCandidateRef.current = "none";
      winkFrameCountRef.current = 0;
      lastGestureRef.current = "none";

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
          setGestureText("카메라 권한은 허용됨 상태입니다. 필요할 때 카메라 사용을 시작하세요.");
        } else {
          setGestureText("카메라 권한이 OFF 상태입니다.");
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
          if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            setGestureText("얼굴을 인식할 수 없음");
            winkCandidateRef.current = "none";
            winkFrameCountRef.current = 0;
            return;
          }

          const landmarks = results.multiFaceLandmarks[0];
          const wink = computeWink(landmarks);

          if (wink === "none") {
            setGestureText("오른쪽 윙크: 다음 페이지, 왼쪽 윙크: 이전 페이지");
            winkCandidateRef.current = "none";
            winkFrameCountRef.current = 0;
            lastGestureRef.current = "none";
            return;
          }

          if (winkCandidateRef.current === wink) {
            winkFrameCountRef.current += 1;
          } else {
            winkCandidateRef.current = wink;
            winkFrameCountRef.current = 1;
          }

          if (winkFrameCountRef.current >= 2 && wink !== lastGestureRef.current) {
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
          setGestureText("오른쪽 윙크: 다음 페이지, 왼쪽 윙크: 이전 페이지");
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
    lastGestureRef.current = direction;
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

  const computeWink = (landmarks: NormalizedLandmark[]) => {
    const leftEyeRatio = computeEyeRatio(landmarks, 33, 133, 159, 145);
    const rightEyeRatio = computeEyeRatio(landmarks, 362, 263, 386, 374);
    const closedThreshold = 0.16;
    const openThreshold = 0.24;

    const leftClosed = leftEyeRatio < closedThreshold;
    const rightClosed = rightEyeRatio < closedThreshold;
    const leftOpen = leftEyeRatio > openThreshold;
    const rightOpen = rightEyeRatio > openThreshold;

    if (rightClosed && leftOpen) return "right";
    if (leftClosed && rightOpen) return "left";
    return "none";
  };

  const canGoPrevious = currentPage > 1;
  const canGoNext = pageCount > 0 && currentPage < pageCount;
  const hasLoadedPdf = pdfDoc !== null;
  const cameraPermissionOn = cameraPermission === "granted";
  const cameraStatusText = cameraPermissionOn ? "ON" : "OFF";
  const cameraButtonText = cameraEnabled
    ? "카메라 사용 중지"
    : cameraPermissionOn
      ? "카메라 사용 시작"
      : "카메라 권한 허용하기";

  return (
    <main className="app-shell">
      <header>
        <h1>Hands-Free PDF Reader</h1>
        <p>오른쪽 눈 윙크로 다음 페이지, 왼쪽 눈 윙크로 이전 페이지</p>
      </header>

      <section className="controls">
        <label className="file-label">
          PDF 파일 선택
          <input type="file" accept="application/pdf" onChange={onFileChange} />
        </label>
        <button type="button" className="secondary-button" onClick={saveCurrentPdf} disabled={!hasLoadedPdf}>
          PDF 저장
        </button>
        <button
          type="button"
          className="secondary-button"
          onClick={() => setIsLibraryOpen((prev) => !prev)}
          aria-expanded={isLibraryOpen}
        >
          저장된 악보 {isLibraryOpen ? "닫기" : "보기"}
        </button>
        <button
          type="button"
          className={`camera-toggle ${cameraPermissionOn ? "is-on" : "is-off"}`}
          onClick={() => setCameraEnabled((prev) => !prev)}
          aria-pressed={cameraEnabled}
        >
          {cameraButtonText}
          <span>{cameraStatusText}</span>
        </button>
        <div className="page-buttons" aria-label="페이지 이동 버튼">
          <button type="button" onClick={() => queuePage(currentPage - 1)} disabled={!canGoPrevious}>
            이전 페이지
          </button>
          <button type="button" onClick={() => queuePage(currentPage + 1)} disabled={!canGoNext}>
            다음 페이지
          </button>
        </div>
        <div className="status">
          <span>{pageText}</span>
          <span>{gestureText}</span>
          {currentFileName ? <span>현재 악보: {currentFileName}</span> : null}
        </div>
      </section>

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

      <div className="pdf-viewer">
        <canvas ref={canvasRef} />
      </div>

      <video ref={videoRef} autoPlay muted playsInline className="camera-feed-hidden" />
    </main>
  );
}
