import React, { useEffect, useMemo, useRef, useState } from "react";
import JSZip from "jszip";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

const PRESETS = {
  passport: { label: "Passport 3:4", width: 300, height: 400, ratio: 3 / 4 },
  id: { label: "ID Portrait 4:5", width: 800, height: 1000, ratio: 4 / 5 },
  square: { label: "Square 1:1", width: 1000, height: 1000, ratio: 1 },
  social: { label: "Social 4:5", width: 1080, height: 1350, ratio: 4 / 5 },
  story: { label: "Story 9:16", width: 1080, height: 1920, ratio: 9 / 16 }
};

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

function App() {
  const [items, setItems] = useState([]);
  const [preset, setPreset] = useState("passport");
  const [customWidth, setCustomWidth] = useState(300);
  const [customHeight, setCustomHeight] = useState(400);
  const [quality, setQuality] = useState(90);
  const [format, setFormat] = useState("image/jpeg");
  const [smartCrop, setSmartCrop] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Ready");
  const [dragging, setDragging] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState("");
  const [previewId, setPreviewId] = useState(null);
  const [faceDetectorReady, setFaceDetectorReady] = useState(false);
  const [faceDetectorError, setFaceDetectorError] = useState("");
  const inputRef = useRef(null);
  const detectorRef = useRef(null);

  const target = useMemo(() => {
    if (preset === "custom") {
      const width = Math.max(1, Number(customWidth) || 1);
      const height = Math.max(1, Number(customHeight) || 1);
      return { width, height, ratio: width / height, label: "Custom" };
    }
    return PRESETS[preset];
  }, [preset, customWidth, customHeight]);

  useEffect(() => {
    return () => {
      items.forEach((item) => URL.revokeObjectURL(item.preview));
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      detectorRef.current?.close?.();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadDetector() {
      if (!smartCrop || detectorRef.current) return;
      try {
        setFaceDetectorError("");
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: FACE_MODEL_URL,
            delegate: "GPU"
          },
          runningMode: "IMAGE",
          minDetectionConfidence: 0.45
        });
        if (!cancelled) {
          detectorRef.current = detector;
          setFaceDetectorReady(true);
        } else {
          detector.close?.();
        }
      } catch {
        if (!cancelled) {
          setFaceDetectorError("Smart crop model could not load; center crop will be used.");
          setFaceDetectorReady(false);
        }
      }
    }

    loadDetector();
    return () => { cancelled = true; };
  }, [smartCrop]);

  const addFiles = (fileList) => {
    const selected = Array.from(fileList || []).filter((file) =>
      file.type.startsWith("image/")
    );
    if (!selected.length) return;

    const incoming = selected.map((file, index) => ({
      id: `${file.name}-${file.size}-${file.lastModified}-${index}-${crypto.randomUUID?.() || Math.random()}`,
      file,
      name: file.name,
      preview: URL.createObjectURL(file),
      size: file.size,
      face: null,
      faceState: smartCrop ? "pending" : "off",
      previewCrop: null
    }));

    setItems((prev) => [...prev, ...incoming]);
    setStatus(`${incoming.length} photo${incoming.length === 1 ? "" : "s"} added`);
  };

  const removeItem = (id) => {
    setItems((prev) => {
      const found = prev.find((item) => item.id === id);
      if (found) {
        URL.revokeObjectURL(found.preview);
        if (found.previewCrop) URL.revokeObjectURL(found.previewCrop);
      }
      return prev.filter((item) => item.id !== id);
    });
    if (previewId === id) setPreviewId(null);
  };

  const clearAll = () => {
    items.forEach((item) => {
      URL.revokeObjectURL(item.preview);
      if (item.previewCrop) URL.revokeObjectURL(item.previewCrop);
    });
    setItems([]);
    setProgress(0);
    setStatus("Ready");
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    setDownloadUrl("");
    setPreviewId(null);
  };

  const resetDownload = () => {
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    setDownloadUrl("");
  };

  const getFace = async (file) => {
    if (!detectorRef.current) return null;
    const bitmap = await createImageBitmap(file);
    try {
      const result = detectorRef.current.detect(bitmap);
      if (!result.detections?.length) return null;
      const box = result.detections[0].boundingBox;
      return {
        x: box.originX,
        y: box.originY,
        width: box.width,
        height: box.height,
        score: result.detections[0].categories?.[0]?.score ?? 0
      };
    } finally {
      bitmap.close();
    }
  };

  const calculateCrop = (sourceW, sourceH, targetRatio, face) => {
    const sourceRatio = sourceW / sourceH;
    let sw = sourceW;
    let sh = sourceH;

    if (sourceRatio > targetRatio) sw = Math.round(sourceH * targetRatio);
    else if (sourceRatio < targetRatio) sh = Math.round(sourceW / targetRatio);

    let sx = Math.round((sourceW - sw) / 2);
    let sy = Math.round((sourceH - sh) / 2);

    if (face) {
      const faceCenterX = face.x + face.width / 2;
      const faceCenterY = face.y + face.height / 2;

      // Put the face around 40% down the final frame, a useful
      // passport/ID composition without changing the crop scale.
      const desiredFaceY = sy + sh * 0.40;

      sx = Math.round(faceCenterX - sw / 2);
      sy = Math.round(faceCenterY - desiredFaceY + sh * 0.40);

      // The horizontal calculation above centers the face.
      // The vertical calculation keeps the face slightly above center.
      sy = Math.round(faceCenterY - sh * 0.40);

      sx = Math.max(0, Math.min(sx, sourceW - sw));
      sy = Math.max(0, Math.min(sy, sourceH - sh));
    }

    return { sx, sy, sw, sh };
  };

  const makePreview = async (item, face) => {
    const bitmap = await createImageBitmap(item.file);
    const crop = calculateCrop(bitmap.width, bitmap.height, target.ratio, face);

    const canvas = document.createElement("canvas");
    canvas.width = 360;
    canvas.height = Math.round(360 / target.ratio);
    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, canvas.width, canvas.height);
    bitmap.close();

    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob ? URL.createObjectURL(blob) : null), "image/jpeg", 0.86);
    });
  };

  const analyzeAll = async () => {
    if (!items.length || !smartCrop) return;
    if (!detectorRef.current) {
      setStatus("Loading face detector…");
      return;
    }

    setAnalyzing(true);
    setProgress(0);

    const updated = [...items];
    let found = 0;

    for (let i = 0; i < updated.length; i++) {
      const item = updated[i];
      setStatus(`Finding faces ${i + 1} of ${updated.length}…`);

      try {
        const face = await getFace(item.file);
        const previewCrop = await makePreview(item, face);
        if (item.previewCrop) URL.revokeObjectURL(item.previewCrop);

        updated[i] = {
          ...item,
          face,
          faceState: face ? "found" : "none",
          previewCrop
        };
        if (face) found++;
      } catch {
        updated[i] = { ...item, face: null, faceState: "none" };
      }

      setItems([...updated]);
      setProgress(Math.round(((i + 1) / updated.length) * 100));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    setAnalyzing(false);
    setStatus(
      found
        ? `Smart crop ready — ${found} face${found === 1 ? "" : "s"} detected`
        : "No faces detected — center crop will be used"
    );
  };

  const processImage = async (item, index) => {
    const bitmap = await createImageBitmap(item.file);
    const face = smartCrop && item.faceState === "found" ? item.face : null;
    const crop = calculateCrop(bitmap.width, bitmap.height, target.ratio, face);

    const canvas = document.createElement("canvas");
    canvas.width = target.width;
    canvas.height = target.height;

    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(
      bitmap,
      crop.sx, crop.sy, crop.sw, crop.sh,
      0, 0, target.width, target.height
    );
    bitmap.close();

    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(
        (result) => result ? resolve(result) : reject(new Error("Could not encode image")),
        format,
        format === "image/png" ? undefined : quality / 100
      );
    });

    const ext = format === "image/png" ? "png" : format === "image/webp" ? "webp" : "jpg";
    const cleanName = item.file.name.replace(/\.[^/.]+$/, "");
    return {
      blob,
      filename: `${String(index + 1).padStart(3, "0")}_${cleanName}_${target.width}x${target.height}.${ext}`
    };
  };

  const convertAll = async () => {
    if (!items.length || processing) return;

    resetDownload();
    setProcessing(true);
    setProgress(0);

    if (smartCrop && faceDetectorReady && items.some((x) => x.faceState === "pending")) {
      await analyzeAll();
    }

    const zip = new JSZip();
    const failed = [];

    for (let i = 0; i < items.length; i++) {
      setStatus(`Processing ${i + 1} of ${items.length}: ${items[i].name}`);
      try {
        const result = await processImage(items[i], i);
        zip.file(result.filename, result.blob);
      } catch {
        failed.push(items[i].name);
      }
      setProgress(Math.round(((i + 1) / items.length) * 90));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    setStatus("Building ZIP archive…");
    const zipBlob = await zip.generateAsync(
      { type: "blob", compression: "DEFLATE", compressionOptions: { level: 6 } },
      (metadata) => setProgress(90 + Math.round(metadata.percent / 10))
    );

    const url = URL.createObjectURL(zipBlob);
    setDownloadUrl(url);
    setProcessing(false);
    setProgress(100);
    setStatus(
      failed.length
        ? `Finished with ${failed.length} failed photo${failed.length === 1 ? "" : "s"}`
        : `Done — ${items.length} photo${items.length === 1 ? "" : "s"} converted`
    );
  };

  const formatLabel = format === "image/png" ? "PNG" : format === "image/webp" ? "WebP" : "JPG";
  const previewItem = items.find((item) => item.id === previewId);

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="grid-overlay" />

      <header className="topbar">
        <div className="brand" onClick={clearAll}>
          <div className="brand-mark"><span>✦</span></div>
          <div><strong>PhotoBatch</strong><small>bulk image studio</small></div>
        </div>
        <div className="privacy-pill"><span className="pulse-dot" />Processed locally</div>
      </header>

      <main className="main">
        <section className="hero">
          <div className="eyebrow"><span /> PRIVATE IMAGE PROCESSING</div>
          <h1>Resize <em>hundreds</em> of photos<br />in one clean pass.</h1>
          <p>Crop, resize, compress and package your photos into one ZIP. Nothing is uploaded to a server.</p>
        </section>

        <section
          className={`drop-zone ${dragging ? "is-dragging" : ""}`}
          onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={(e) => { e.preventDefault(); setDragging(false); }}
          onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}
          onClick={() => inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" accept="image/*" multiple hidden
            onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
          <div className="drop-orbit"><div className="orbit orbit-a" /><div className="orbit orbit-b" /><div className="upload-symbol">↥</div></div>
          <h2>{dragging ? "Drop them here" : "Drop your photos here"}</h2>
          <p>or click to browse · JPG, PNG, WebP and more</p>
          <span className="browse-chip">Choose photos</span>
        </section>

        <section className="workspace">
          <aside className="settings-card">
            <div className="section-heading">
              <div><span className="section-kicker">01 / OUTPUT</span><h3>Choose a format</h3></div>
              <span className="count-badge">{items.length}</span>
            </div>

            <label className="field-label">Preset</label>
            <div className="preset-grid">
              {Object.entries(PRESETS).map(([key, value]) => (
                <button key={key} className={`preset ${preset === key ? "active" : ""}`}
                  onClick={() => { setPreset(key); resetDownload(); }}>
                  <span className="preset-icon">{key === "passport" ? "▯" : key === "square" ? "□" : key === "story" ? "▥" : "▤"}</span>
                  <span><b>{value.label}</b><small>{value.width} × {value.height}</small></span>
                </button>
              ))}
              <button className={`preset ${preset === "custom" ? "active" : ""}`}
                onClick={() => { setPreset("custom"); resetDownload(); }}>
                <span className="preset-icon">⌗</span><span><b>Custom</b><small>your dimensions</small></span>
              </button>
            </div>

            {preset === "custom" && (
              <div className="custom-dimensions">
                <div><label>Width</label><input type="number" min="1" value={customWidth} onChange={(e) => setCustomWidth(e.target.value)} /></div>
                <span>×</span>
                <div><label>Height</label><input type="number" min="1" value={customHeight} onChange={(e) => setCustomHeight(e.target.value)} /></div>
              </div>
            )}

            <div className="settings-divider" />

            <div className="smart-card">
              <div className="smart-top">
                <div>
                  <span className="smart-badge">AI CROP</span>
                  <b>Smart face positioning</b>
                </div>
                <button className={`switch ${smartCrop ? "on" : ""}`} onClick={() => setSmartCrop(!smartCrop)} aria-label="Toggle smart crop">
                  <span />
                </button>
              </div>
              <p>Detects a face locally and shifts the crop so the subject stays naturally positioned.</p>
              {smartCrop && (
                <div className={`detector-status ${faceDetectorReady ? "ready" : ""}`}>
                  <span />{faceDetectorReady ? "Face detector ready" : faceDetectorError || "Loading face detector…"}
                </div>
              )}
            </div>

            <button className="analyze-button" disabled={!smartCrop || !items.length || !faceDetectorReady || analyzing}
              onClick={analyzeAll}>
              {analyzing ? "Analyzing faces…" : "⌁ Analyze & Preview Crop"}
            </button>

            <div className="settings-divider" />

            <div className="setting-row">
              <div><label className="field-label">Format</label><small>Output file type</small></div>
              <select value={format} onChange={(e) => { setFormat(e.target.value); resetDownload(); }}>
                <option value="image/jpeg">JPG</option><option value="image/png">PNG</option><option value="image/webp">WebP</option>
              </select>
            </div>

            {format !== "image/png" && (
              <div className="quality-control">
                <div className="quality-head"><label className="field-label">Quality</label><strong>{quality}%</strong></div>
                <input type="range" min="10" max="100" value={quality} onChange={(e) => { setQuality(Number(e.target.value)); resetDownload(); }} />
                <div className="range-labels"><span>Smaller</span><span>Sharper</span></div>
              </div>
            )}

            <div className="privacy-note">
              <span>⌁</span><div><b>Private by design</b><p>Images and face detection run inside your browser. No image upload.</p></div>
            </div>
          </aside>

          <section className="results-card">
            <div className="section-heading">
              <div><span className="section-kicker">02 / PHOTOS</span><h3>{items.length ? "Ready to convert" : "Your photos"}</h3></div>
              {items.length > 0 && <button className="text-button" onClick={clearAll}>Clear all</button>}
            </div>

            {items.length === 0 ? (
              <div className="empty-state"><div className="empty-icon">◎</div><h4>No photos yet</h4><p>Add your images above and they will appear here.</p></div>
            ) : (
              <>
                <div className="photo-meta"><span><b>{items.length}</b> selected</span><span className="target-chip">→ {target.width} × {target.height} · {formatLabel}</span></div>
                <div className="photo-grid">
                  {items.map((item, index) => (
                    <article className="photo-card" key={item.id} style={{ "--delay": `${Math.min(index, 12) * 35}ms` }}>
                      <div className="photo-frame" onClick={() => setPreviewId(item.id)}>
                        <img src={item.previewCrop || item.preview} alt={item.name} />
                        <button className="remove-button" onClick={(e) => { e.stopPropagation(); removeItem(item.id); }}>×</button>
                        <span className="index-badge">{String(index + 1).padStart(2, "0")}</span>
                        {smartCrop && <span className={`face-badge ${item.faceState}`}>{item.faceState === "found" ? "FACE" : item.faceState === "none" ? "CENTER" : "SCAN"}</span>}
                      </div>
                      <div className="photo-name" title={item.name}>{item.name}</div>
                      <div className="photo-size">{formatBytes(item.size)}</div>
                    </article>
                  ))}
                </div>
              </>
            )}
          </section>
        </section>

        <section className="convert-panel">
          <div className="conversion-info">
            <div className="conversion-icon">✦</div>
            <div><span className="section-kicker">03 / EXPORT</span><h3>{status}</h3>
              <p>{items.length ? `${items.length} image${items.length === 1 ? "" : "s"} → ${target.width} × ${target.height} → ${formatLabel} → ZIP` : "Add photos to start a batch conversion."}</p>
            </div>
          </div>
          {processing && <div className="progress-wrap"><div className="progress-track"><div className="progress-value" style={{ width: `${progress}%` }} /></div><strong>{progress}%</strong></div>}
          <div className="export-actions">
            {downloadUrl && !processing && <a className="download-button secondary" href={downloadUrl} download={`photobatch_${target.width}x${target.height}.zip`}>↓ Download ZIP</a>}
            <button className="download-button" onClick={convertAll} disabled={!items.length || processing}>{processing ? "Processing…" : "⚡ Convert & Download ZIP"}</button>
          </div>
        </section>
      </main>

      <footer><span>PhotoBatch v1.1</span><span>Client-side · Face-aware crop · No uploads</span></footer>

      {previewItem && (
        <div className="preview-modal" onClick={() => setPreviewId(null)}>
          <div className="preview-dialog" onClick={(e) => e.stopPropagation()}>
            <button className="preview-close" onClick={() => setPreviewId(null)}>×</button>
            <div className="preview-columns">
              <div><span className="section-kicker">ORIGINAL</span><img src={previewItem.preview} alt="" /></div>
              <div><span className="section-kicker">CROP PREVIEW</span>
                <div className="preview-result">
                  <img src={previewItem.previewCrop || previewItem.preview} alt="" />
                  {previewItem.faceState === "found" && <span className="preview-face">FACE DETECTED</span>}
                </div>
              </div>
            </div>
            <div className="preview-caption">
              <b>{previewItem.name}</b>
              <span>{target.width} × {target.height} · {previewItem.faceState === "found" ? "Face-aware crop" : "Center crop fallback"}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

export default App;
