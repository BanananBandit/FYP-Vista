import "./App.css";

import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://localhost:8000";

function toFormData(file, settings) {
  const fd = new FormData();
  fd.append("video", file);
  Object.entries(settings).forEach(([k, v]) => fd.append(k, String(v)));
  return fd;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [rows, setRows] = useState([]);
  const [summary, setSummary] = useState(null);
  const [keepIntervals, setKeepIntervals] = useState([]);
  const [busy, setBusy] = useState(false);
  const [rendering, setRendering] = useState(false);
  const [error, setError] = useState("");
  const [analysisMessage, setAnalysisMessage] = useState("");
  const [renderMessage, setRenderMessage] = useState("");
  const [videoUrl, setVideoUrl] = useState(null);
  const [analysisComplete, setAnalysisComplete] = useState(false);

  const configRef = useRef(null);
  const previewRef = useRef(null);
  const resultsRef = useRef(null);
  const outputRef = useRef(null);

  const [settings, setSettings] = useState({
    sample_fps: 2.0,
    segment_seconds: 1.0,
    dark_thresh: 45.0,
    blur_thresh: 60.0,
    shake_thresh: 1.5,
    dark_extent_pct: 40.0,
    blur_extent_pct: 30.0,
    shake_extent_pct: 50.0
  });

  const canAnalyse = useMemo(
    () => !!file && !busy && !rendering,
    [file, busy, rendering]
  );
  const canRender = useMemo(
    () => analysisComplete && !!file && !busy && !rendering,
    [analysisComplete, file, busy, rendering]
  );

  function updateSetting(name, value) {
    setSettings((s) => ({ ...s, [name]: value }));
  }

  function resetResults() {
    setRows([]);
    setSummary(null);
    setKeepIntervals([]);
    setVideoUrl(null);
    setAnalysisComplete(false);
    setAnalysisMessage("");
    setRenderMessage("");
    setError("");
  }

  function scrollToSection(ref) {
    if (ref?.current) {
      ref.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function handleFileChange(e) {
    const selected = e.target.files?.[0] || null;
    setFile(selected);
    resetResults();

    if (selected) {
      setTimeout(() => {
        scrollToSection(configRef);
      }, 150);
    }
  }

  async function runAnalyse() {
    setError("");
    setAnalysisMessage("");
    setRenderMessage("");
    setBusy(true);
    setRows([]);
    setSummary(null);
    setKeepIntervals([]);
    setVideoUrl(null);
    setAnalysisComplete(false);

    try {
      const fd = toFormData(file, settings);

      const res = await fetch(`${API_BASE}/analyse`, {
        method: "POST",
        body: fd
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Analysis failed (${res.status})`);
      }

      const data = await res.json();
      setRows(data.rows || []);
      setSummary(data.summary || null);
      setKeepIntervals(data.keep_intervals || []);
      setAnalysisComplete(true);
      setAnalysisMessage(
        "Analysis successful. Segment classification results are now available."
      );

      setTimeout(() => {
        scrollToSection(resultsRef);
      }, 200);
    } catch (e) {
      setError(e.message || "Unknown error");
      setTimeout(() => {
        scrollToSection(configRef);
      }, 150);
    } finally {
      setBusy(false);
    }
  }

  async function runRender() {
    setError("");
    setRenderMessage("");
    setRendering(true);
    setVideoUrl(null);

    try {
      const fd = toFormData(file, settings);

      const res = await fetch(`${API_BASE}/render`, {
        method: "POST",
        body: fd
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Render failed (${res.status})`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setVideoUrl(url);
      setRenderMessage(
        "Rendering successful. The cleaned output video is ready for preview and download."
      );

      setTimeout(() => {
        scrollToSection(outputRef);
      }, 200);
    } catch (e) {
      setError(e.message || "Unknown error");
      setTimeout(() => {
        scrollToSection(outputRef);
      }, 150);
    } finally {
      setRendering(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero-card">
        <p className="eyebrow">
          Video Intelligent Screening and Trimming Assistant
        </p>
        <h1>VISTA</h1>
        <p className="hero-text">
          A computer vision interface for analysing raw footage, classifying
          unusable segments, and generating a cleaned output video for
          downstream editing workflows.
        </p>
      </header>

      <main className="stack-layout">
        <section className="card compact-card">
          <h2>1. Video Input</h2>
          <p className="section-text">
            Upload a digital video file to begin frame-level analysis. This
            supports the system requirement for accepting source footage and
            preparing it for automated quality assessment.
          </p>

          <label className="upload-box">
            <span className="upload-title">Select video file</span>
            <span className="upload-subtitle">
              Supported formats: MP4, MOV, M4V
            </span>
            <input
              type="file"
              accept="video/mp4,video/quicktime,video/x-m4v"
              onChange={handleFileChange}
            />
          </label>

          {file && (
            <div className="file-meta">
              <strong>{file.name}</strong>
              <span>{Math.round(file.size / 1024 / 1024)} MB</span>
            </div>
          )}
        </section>

        <section className="card" ref={configRef}>
          <h2>2. Analysis Configuration</h2>
          <p className="section-text">
            Configure how the system samples frames, groups them into segments,
            and applies quality criteria for exposure, blur, and motion
            instability.
          </p>

          {error && !analysisComplete && (
            <div className="status error">Analysis error: {error}</div>
          )}

          <div className="settings-grid">
            <div className="settings-group">
              <h3>Sampling Parameters</h3>

              <div className="control">
                <label>Frame Sampling Rate</label>
                <div className="value-row">
                  <span>{settings.sample_fps} fps</span>
                </div>
                <p>
                  Controls how many frames per second are analysed. Higher
                  values improve precision but increase processing time.
                </p>
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={settings.sample_fps}
                  onChange={(e) =>
                    updateSetting("sample_fps", Number(e.target.value))
                  }
                />
              </div>

              <div className="control">
                <label>Segment Length</label>
                <div className="value-row">
                  <span>{settings.segment_seconds} seconds</span>
                </div>
                <p>
                  Defines how footage is grouped before classification. Shorter
                  segments provide more precise trimming decisions.
                </p>
                <input
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.5"
                  value={settings.segment_seconds}
                  onChange={(e) =>
                    updateSetting("segment_seconds", Number(e.target.value))
                  }
                />
              </div>
            </div>

            <div className="settings-group">
              <h3>Quality Detection Thresholds</h3>

              <div className="control">
                <label>Darkness Threshold</label>
                <div className="value-row">
                  <span>{settings.dark_thresh}</span>
                </div>
                <p>
                  Frames below this brightness level are treated as dark.
                  Increasing the threshold makes exposure detection stricter.
                </p>
                <input
                  type="range"
                  min="0"
                  max="255"
                  step="1"
                  value={settings.dark_thresh}
                  onChange={(e) =>
                    updateSetting("dark_thresh", Number(e.target.value))
                  }
                />
              </div>

              <div className="control">
                <label>Blur Threshold</label>
                <div className="value-row">
                  <span>{settings.blur_thresh}</span>
                </div>
                <p>
                  Frames below this sharpness level are flagged as blurred or
                  out of focus. Higher values require sharper footage.
                </p>
                <input
                  type="range"
                  min="0"
                  max="500"
                  step="1"
                  value={settings.blur_thresh}
                  onChange={(e) =>
                    updateSetting("blur_thresh", Number(e.target.value))
                  }
                />
              </div>

              <div className="control">
                <label>Shake Threshold</label>
                <div className="value-row">
                  <span>{settings.shake_thresh}</span>
                </div>
                <p>
                  Segments above this motion value are flagged as unstable.
                  Lower values detect more movement as shake.
                </p>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.1"
                  value={settings.shake_thresh}
                  onChange={(e) =>
                    updateSetting("shake_thresh", Number(e.target.value))
                  }
                />
              </div>
            </div>

            <div className="settings-group">
              <h3>Removal Strictness</h3>
              <p className="mini-text">
                These controls determine how severe a defect must be before a
                flagged segment is removed from the final output.
              </p>

              <div className="control">
                <label>Dark Removal Strictness</label>
                <div className="value-row">
                  <span>{settings.dark_extent_pct}%</span>
                </div>
                <p>
                  Higher values only remove very dark segments. Lower values
                  remove a wider range of underexposed footage.
                </p>
                <input
                  type="range"
                  min="0"
                  max="90"
                  step="5"
                  value={settings.dark_extent_pct}
                  onChange={(e) =>
                    updateSetting("dark_extent_pct", Number(e.target.value))
                  }
                />
              </div>

              <div className="control">
                <label>Blur Removal Strictness</label>
                <div className="value-row">
                  <span>{settings.blur_extent_pct}%</span>
                </div>
                <p>
                  Higher values only remove severely blurred footage. Lower
                  values make the system more aggressive.
                </p>
                <input
                  type="range"
                  min="0"
                  max="90"
                  step="5"
                  value={settings.blur_extent_pct}
                  onChange={(e) =>
                    updateSetting("blur_extent_pct", Number(e.target.value))
                  }
                />
              </div>

              <div className="control">
                <label>Shake Removal Strictness</label>
                <div className="value-row">
                  <span>{settings.shake_extent_pct}%</span>
                </div>
                <p>
                  Higher values only remove highly unstable motion. Lower values
                  make shake filtering more sensitive.
                </p>
                <input
                  type="range"
                  min="0"
                  max="200"
                  step="10"
                  value={settings.shake_extent_pct}
                  onChange={(e) =>
                    updateSetting("shake_extent_pct", Number(e.target.value))
                  }
                />
              </div>
            </div>
          </div>

          <div className="action-row">
            <button
              className="primary-btn"
              disabled={!canAnalyse}
              onClick={runAnalyse}
            >
              {busy ? "Running Analysis..." : "Run Analysis"}
            </button>
          </div>
        </section>

        <section className="card preview-card" ref={previewRef}>
          <h2>3. Source Preview</h2>
          <p className="section-text">
            A compact preview of the uploaded source footage for validation
            before automated analysis.
          </p>

          {file ? (
            <video
              className="video-frame"
              controls
              src={URL.createObjectURL(file)}
            />
          ) : (
            <div className="empty-state">No video selected yet.</div>
          )}
        </section>

        <section className="card" ref={resultsRef}>
          <h2>4. Classification Results</h2>
          <p className="section-text">
            This section presents the segment-level output generated by the
            computer vision pipeline, including usability decisions and quality
            scores.
          </p>

          {analysisMessage && (
            <div className="status success">{analysisMessage}</div>
          )}

          {!analysisComplete ? (
            <div className="empty-state">
              Run the analysis step to generate frame-derived segment
              classifications.
            </div>
          ) : (
            <>
              <div className="metrics-row">
                <div className="metric-box">
                  <span className="metric-label">Total Segments</span>
                  <strong>{summary?.segments ?? 0}</strong>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Kept</span>
                  <strong>{summary?.kept ?? 0}</strong>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Removed</span>
                  <strong>{summary?.removed ?? 0}</strong>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Removed %</span>
                  <strong>{summary?.removed_pct?.toFixed?.(1) ?? 0}%</strong>
                </div>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Start</th>
                      <th>End</th>
                      <th>Brightness</th>
                      <th>Blur</th>
                      <th>Shake</th>
                      <th>Keep</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, idx) => (
                      <tr key={idx}>
                        <td>{Number(row.t_start).toFixed(2)}</td>
                        <td>{Number(row.t_end).toFixed(2)}</td>
                        <td>{Number(row.brightness).toFixed(2)}</td>
                        <td>{Number(row.blur).toFixed(2)}</td>
                        <td>{Number(row.shake).toFixed(2)}</td>
                        <td>{row.keep ? "Yes" : "No"}</td>
                        <td>{row.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>

        <section className="card" ref={outputRef}>
          <h2>5. Output Generation</h2>
          <p className="section-text">
            Once classification is complete, the system generates a cleaned
            output video composed only of usable segments, supporting
            integration with post-production workflows.
          </p>

          {error && analysisComplete && !videoUrl && !rendering && (
            <div className="status error">Rendering error: {error}</div>
          )}

          {renderMessage && (
            <div className="status success">{renderMessage}</div>
          )}

          {!analysisComplete ? (
            <div className="empty-state">
              The render stage is unavailable until analysis has completed
              successfully.
            </div>
          ) : (
            <>
              <div className="action-row">
                <button
                  className="primary-btn"
                  disabled={!canRender}
                  onClick={runRender}
                >
                  {rendering ? "Rendering Output..." : "Render Cleaned Video"}
                </button>
              </div>

              {videoUrl && (
                <div className="output-block">
                  <video className="video-frame" controls src={videoUrl} />
                  <a
                    className="download-link"
                    href={videoUrl}
                    download="vista_edited_output.mp4"
                  >
                    Download Edited Video
                  </a>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
