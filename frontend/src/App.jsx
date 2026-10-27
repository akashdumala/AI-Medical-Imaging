
import { useState } from "react";
import { Activity, Upload, Scan, FileImage, CheckCircle, AlertCircle, LoaderCircle, Download } from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("xray");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState("");
  const [explanation, setExplanation] = useState("");

  function chooseFile(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    setResult(null);
    setError("");
    setExplanation("");
    setPreview(URL.createObjectURL(selected));
  }

  async function submit(endpoint) {
    if (!file) {
      setError("Choose a file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setExplanation("");

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API}/${endpoint}`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${response.status})`);
      }

      if (endpoint === "explain") {
        const blob = await response.blob();
        setExplanation(URL.createObjectURL(blob));
      } else if (endpoint === "segment-ct") {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "experimental_ct_mask.nii.gz";
        link.click();
        URL.revokeObjectURL(url);
        setResult({ message: "CT mask downloaded successfully." });
      } else {
        setResult(await response.json());
      }
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><Activity size={28} /> <span>MedVision AI</span></div>
        <p className="side-label">WORKSPACE</p>
        <div className="nav-item active"><Scan size={18} /> Imaging Studio</div>
        <div className="sidebar-note">
          <span className="status-dot" /> API connection
          <small>Backend: 127.0.0.1:8000</small>
        </div>
        <div className="side-footer">Educational research demo<br />Not for clinical use</div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div><span className="eyebrow">AI-POWERED WORKSPACE</span><h1>Medical Imaging Studio</h1></div>
          <span className="api-badge"><span className="status-dot" /> Local API</span>
        </header>

        <section className="intro">
          <div>
            <span className="eyebrow">IMAGING ANALYSIS</span>
            <h2>Explore medical images<br />with AI assistance.</h2>
            <p>Upload a chest X-ray or CT scan to run an experimental model.</p>
          </div>
          <div className="intro-icon"><Activity size={64} /></div>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="panel-heading"><div><span className="step">01</span><h3>Upload scan</h3></div><Upload size={20} /></div>

            <div className="mode-switch">
              <button className={mode === "xray" ? "selected" : ""} onClick={() => {setMode("xray");setFile(null);setResult(null);setError("");setExplanation("");setPreview("");}}>
                <FileImage size={17} /> Chest X-ray
              </button>
              <button className={mode === "ct" ? "selected" : ""} onClick={() => {setMode("ct");setFile(null);setResult(null);setError("");setExplanation("");setPreview("");}}>
                <Scan size={17} /> CT scan
              </button>
            </div>

            <label className="drop-zone">
              <Upload size={30} />
              <strong>Choose a scan file</strong>
              <span>{mode === "xray" ? "JPG, JPEG or PNG image" : "NIfTI .nii or .nii.gz CT volume"}</span>
              <input type="file" accept={mode === "xray" ? "image/*" : ".nii,.gz,.nii.gz"} onChange={chooseFile} />
            </label>

            {file && <div className="file-chip"><CheckCircle size={17} /><span>{file.name}</span></div>}

            <div className="action-buttons">
              {mode === "xray" ? (
                <>
                  <button className="primary" disabled={loading || !file} onClick={() => submit("predict")}>
                    {loading ? <LoaderCircle className="spin" size={17}/> : <Activity size={17}/>} Predict X-ray
                  </button>
                  <button className="secondary" disabled={loading || !file} onClick={() => submit("explain")}>
                    Explain with Grad-CAM
                  </button>
                </>
              ) : (
                <button className="primary" disabled={loading || !file} onClick={() => submit("segment-ct")}>
                  {loading ? <LoaderCircle className="spin" size={17}/> : <Scan size={17}/>} Segment CT
                </button>
              )}
            </div>
            <p className="hint">Use sample research data only. Avoid identifiable patient information.</p>
          </div>

          <div className="panel results-panel">
            <div className="panel-heading"><div><span className="step">02</span><h3>Analysis results</h3></div><Activity size={20}/></div>
            {!file && <div className="empty-state"><Scan size={42}/><strong>Ready for analysis</strong><span>Your results will appear here after upload.</span></div>}
            {file && preview && mode === "xray" && <div className="preview"><img src={preview} alt="Selected chest X-ray preview"/><small>Uploaded image preview</small></div>}
            {loading && <div className="loading"><LoaderCircle className="spin" size={25}/> Processing image…</div>}
            {error && <div className="error-box"><AlertCircle size={19}/><span>{error}</span></div>}
            {result && (
              <div className="result-card">
                {result.predicted_class && <><span className="eyebrow">PREDICTED CLASS</span><h2>{result.predicted_class}</h2><p>Model score: {typeof result.model_score === "number" ? (result.model_score * 100).toFixed(2) + "%" : "Not available"}</p><div className="scores">{Object.entries(result.class_scores || {}).map(([label, score]) => <div key={label}><span>{label}</span><strong>{(score * 100).toFixed(2)}%</strong><div className="score-track"><div style={{width: `${Math.max(0, Math.min(100, score * 100))}%`}} /></div></div>)}</div></>}
                {result.message && <p><CheckCircle size={18}/> {result.message}</p>}
                <p className="disclaimer">{result.notice || "Experimental output only. Not a medical diagnosis."}</p>
              </div>
            )}
            {explanation && <div className="gradcam-result"><h4>Grad-CAM visualization</h4><img src={explanation} alt="Grad-CAM explanatory heatmap"/><a href={explanation} download="gradcam_overlay.png">Download visualization</a><p className="disclaimer">A model attention visualization, not a confirmed lesion location.</p></div>}
          </div>
        </section>
        <footer>MedVision AI · Research prototype · No clinical decision-making</footer>
      </main>
    </div>
  );
}
