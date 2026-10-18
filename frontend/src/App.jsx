
import { useCallback, useEffect, useState } from 'react'
import './App.css'

const API = 'http://127.0.0.1:8000'

function App() {
  const [monitor, setMonitor] = useState(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [apiError, setApiError] = useState('')

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/monitor/status`)
      if (!response.ok) throw new Error('Could not fetch monitor status')
      const data = await response.json()
      setMonitor(data)
      setApiError('')
    } catch {
      setApiError('Backend se connection nahi ho raha. Check karo ki FastAPI server running hai.')
    }
  }, [])

  useEffect(() => {
    refreshStatus()
    const timer = setInterval(refreshStatus, 2000)
    return () => clearInterval(timer)
  }, [refreshStatus])

  const startMonitoring = async () => {
    setLoading(true)
    setMessage('')
    try {
      const response = await fetch(`${API}/api/monitor/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: 0 }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not start monitoring')
      setMessage(data.message || 'Monitoring started.')
      await refreshStatus()
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  const stopMonitoring = async () => {
    setLoading(true)
    setMessage('')
    try {
      const response = await fetch(`${API}/api/monitor/stop`, {
        method: 'POST',
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not stop monitoring')
      setMessage(data.message || 'Monitoring stopped.')
      await refreshStatus()
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  const stats = monitor?.stats || {}
  const evaluation = monitor?.evaluation || {}
  const running = Boolean(monitor?.monitor_running)
  const previewReady = running && Number(stats.frames_processed) > 0

  const firstValue = (...values) =>
    values.find((value) => value !== undefined && value !== null)

  const formatValue = (value, digits = 3) =>
    typeof value === 'number' ? value.toFixed(digits) : value ?? '—'

  const faceDetected = firstValue(
    evaluation.face_detected,
    evaluation.faceDetected
  )

  const ear = firstValue(
    evaluation.ear_avg,
    evaluation.average_ear,
    evaluation.avg_ear,
    evaluation.ear
  )

  const mar = firstValue(
    evaluation.mar,
    evaluation.mar_value,
    evaluation.mouth_aspect_ratio
  )

  const status = firstValue(
    evaluation.primary_status,
    evaluation.status,
    running ? 'Monitoring active' : 'Monitoring stopped'
  )

  const drowsy = Boolean(firstValue(
    evaluation.drowsiness_alert,
    evaluation.drowsy_alert,
    evaluation.drowsiness_detected,
    false
  ))

  const yawning = Boolean(firstValue(
    evaluation.yawn_alert,
    evaluation.yawning_alert,
    evaluation.yawning_detected,
    false
  ))

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">DF</div>
          <div>
            <strong>DriveSafe</strong>
            <span>Driver Safety System</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <div className="nav-item active"><span>▦</span> Dashboard</div>
        <div className="nav-item"><span>◉</span> Driver Monitor</div>
        <div className="nav-item"><span>▤</span> System Status</div>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className={`status-dot ${running ? 'green' : ''}`} />
            <div>
              <strong>{running ? 'System active' : 'System idle'}</strong>
              <span>Local monitoring service</span>
            </div>
          </div>
          <small>Driver Fatigue &amp; Distraction Monitor</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">SAFETY MONITORING / OVERVIEW</div>
            <h1>Driver Dashboard</h1>
          </div>
          <div className="topbar-right">
            <span className={`connection-pill ${apiError ? 'offline' : 'online'}`}>
              <span className="status-dot" />
              {apiError ? 'Backend offline' : 'API connection'}
            </span>
            <div className="avatar">YS</div>
          </div>
        </header>

        {apiError && <div className="notice error">{apiError}</div>}
        {message && <div className="notice">{message}</div>}

        <section className="welcome-row">
          <div>
            <h2>Monitoring Overview</h2>
            <p>Real-time driver status and fatigue indicators.</p>
          </div>
          <div className={`monitor-badge ${running ? 'active' : ''}`}>
            <span className="status-dot" />
            {running ? 'MONITORING ACTIVE' : 'MONITORING STOPPED'}
          </div>
        </section>

        <section className="metrics-grid">
          <article className="metric-card">
            <div className="metric-top"><span>Face Detection</span><span className="metric-symbol">◎</span></div>
            <div className="metric-value small-value">
              {faceDetected === true ? 'Detected' : faceDetected === false ? 'Not detected' : '—'}
            </div>
            <div className="metric-foot">{faceDetected === true ? 'Face visible to camera' : 'Waiting for face data'}</div>
          </article>

          <article className="metric-card">
            <div className="metric-top"><span>Eye Aspect Ratio</span><span className="metric-symbol">◉</span></div>
            <div className="metric-value">{formatValue(ear)}</div>
            <div className="metric-foot">EAR · Eye closure indicator</div>
          </article>

          <article className="metric-card">
            <div className="metric-top"><span>Mouth Aspect Ratio</span><span className="metric-symbol">↔</span></div>
            <div className="metric-value">{formatValue(mar)}</div>
            <div className="metric-foot">MAR · Mouth opening indicator</div>
          </article>

          <article className="metric-card">
            <div className="metric-top"><span>Processing Speed</span><span className="metric-symbol">ϟ</span></div>
            <div className="metric-value">{formatValue(stats.fps, 1)}<small> FPS</small></div>
            <div className="metric-foot">{stats.frames_processed ?? 0} frames processed</div>
          </article>
        </section>

        <section className="content-grid">
          <article className="panel camera-panel">
            <div className="panel-heading">
              <div>
                <h3>Live Driver Monitor</h3>
                <p>Camera processing session</p>
              </div>
              <span className={`live-tag ${running ? '' : 'muted'}`}>
                <span className="status-dot" />
                {running ? 'LIVE' : 'IDLE'}
              </span>
            </div>

            {previewReady ? (
              <div className="camera-stream-frame">
                <img
                  className="camera-stream"
                  src="http://127.0.0.1:8000/api/monitor/stream"
                  alt="Live camera preview with face landmarks"
                />
              </div>
            ) : running ? (
              <div className="camera-placeholder">
                <div className="camera-icon">◉</div>
                <h3>Preparing camera preview</h3>
                <p>Monitoring is active. The preview will appear as soon as the first frame is processed.</p>
                <span className="preview-note">WAITING FOR FIRST FRAME</span>
              </div>
            ) : (
            <div className="camera-placeholder">
              <div className="camera-icon">◉</div>
              <h3>Camera preview unavailable</h3>
              <p>Start monitoring to begin camera processing.</p>
              <span className="preview-note">CAMERA MONITORING STOPPED</span>
            </div>
            )}

            <div className="camera-footer">
              <div><span className="mini-label">CAMERA</span><strong>Device {monitor?.config?.camera_index ?? 0}</strong></div>
              <div><span className="mini-label">FRAMES</span><strong>{stats.frames_processed ?? 0}</strong></div>
            </div>
          </article>

          <article className="panel alerts-panel">
            <div className="panel-heading">
              <div>
                <h3>Driver Safety Status</h3>
                <p>Latest backend evaluation</p>
              </div>
              <span className="shield-icon">✓</span>
            </div>

            <div className={`overall-status ${drowsy || yawning ? 'warning' : ''}`}>
              <div className="overall-icon">{drowsy || yawning ? '!' : '✓'}</div>
              <div>
                <strong>{String(status)}</strong>
                <span>{drowsy || yawning ? 'Attention required' : 'Current evaluation status'}</span>
              </div>
            </div>

            <div className="alert-row">
              <div className="alert-icon eye">◉</div>
              <div className="alert-copy"><strong>Drowsiness Alert</strong><span>Prolonged eye closure</span></div>
              <span className={`alert-state ${drowsy ? 'warning' : ''}`}>{drowsy ? 'ALERT' : 'CLEAR'}</span>
            </div>

            <div className="alert-row">
              <div className="alert-icon mouth">↔</div>
              <div className="alert-copy"><strong>Yawning Alert</strong><span>Prolonged mouth opening</span></div>
              <span className={`alert-state ${yawning ? 'warning' : ''}`}>{yawning ? 'ALERT' : 'CLEAR'}</span>
            </div>

            <div className="alert-row">
              <div className="alert-icon face">◎</div>
              <div className="alert-copy"><strong>Face Visibility</strong><span>Driver face in camera view</span></div>
              <span className={`alert-state ${faceDetected === false ? 'warning' : ''}`}>
                {faceDetected === true ? 'CLEAR' : faceDetected === false ? 'CHECK' : '—'}
              </span>
            </div>

            <div className="control-area">
              <button className="start-button" onClick={startMonitoring} disabled={loading || running}>
                {loading ? 'Please wait…' : '▶  Start Monitoring'}
              </button>
              <button className="stop-button" onClick={stopMonitoring} disabled={loading || !running}>
                ■  Stop Monitoring
              </button>
            </div>
          </article>
        </section>

        <footer className="footer">
          <span>DriveSafe · Driver Fatigue &amp; Distraction Safety Monitor</span>
          <span>Backend: FastAPI · Frontend: React</span>
        </footer>
      </main>
    </div>
  )
}

export default App
