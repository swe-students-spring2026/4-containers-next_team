(function () {
  "use strict";

  const refreshMs = Number(document.body.dataset.refreshMs || 1000);
  const mlApiBaseUrl =
    document.body.dataset.mlApiBaseUrl || "http://127.0.0.1:8000";

  let currentCameraStream = null;
  let predictionIntervalId = null;
  let refreshIntervalId = null;
  let frameNumber = 0;
  let lastAutoSpokenLabel = "";
  let isSendingFrame = false;

  function updateText(id, value) {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  }

  function updateValue(id, value) {
    const element = document.getElementById(id);
    if (element) {
      element.value = value;
    }
  }

  function buildRecentRow(item) {
    return `
      <tr>
        <td>${item.timestamp ?? "N/A"}</td>
        <td>${item.predicted_label ?? "N/A"}</td>
        <td>${item.confidence ?? 0}</td>
      </tr>
    `;
  }

  async function fetchRecent() {
    const response = await fetch("/api/recent", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Failed to fetch recent predictions");
    }
    return response.json();
  }

  async function fetchStats() {
    const response = await fetch("/api/stats", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Failed to fetch stats");
    }
    return response.json();
  }

  async function refreshPanels() {
    try {
      const [recent, stats] = await Promise.all([fetchRecent(), fetchStats()]);

      updateText("avg-confidence", stats.average_confidence ?? 0);

      const recentTableBody = document.getElementById("recent-table-body");
      if (recentTableBody) {
        if (recent.length) {
          recentTableBody.innerHTML = recent.map(buildRecentRow).join("");
        } else {
          recentTableBody.innerHTML =
            '<tr><td colspan="3">No prediction data available yet.</td></tr>';
        }
      }
    } catch (error) {
      console.error(error);
    }
  }

  function captureFrame(video) {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Could not create canvas context.");
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.8);
  }

  function maybeAutoSpeak(label) {
    if (!window.speakerAPI || !window.speakerAPI.isAutoSpeakEnabled()) {
      return;
    }

    if (!label || label === "N/A") {
      return;
    }

    if (label !== lastAutoSpokenLabel) {
      lastAutoSpokenLabel = label;
      window.speakerAPI.speakText(label);
    }
  }

  async function sendFrameForPrediction() {
    if (isSendingFrame) {
      return;
    }

    const video = document.getElementById("camera-preview");
    if (!video || video.readyState < 2 || !currentCameraStream) {
      return;
    }

    isSendingFrame = true;

    try {
      frameNumber += 1;
      const image = captureFrame(video);

      const response = await fetch(`${mlApiBaseUrl}/predict-frame`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          image,
          frame_number: frameNumber
        })
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Prediction request failed: ${text}`);
      }

      const data = await response.json();
      const timestamp = new Date().toISOString();
      const label = data.predicted_label ?? "N/A";
      const confidence = data.confidence ?? 0;

      updateText("latest-label", label);
      updateText("latest-label-inline", label);
      updateText("latest-confidence", confidence);
      updateText("latest-timestamp", timestamp);
      updateValue("accumulated-text", label);

      document.dispatchEvent(
        new CustomEvent("prediction-update", {
          detail: {
            label,
            confidence,
            timestamp
          }
        })
      );

      await refreshPanels();
      maybeAutoSpeak(label);
    } catch (error) {
      console.error(error);
    } finally {
      isSendingFrame = false;
    }
  }

  function stopPredictionLoop() {
    if (predictionIntervalId) {
      window.clearInterval(predictionIntervalId);
      predictionIntervalId = null;
    }
  }

  function startPredictionLoop() {
    stopPredictionLoop();
    predictionIntervalId = window.setInterval(() => {
      sendFrameForPrediction().catch(console.error);
    }, 800);
  }

  function stopRefreshLoop() {
    if (refreshIntervalId) {
      window.clearInterval(refreshIntervalId);
      refreshIntervalId = null;
    }
  }

  function startRefreshLoop() {
    stopRefreshLoop();
    refreshIntervalId = window.setInterval(() => {
      refreshPanels().catch(console.error);
    }, refreshMs);
  }

  function resetPredictionDisplay() {
    updateText("latest-label", "N/A");
    updateText("latest-label-inline", "N/A");
    updateText("latest-confidence", 0);
    updateText("latest-timestamp", "N/A");
    updateValue("accumulated-text", "N/A");
    lastAutoSpokenLabel = "";
  }

  async function toggleCamera() {
    const video = document.getElementById("camera-preview");
    const button = document.getElementById("camera-toggle-btn");

    if (!video) {
      return;
    }

    if (!window.isSecureContext) {
      alert("Camera access requires localhost or HTTPS.");
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Camera is unavailable in this browser.");
      return;
    }

    if (currentCameraStream) {
      currentCameraStream.getTracks().forEach((track) => track.stop());
      currentCameraStream = null;
      video.srcObject = null;
      stopPredictionLoop();
      resetPredictionDisplay();

      if (button) {
        button.textContent = "Start Camera";
      }

      return;
    }

    try {
      currentCameraStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      video.srcObject = currentCameraStream;
      await video.play();

      if (button) {
        button.textContent = "Stop Camera";
      }

      await sendFrameForPrediction();
      startPredictionLoop();
    } catch (error) {
      console.error(error);
      alert("Unable to access the camera.");
    }
  }

  window.addEventListener("load", function () {
    window.setTimeout(function () {
      document.body.classList.remove("is-preload");
    }, 100);

    const cameraButton = document.getElementById("camera-toggle-btn");
    if (cameraButton) {
      cameraButton.addEventListener("click", function () {
        toggleCamera().catch(console.error);
      });
    }

    refreshPanels().catch(console.error);
    startRefreshLoop();
  });
})();
