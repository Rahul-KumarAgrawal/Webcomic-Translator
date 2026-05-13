/**
 * web/static/main.js
 * Client-side logic for CBZ Translator web UI.
 * - SSE client for /progress_stream
 * - Drag-and-drop file upload
 * - Bubble approve / reject / edit
 * - Memory inline editing
 * - Toast notifications
 */

// ── Toast helper ──────────────────────────────────────────────────────────────
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

// ── SSE Queue Progress ────────────────────────────────────────────────────────
function initQueueSSE() {
  const list = document.getElementById("queue-list");
  if (!list) return;

  const evts = new EventSource("/progress_stream");

  evts.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === "ping") return;
    if (data.type === "init") {
      // Clear and re-render
      const ql = document.getElementById("queue-list");
      const cl = document.getElementById("completed-list");
      if (ql) ql.innerHTML = '';
      if (cl) cl.innerHTML = '';
      data.jobs.forEach(job => upsertJobCard(job));
    } else if (data.type === "queued") {
      upsertJobCard(data);
    } else if (data.type === "update") {
      const oldCard = document.getElementById(`job-${data.id}`);
      let oldStatus = "";
      if (oldCard) {
        const badge = oldCard.querySelector(".badge");
        if (badge) oldStatus = badge.textContent.toLowerCase().trim();
      }

      upsertJobCard(data);

      if (oldStatus !== "done" && data.status === "done") {
        if (data.translation_engine === "Rerender") {
          showToast(`✅ ${data.cbz_name} re-rendered successfully!`, "success");
        } else {
          showToast(`✅ ${data.cbz_name} translated successfully!`, "success");
        }
      } else if (data.status === "error") {
        showToast(`❌ ${data.cbz_name}: ${data.error}`, "error");
      }
    } else if (data.type === "deleted") {
      const btn = document.querySelector(`.delete-output-btn[data-cbz="${CSS.escape(data.cbz_name)}"]`);
      if (btn) {
        const card = btn.closest(".card");
        if (card) card.remove();
      }
    }
  };

  evts.onerror = () => {
    console.warn("SSE disconnected, retrying...");
  };
}

function upsertJobCard(job) {
  // Select the correct list based on job type
  const targetListId = job.translation_engine === "Rerender" ? "completed-list" : "queue-list";
  const list = document.getElementById(targetListId);

  let card = document.getElementById(`job-${job.id}`);
  if (!card) {
    card = document.createElement("div");
    card.id = `job-${job.id}`;
    card.className = "card mb-16";
    if (job.translation_engine === "Rerender") {
      card.style.borderLeft = "3px solid #4ade80";
    }

    // Insert at top of the correct list
    if (list.firstChild) {
      list.insertBefore(card, list.firstChild);
    } else {
      list.appendChild(card);
    }
  }

  let badgeClass = "badge-processing";
  if (job.status === "done") badgeClass = "badge-done";
  if (job.status === "error") badgeClass = "badge-error";
  if (job.status === "cancelled") badgeClass = "badge-cancelled";

  const pct = job.total > 0 ? Math.round((job.progress / job.total) * 100) : 0;

  card.innerHTML = `
    <div class="flex-between">
      <div class="flex-gap-8">
        <span style="font-weight:600">${escHtml(job.cbz_name)}</span>
        <span class="badge ${badgeClass}">${job.status || "queued"}</span>
      </div>
      ${(job.status === "done" || job.status === "error" || job.status === "cancelled")
      ? (job.translation_engine === "Rerender"
        ? `<div class="flex-gap-8">
                ${job.status === "done" ? `<span class="badge badge-done" style="font-weight:bold;">✅ Final CBZ Ready in /output!</span>` : ""}
                <a class="btn btn-outline btn-sm" href="/review/${encodeURIComponent(job.cbz_name)}">Review →</a>
                <button class="btn btn-sm" style="background:#4ade80;color:#000;font-weight:600;" onclick="rerenderFromCard(this)" data-cbz="${escAttr(job.cbz_name)}">🔄 Re-render</button>
                <button class="btn btn-danger btn-sm delete-output-btn" data-cbz="${escAttr(job.cbz_name)}" title="Delete output & session data">🗑️ Delete</button>
               </div>`
        : `<div class="flex-gap-8">
                <a class="btn btn-outline btn-sm" href="/review/${encodeURIComponent(job.cbz_name)}">Review →</a>
                <button class="btn btn-danger btn-sm delete-output-btn" data-cbz="${escAttr(job.cbz_name)}" title="Delete output & session data">🗑️ Delete</button>
               </div>`)
      : (!job.status || job.status === "queued" || job.status === "processing"
        ? `<button class="btn btn-danger btn-sm" onclick="cancelJob('${job.id}')" title="Cancel this job">Cancel</button>`
        : "")}
    </div>
    ${job.series ? `<div class="text-sm text-muted mt-8">Series: ${escHtml(job.series)}</div>` : ""}
    ${job.error ? `<div class="text-danger text-sm mt-8">Error: ${escHtml(job.error)}</div>` : ""}
    ${job.status === "processing" ? `
      <div class="progress-bar-wrap mt-8">
        <div class="progress-bar-fill" style="width:${pct}%"></div>
      </div>
      <div class="text-sm text-muted mt-8">
        ${job.progress_text ? escHtml(job.progress_text) : (job.total > 0 ? `Page ${job.progress} / ${job.total}` : "")}
      </div>
    ` : ""}
  `;

  // Re-attach delete handlers for newly rendered buttons
  initDeleteOutput();
}

function cancelJob(jobId) {
  if (confirm("Are you sure you want to cancel this job?")) {
    fetch(`/cancel_job/${jobId}`, { method: 'POST' })
      .catch(err => console.error("Cancel failed", err));
  }
}

// ── Drag-and-drop upload ──────────────────────────────────────────────────────
function initDropZone() {
  const zone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("cbz-file-input");
  const seriesInput = document.getElementById("series-input");
  const srcLangSel = document.getElementById("source-lang-select");
  const tgtLangSel = document.getElementById("target-lang-select");
  const engineSel = document.getElementById("engine-select");
  const inpaintSel = document.getElementById("inpaint-engine-select");
  const mitTransSel = document.getElementById("mit-translator-select");
  const mitLangSel = document.getElementById("mit-target-lang-select");
  const forceToggle = document.getElementById("force-retranslate-toggle");
  if (!zone) return;

  // Manhwa chunk toggle visibility
  const chunkToggle = document.getElementById("chunk-enable-toggle");
  const chunkOpts = document.getElementById("chunk-options");
  if (chunkToggle && chunkOpts) {
    chunkToggle.addEventListener("change", () => {
      chunkOpts.style.display = chunkToggle.checked ? "" : "none";
    });
  }

  zone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => uploadFiles(fileInput.files, seriesInput, srcLangSel, tgtLangSel, engineSel, inpaintSel, mitTransSel, mitLangSel, forceToggle));

  ["dragenter", "dragover"].forEach(evt => {
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add("drag-over"); });
  });
  ["dragleave", "drop"].forEach(evt => {
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove("drag-over"); });
  });
  zone.addEventListener("drop", e => {
    uploadFiles(e.dataTransfer.files, seriesInput, srcLangSel, tgtLangSel, engineSel, inpaintSel, mitTransSel, mitLangSel, forceToggle);
  });
}

async function uploadFiles(files, seriesInput, srcLangSel, tgtLangSel, engineSel, inpaintSel, mitTransSel, mitLangSel, forceToggle) {
  if (!files || files.length === 0) return;
  const series = seriesInput ? seriesInput.value.trim() || "Unknown" : "Unknown";
  const srcLang = srcLangSel ? srcLangSel.value : "";
  const tgtLang = tgtLangSel ? tgtLangSel.value || "eng_Latn" : "eng_Latn";
  const engine = engineSel ? engineSel.value || "nllb" : "nllb";

  // Pipeline fields
  const useMit = engine === "mit";
  const useKoharu = engine === "koharu";
  const inpaintEngine = inpaintSel ? inpaintSel.value || "lama" : "lama";
  const mitTranslator = mitTransSel ? mitTransSel.value || "sugoi" : "sugoi";
  const mitTargetLang = mitLangSel ? mitLangSel.value || "ENG" : "ENG";
  const forceRetranslate = forceToggle ? forceToggle.checked : false;
  const ocrEngineSel = document.getElementById("ocr-engine-select");
  const ocrEngine = ocrEngineSel ? ocrEngineSel.value : "auto";
  const detEngineSel = document.getElementById("detection-engine-select");
  const detEngine = detEngineSel ? detEngineSel.value : "mit";

  // Manhwa chunking
  const chunkToggle = document.getElementById("chunk-enable-toggle");
  const chunkEnabled = chunkToggle ? chunkToggle.checked : false;
  const chunkHeight = chunkEnabled ? parseInt(document.getElementById("chunk-height-input")?.value || "0", 10) : 0;
  const chunkOverlap = chunkEnabled ? parseInt(document.getElementById("chunk-overlap-input")?.value || "0", 10) : 0;

  // OCR Quality
  const superResToggle = document.getElementById("ocr-super-res-toggle");
  const superResEnabled = superResToggle ? superResToggle.checked : false;
  const upscaleFactor = document.getElementById("ocr-upscale-factor-select")?.value || "2";
  const globalUpscaleModeSelect = document.getElementById("global-upscale-mode");
  const globalUpscaleMode = globalUpscaleModeSelect ? globalUpscaleModeSelect.value : "none";

  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".cbz")) {
      showToast(`Skipped ${file.name} — only .cbz files accepted.`, "error");
      continue;
    }

    let finalSrcLang = srcLang;
    let finalOcrEngine = ocrEngine;

    if (srcLang === "") {
      try {
        showToast(`🔍 Detecting language for ${file.name}...`, "info");
        const detectFd = new FormData();
        detectFd.append("file", file);

        const autoEngine = localStorage.getItem("active_autodetect_engine") || "";
        const autoKey = localStorage.getItem(`autodetect_key_${autoEngine}`) || "";

        const detectRes = await fetch(`${DETECT_API_BASE}/autodetect/detect`, {
          method: "POST",
          headers: { "X-API-Key": autoKey },
          body: detectFd
        });
        const detectData = await detectRes.json();
        if (detectData.languages && detectData.languages.length > 0) {
          const topL = detectData.languages[0];
          const dName = topL.name.toLowerCase();
          const dScript = (topL.script || "").toLowerCase();

          if (srcLangSel) {
            let matched = false;
            // Try specific match for Chinese variants
            if (dName.includes("chinese")) {
              const target = (dScript.includes("traditional") || dName.includes("traditional")) ? "traditional" : "simplified";
              for (let i = 0; i < srcLangSel.options.length; i++) {
                const optText = srcLangSel.options[i].text.toLowerCase();
                if (optText.includes("chinese") && optText.includes(target)) {
                  finalSrcLang = srcLangSel.options[i].value;
                  matched = true;
                  break;
                }
              }
            }

            // Fallback to general include match
            if (!matched) {
              for (let i = 0; i < srcLangSel.options.length; i++) {
                if (srcLangSel.options[i].text.toLowerCase().includes(dName)) {
                  finalSrcLang = srcLangSel.options[i].value;
                  matched = true;
                  break;
                }
              }
            }

            if (matched) {
              showToast(`✅ Auto-detected: ${topL.name} ${topL.script ? '(' + topL.script + ')' : ''}`, "success");
            }
          }
        }
      } catch (err) {
        console.error("Auto-detect failed:", err);
        showToast("Auto-detection failed, proceeding with manual selection.", "warning");
      }
    }

    // JPN/CHN -> manga-ocr, KOR -> pororo, others -> paddle
    if (finalSrcLang === "jpn_Jpan" || finalSrcLang.includes("zho")) {
      finalOcrEngine = "manga-ocr";
    } else if (finalSrcLang === "kor_Hang") {
      finalOcrEngine = "pororo";
    } else if (finalSrcLang !== "") {
      finalOcrEngine = "paddle";
    }

    if (finalOcrEngine !== ocrEngine) {
      console.log(`[Auto-OCR] Switched ${ocrEngine} -> ${finalOcrEngine} for lang ${finalSrcLang}`);
    }

    const fd = new FormData();
    fd.append("cbz_file", file);
    fd.append("series", series);
    fd.append("source_lang", finalSrcLang);
    fd.append("target_lang", tgtLang);
    fd.append("translation_engine", engine);
    fd.append("ocr_engine", finalOcrEngine);
    fd.append("detection_engine", detEngine);
    fd.append("inpaint_engine", inpaintEngine);
    fd.append("use_mit_pipeline", useMit ? "true" : "false");
    fd.append("use_koharu_pipeline", useKoharu ? "true" : "false");
    fd.append("mit_translator", mitTranslator);
    fd.append("mit_target_lang", mitTargetLang);
    fd.append("force_retranslate", forceRetranslate ? "true" : "false");
    fd.append("chunk_height", chunkHeight.toString());
    fd.append("chunk_overlap", chunkOverlap.toString());
    fd.append("ocr_super_res", superResEnabled ? "true" : "false");
    fd.append("ocr_upscale_factor", upscaleFactor);
    fd.append("global_upscale_mode", globalUpscaleMode);

    try {
      const resp = await fetch("/upload", { method: "POST", body: fd });
      const data = await resp.json();
      if (data.ok) {
        let label = `Queued: ${file.name}`;
        if (useMit) label = `Queued (MIT): ${file.name}`;
        else if (useKoharu) label = `Queued (Koharu): ${file.name}`;
        showToast(label, "success");
      } else {
        showToast(`Upload error: ${data.error}`, "error");
      }
    } catch (err) {
      showToast(`Upload failed: ${err.message}`, "error");
    }
  }
}

// ── Review progress tracker ───────────────────────────────────────────────────
const reviewProgress = { approved: 0, edited: 0, rejected: 0, total: 0 };

function initReviewProgress() {
  const allCards = document.querySelectorAll(".bubble-card");
  reviewProgress.total = allCards.length;
  reviewProgress.approved = 0;
  reviewProgress.edited = 0;
  reviewProgress.rejected = 0;

  // ── Restore persisted state from server-rendered data attributes ──────────
  allCards.forEach(card => {
    const isIgnored = card.dataset.skipInpaint === "true";
    const isApproved = card.dataset.approved === "true";
    const isEdited = card.dataset.edited === "true";
    const isRejected = card.dataset.rejected === "true";

    // Restore ignore button visual state
    if (isIgnored) {
      card.dataset.ignored = "true";
      card.style.opacity = "0.5";
      const ignoreBtn = card.querySelector(".btn-ignore");
      if (ignoreBtn) {
        ignoreBtn.innerHTML = "↺ Undo Ignore";
        ignoreBtn.classList.remove("btn-warning");
        ignoreBtn.classList.add("btn-info");
        ignoreBtn.dataset.action = "undo_ignore";
      }
    }

    // Restore opacity for approved/edited/rejected cards - GOAL 2: Keep them bright
    if (!isIgnored && (isApproved || isEdited || isRejected)) {
      card.style.opacity = "1.0";
    }

    // Count into progress and set current state
    if (isIgnored) {
      reviewProgress.edited++;
      card.dataset.currentState = "ignore";
    } else if (isEdited) {
      reviewProgress.edited++;
      card.dataset.currentState = "edit";
    } else if (isApproved) {
      reviewProgress.approved++;
      card.dataset.currentState = "approve";
    } else if (isRejected) {
      reviewProgress.rejected++;
      card.dataset.currentState = "reject";
    } else {
      card.dataset.currentState = "unreviewed";
    }
  });

  updateReviewProgressUI();
}

function updateReviewProgressUI() {
  const pctEl = document.getElementById("rp-pct");
  const fillEl = document.getElementById("rp-fill");
  const approvedEl = document.getElementById("rp-approved");
  const editedEl = document.getElementById("rp-edited");
  const rejectedEl = document.getElementById("rp-rejected");
  const remainingEl = document.getElementById("rp-remaining");
  if (!pctEl || !fillEl) return;

  const done = reviewProgress.approved + reviewProgress.edited + reviewProgress.rejected;
  const remaining = Math.max(0, reviewProgress.total - done);
  const pct = reviewProgress.total > 0 ? Math.round((done / reviewProgress.total) * 100) : 0;

  pctEl.textContent = `${pct}%`;
  fillEl.style.width = `${pct}%`;

  if (approvedEl) approvedEl.textContent = `✅ ${reviewProgress.approved}`;
  if (editedEl) editedEl.textContent = `✏️ ${reviewProgress.edited}`;
  if (rejectedEl) rejectedEl.textContent = `❌ ${reviewProgress.rejected}`;
  if (remainingEl) remainingEl.textContent = `⏳ ${remaining}`;

  if (pct >= 100) {
    fillEl.classList.add("complete");
    pctEl.classList.add("complete");
  } else {
    fillEl.classList.remove("complete");
    pctEl.classList.remove("complete");
  }
}

// ── Bubble review ─────────────────────────────────────────────────────────────
function initBubbleReview() {
  document.querySelectorAll(".btn-approve").forEach(btn => {
    btn.addEventListener("click", () => bubbleAction("approve", btn));
  });
  document.querySelectorAll(".btn-reject").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (!confirm("Reject this translation? It will be saved as rejected.")) return;
      bubbleAction("reject", btn);
    });
  });
  document.querySelectorAll(".btn-edit-save").forEach(btn => {
    btn.addEventListener("click", () => bubbleAction("edit", btn));
  });
  document.querySelectorAll(".btn-ignore").forEach(btn => {
    if (!btn.dataset.action) btn.dataset.action = "ignore";
    btn.addEventListener("click", async () => {
      const currentAction = btn.dataset.action;
      bubbleAction(currentAction, btn);
    });
  });
  document.querySelectorAll(".btn-ignore-page").forEach(btn => {
    btn.addEventListener("click", async () => {
      const pageNum = btn.dataset.page;
      const cards = document.querySelectorAll(`.bubble-card[data-page="${pageNum}"]`);
      let count = 0;
      const promises = [];
      for (const card of cards) {
        if (card.dataset.ignored !== "true") {
          const ignoreBtn = card.querySelector(".btn-ignore");
          if (ignoreBtn) {
            promises.push(bubbleAction("ignore", ignoreBtn));
            count++;
          }
        }
      }
      await Promise.all(promises);
      if (count > 0) {
        showToast(`🙈 Ignored ${count} bubbles on Page ${pageNum}`, "success");
      }
    });
  });

  document.querySelectorAll(".btn-undo-ignore-page").forEach(btn => {
    btn.addEventListener("click", async () => {
      const pageNum = btn.dataset.page;
      const cards = document.querySelectorAll(`.bubble-card[data-page="${pageNum}"]`);
      let count = 0;
      for (const card of cards) {
        if (card.dataset.ignored === "true") {
          const ignoreBtn = card.querySelector(".btn-info"); // Undo button has btn-info class
          if (ignoreBtn && ignoreBtn.dataset.action === "undo_ignore") {
            await bubbleAction("undo_ignore", ignoreBtn);
            count++;
          }
        }
      }
      if (count > 0) {
        showToast(`↺ Restored ${count} bubbles on Page ${pageNum}`, "success");
      }
    });
  });
}

async function bubbleAction(action, btn) {
  const card = btn.closest(".bubble-card");
  const grid = card.closest(".bubble-grid");
  const cbzName = grid ? grid.dataset.cbzName : "";
  const memId = card.dataset.memoryId;
  const series = card.dataset.series;
  const srcText = card.dataset.sourceText;
  const srcLang = card.dataset.sourceLang;
  const textarea = card.querySelector("textarea");
  const transText = textarea ? textarea.value.trim() : card.dataset.translatedText;

  const isIgnore = action === "ignore";
  const isUndoIgnore = action === "undo_ignore";
  const endpoint = { approve: "/approve", reject: "/reject", edit: "/edit", ignore: "/edit", undo_ignore: "/edit" }[action];
  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        memory_id: memId ? parseInt(memId) : null,
        series, source_text: srcText,
        translated_text: isIgnore ? "" : transText, source_lang: srcLang,
        cbz_name: cbzName,
        skip_inpaint: isIgnore,
        bubble_index: parseInt(card.dataset.bubbleNum) - 1
      }),
    });
    const data = await resp.json();
    if (data.ok) {
      if (typeof reviewProgress !== "undefined") {
        const oldState = card.dataset.currentState;
        if (oldState === "approve") reviewProgress.approved = Math.max(0, reviewProgress.approved - 1);
        else if (oldState === "edit") reviewProgress.edited = Math.max(0, reviewProgress.edited - 1);
        else if (oldState === "reject") reviewProgress.rejected = Math.max(0, reviewProgress.rejected - 1);
        else if (oldState === "ignore") reviewProgress.edited = Math.max(0, reviewProgress.edited - 1);

        if (action === "approve") reviewProgress.approved++;
        else if (action === "edit") reviewProgress.edited++;
        else if (action === "reject") reviewProgress.rejected++;
        else if (action === "ignore") reviewProgress.edited++;

        card.dataset.currentState = (action === "undo_ignore") ? "unreviewed" : action;
      }

      if (isIgnore) {
        card.dataset.ignored = "true";
        card.style.opacity = "0.5";
        btn.innerHTML = "↺ Undo Ignore";
        btn.classList.remove("btn-warning");
        btn.classList.add("btn-info");
        btn.dataset.action = "undo_ignore";
      } else if (isUndoIgnore) {
        card.dataset.ignored = "false";
        card.style.opacity = "1";
        btn.innerHTML = "🙈 Ignore (Keep Original)";
        btn.classList.remove("btn-info");
        btn.classList.add("btn-warning");
        btn.dataset.action = "ignore";
      } else {
        card.style.opacity = "1.0"; // Approved/Edited/Rejected remain fully bright

        // Reset Ignore button if it was in Undo state
        const ignoreBtn = card.querySelector(".btn-info");
        if (ignoreBtn && ignoreBtn.dataset.action === "undo_ignore") {
          ignoreBtn.innerHTML = "🙈 Ignore (Keep Original)";
          ignoreBtn.classList.remove("btn-info");
          ignoreBtn.classList.add("btn-warning");
          ignoreBtn.dataset.action = "ignore";
          card.dataset.ignored = "false";
        }
      }
      updateReviewProgressUI();
    } else {
      showToast(`Error: ${data.error}`, "error");
    }
  } catch (err) {
    showToast(`Request failed: ${err.message}`, "error");
  }
}

// ── Memory inline edit ────────────────────────────────────────────────────────
function initMemoryEdit() {
  document.querySelectorAll(".mem-edit-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const row = btn.closest("tr");
      const cell = row.querySelector(".translated-text");
      const cur = cell.textContent.trim();
      cell.innerHTML = `<input type="text" class="mem-inline-input" value="${escAttr(cur)}" style="width:100%">`;
      btn.textContent = "Save";
      btn.onclick = async () => {
        const newVal = row.querySelector(".mem-inline-input").value.trim();
        const rowId = btn.dataset.id;
        const rowSeries = btn.dataset.series || null;
        try {
          const resp = await fetch("/memory/edit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: rowId, translated_text: newVal, series: rowSeries }),
          });
          const data = await resp.json();
          if (data.ok) {
            cell.textContent = newVal;
            btn.textContent = "Edit";
            btn.onclick = null;
            initMemoryEdit();
            showToast("Memory updated.", "success");
          } else {
            showToast(`Error: ${data.error}`, "error");
          }
        } catch (err) {
          showToast(err.message, "error");
        }
      };
    });
  });
}

// ── Memory search filter ──────────────────────────────────────────────────────
function initMemorySearch() {
  const input = document.getElementById("memory-search");
  if (!input) return;
  input.addEventListener("input", () => {
    const q = input.value.toLowerCase();
    document.querySelectorAll("#memory-table tbody tr").forEach(row => {
      row.style.display = row.textContent.toLowerCase().includes(q) ? "" : "none";
    });
  });
}

// ── Training start ────────────────────────────────────────────────────────────
function initTrainButton() {
  const btn = document.getElementById("train-now-btn");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    if (!confirm("Start fine-tuning now? This may take several minutes.")) return;
    btn.disabled = true;
    btn.textContent = "⏳ Training in background...";
    try {
      const resp = await fetch("/train/start", { method: "POST" });
      const data = await resp.json();
      if (data.ok) {
        showToast("🧠 Training started! You'll get a toast when done.", "success");
      } else {
        showToast(`Cannot train: ${data.error}`, "error");
        btn.disabled = false;
        btn.textContent = "Train Now";
      }
    } catch (err) {
      showToast(err.message, "error");
      btn.disabled = false;
      btn.textContent = "⚡ Train Now";
    }
  });
}

function initTrainProgressSSE() {
  const container = document.getElementById("train-progress-container");
  if (!container) return; // Only run on /train page

  const btn = document.getElementById("train-now-btn");
  const idleText = document.getElementById("train-idle-text");
  const fill = document.getElementById("train-progress-fill");
  const text = document.getElementById("train-progress-text");
  const etaText = document.getElementById("train-progress-eta");

  const formatTime = (seconds) => {
    if (!seconds || seconds <= 0) return "--";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };

  const evts = new EventSource("/train_progress_stream");

  evts.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === "ping") return;

    if (data.status === "training") {
      container.classList.remove("hidden");
      btn.style.display = "none";
      idleText.style.display = "none";

      const pct = data.total_steps > 0 ? Math.min(100, Math.round((data.step / data.total_steps) * 100)) : 0;
      fill.style.width = `${pct}%`;
      text.textContent = `Step ${data.step} / ${data.total_steps}`;
      etaText.textContent = `ETA: ${formatTime(data.eta_s)}`;
    } else if (data.status === "completed") {
      container.classList.add("hidden");
      btn.style.display = "block";
      idleText.style.display = "block";
      btn.textContent = "⚡ Train Now";
      btn.disabled = false;
      location.reload(); // Reload to show new version in history
    } else if (data.status === "error") {
      container.classList.add("hidden");
      btn.style.display = "block";
      idleText.style.display = "block";
      btn.textContent = "⚡ Train Now";
      btn.disabled = false;
      showToast(`Training error: ${data.error}`, "error");
    } else {
      // Idle
      container.classList.add("hidden");
      btn.style.display = "block";
      idleText.style.display = "block";
    }
  };
}

// ── Backup actions ────────────────────────────────────────────────────────────
function initBackupActions() {
  // Manual backup
  const backupBtn = document.getElementById("backup-now-btn");
  if (backupBtn) {
    backupBtn.addEventListener("click", async () => {
      backupBtn.disabled = true;
      try {
        const resp = await fetch("/backup/now", { method: "POST" });
        const data = await resp.json();
        if (data.ok) { showToast("💾 Backup created!", "success"); location.reload(); }
        else showToast(`Error: ${data.error}`, "error");
      } catch (err) { showToast(err.message, "error"); }
      backupBtn.disabled = false;
    });
  }

  // Restore buttons
  document.querySelectorAll(".restore-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const ts = btn.dataset.timestamp;
      if (!confirm(`Restore backup from ${ts}? This will overwrite your current memory and model.`)) return;
      btn.disabled = true;
      btn.textContent = "Restoring...";
      try {
        const resp = await fetch("/backup/restore", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ timestamp: ts }),
        });
        const data = await resp.json();
        if (data.ok) { showToast("✅ Restore complete!", "success"); }
        else showToast(`Error: ${data.error}`, "error");
      } catch (err) { showToast(err.message, "error"); }
      btn.disabled = false;
      btn.textContent = "Restore";
    });
  });
}

// ── Font upload (settings) ────────────────────────────────────────────────────
function initFontUpload() {
  const btn = document.getElementById("font-upload-btn");
  const inp = document.getElementById("font-file-input");
  if (!btn || !inp) return;
  btn.addEventListener("click", () => inp.click());
  inp.addEventListener("change", async () => {
    if (!inp.files.length) return;
    const fd = new FormData();
    fd.append("font_file", inp.files[0]);
    try {
      const resp = await fetch("/settings/upload_font", { method: "POST", body: fd });
      const data = await resp.json();
      if (data.ok) showToast(`Font uploaded: ${data.path}`, "success");
      else showToast(`Error: ${data.error}`, "error");
    } catch (err) { showToast(err.message, "error"); }
  });
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function escAttr(s) {
  return String(s).replace(/"/g, "&quot;");
}

// ── Re-render CBZ ─────────────────────────────────────────────────────────────
function initRerender() {
  const btn = document.getElementById("rerender-btn");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    if (!confirm("Re-render the CBZ with your reviewed/edited translations? This will replace the output file.")) return;
    btn.disabled = true;
    btn.textContent = "⏳ Re-rendering...";
    try {
      const cbzName = btn.dataset.cbz;
      const resp = await fetch(`/rerender/${encodeURIComponent(cbzName)}`, { method: "POST" });
      const data = await resp.json();
      if (data.ok) {
        showToast("🔄 Re-render started! The output CBZ will be rebuilt in the background.", "success");
      } else {
        showToast(`Error: ${data.error}`, "error");
      }
    } catch (err) {
      showToast(`Request failed: ${err.message}`, "error");
    }
    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = "🔄 Re-render CBZ";
    }, 5000);
  });
}

// ── Re-render from completed card ─────────────────────────────────────────────
async function rerenderFromCard(btn) {
  const cbzName = btn.dataset.cbz;
  if (!confirm(`Re-render "${cbzName}" with your reviewed/edited translations?`)) return;
  btn.disabled = true;
  btn.textContent = "⏳ Re-rendering...";
  try {
    const resp = await fetch(`/rerender/${encodeURIComponent(cbzName)}`, { method: "POST" });
    const data = await resp.json();
    if (data.ok) {
      showToast("🔄 Re-render started! The output CBZ will be rebuilt in the background.", "success");
    } else {
      showToast(`Error: ${data.error}`, "error");
    }
  } catch (err) {
    showToast(`Request failed: ${err.message}`, "error");
  }
  setTimeout(() => {
    btn.disabled = false;
    btn.textContent = "🔄 Re-render";
  }, 5000);
}

// ── MIT Pipeline Toggle ───────────────────────────────────────────────────────
function initMITToggle() {
  const engineSel = document.getElementById("engine-select");
  const mitOpts = document.getElementById("mit-pipeline-opts");

  function applyToggle() {
    if (!engineSel || !mitOpts) return;

    // Attempt to hide OCR, Inpaint, SourceLang
    const ocrSel = document.getElementById("ocr-engine-select");
    const inpSel = document.getElementById("inpaint-engine-select");
    const srcSel = document.getElementById("source-lang-select");
    const tgtSel = document.getElementById("target-lang-select");

    if (engineSel.value === "mit") {
      if (ocrSel) ocrSel.closest(".form-group").style.display = "none";
      if (inpSel) inpSel.closest(".form-group").style.display = "none";
      if (srcSel) srcSel.closest(".form-group").style.display = "none";
      if (tgtSel) tgtSel.closest(".form-group").style.display = "none";
      mitOpts.style.display = "flex";
    } else if (engineSel.value === "koharu") {
      if (ocrSel) ocrSel.closest(".form-group").style.display = "none";
      if (inpSel) inpSel.closest(".form-group").style.display = "none";
      if (srcSel) srcSel.closest(".form-group").style.display = "none";
      if (tgtSel) tgtSel.closest(".form-group").style.display = "none";
      mitOpts.style.display = "flex";
    } else {
      if (ocrSel) ocrSel.closest(".form-group").style.display = "block";
      if (inpSel) inpSel.closest(".form-group").style.display = "block";
      if (srcSel) srcSel.closest(".form-group").style.display = "block";
      if (tgtSel) tgtSel.closest(".form-group").style.display = "block";
      mitOpts.style.display = "none";
    }
  }

  if (engineSel) {
    engineSel.addEventListener("change", applyToggle);
    applyToggle(); // apply initial state on page load
  }
}

// ── Delete Output ─────────────────────────────────────────────────────────────
function initDeleteOutput() {
  document.querySelectorAll(".delete-output-btn").forEach(btn => {
    // Remove existing listeners to avoid duplicates
    const newBtn = btn.cloneNode(true);
    btn.parentNode.replaceChild(newBtn, btn);

    newBtn.addEventListener("click", async () => {
      const cbzName = newBtn.dataset.cbz;
      if (!confirm(`Delete output & session data for "${cbzName}"?\n\nThis will remove the translated CBZ, review data, and crop images so you can re-translate this file.`)) return;

      newBtn.disabled = true;
      newBtn.textContent = "Deleting...";

      try {
        const resp = await fetch(`/delete_output/${encodeURIComponent(cbzName)}`, { method: "POST" });
        const data = await resp.json();
        if (data.ok) {
          showToast(`🗑️ Deleted output for ${cbzName}`, "success");
          // Completely remove the card from UI
          const card = newBtn.closest(".card");
          if (card) {
            card.remove();
          }
        } else {
          showToast(`Error: ${data.error}`, "error");
          newBtn.disabled = false;
          newBtn.textContent = "🗑️ Delete";
        }
      } catch (err) {
        showToast(`Failed: ${err.message}`, "error");
        newBtn.disabled = false;
        newBtn.textContent = "🗑️ Delete";
      }
    });
  });
}

// ── Bulk LLM Translation (Manual) ─────────────────────────────────────────────
function initBulkLLMTranslator() {
  const btnCopy = document.getElementById("btn-copy-prompt");
  const btnApply = document.getElementById("btn-apply-bulk");
  const btnCancel = document.getElementById("btn-cancel-bulk");
  const txtResult = document.getElementById("bulk-llm-result");
  if (!btnCopy || !btnApply || !txtResult) return;

  let isCancelled = false;
  if (btnCancel) {
    btnCancel.addEventListener("click", () => {
      isCancelled = true;
      btnCancel.disabled = true;
      btnCancel.textContent = "Cancelling...";
    });
  }

  btnCopy.addEventListener("click", () => {
    const allCards = document.querySelectorAll(".bubble-card");
    const activeCards = Array.from(allCards).filter(c => c.dataset.ignored !== "true");

    if (activeCards.length === 0) {
      showToast("No active bubbles to translate.", "warning");
      return;
    }

    let prompt = "Translate the following manga text blocks to English. Keep the exact numbering and line count output as a numbered list:\n\n";
    activeCards.forEach((card) => {
      const num = card.dataset.bubbleNum;
      const srcText = card.querySelector(".bubble-source-text").innerText.replace(/\n /g, "").trim();
      prompt += `${num}. ${srcText}\n`;
    });

    navigator.clipboard.writeText(prompt).then(() => {
      showToast("📋 Prompt copied! Paste it into ChatGPT.", "success");
    }).catch(err => {
      showToast(`Failed to copy: ${err}`, "error");
    });
  });

  btnApply.addEventListener("click", async () => {
    const rawText = txtResult.value;
    const lines = rawText.split('\n');
    if (lines.length === 0) {
      showToast("No text pasted to apply.", "warning");
      return;
    }

    const allCards = document.querySelectorAll(".bubble-card");
    const activeCards = Array.from(allCards).filter(c => c.dataset.ignored !== "true");

    // 1. Build a map of all translations found in the pasted text
    const translationMap = new Map();
    lines.forEach(line => {
      // Regex: Matches "21. text", "21) text", "21: text", or "21 text"
      // Using a non-greedy delimiter to ensure translations starting with dots (e.g. "...") aren't eaten.
      const match = line.match(/^\s*(\d+)(?:[\.\)\:]\s*|\s+)(.*)/);
      if (match) {
        const num = match[1];
        const text = match[2].trim();
        translationMap.set(num, text);
      }
    });

    let matchedCount = 0;
    let matches = [];

    // 2. Map the translations to the active cards
    activeCards.forEach((card) => {
      const num = card.dataset.bubbleNum;
      const translated = translationMap.get(num);

      if (translated !== undefined) {
        matchedCount++;
        const textarea = card.querySelector("textarea");
        if (textarea) {
          textarea.value = translated;
          matches.push({ card, translated });
        }
      }
    });

    if (matches.length > 0) {
      isCancelled = false;
      btnApply.style.display = "none";
      if (btnCancel) {
        btnCancel.style.display = "flex";
        btnCancel.disabled = false;
        btnCancel.textContent = "🛑 Cancel";
      }

      const pWrap = document.getElementById("bulk-progress-wrap");
      const pFill = document.getElementById("bulk-progress-fill");
      const pText = document.getElementById("bulk-progress-text");

      if (pWrap && pFill && pText) {
        pWrap.style.display = "block";
        pText.style.display = "block";
        pFill.style.width = "0%";
        pText.textContent = `Saving: 0 / ${matches.length}`;
      }

      let successCount = 0;

      const payload = matches.map(m => {
        const grid = m.card.closest(".bubble-grid");
        return {
          memory_id: m.card.dataset.memoryId ? parseInt(m.card.dataset.memoryId) : null,
          series: m.card.dataset.series,
          source_text: m.card.dataset.sourceText,
          translated_text: m.translated,
          source_lang: m.card.dataset.sourceLang,
          cbz_name: grid ? grid.dataset.cbzName : ""
        };
      });

      if (!isCancelled) {
        try {
          const resp = await fetch("/edit_bulk", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await resp.json();
          if (data.ok) {
            successCount = matches.length;
            // Update UI states and review progress
            matches.forEach(m => {
              const oldState = m.card.dataset.currentState;
              // Only update counts if the state actually changes to 'edit'
              if (oldState !== "edit" && oldState !== "ignore") {
                if (oldState === "approve") reviewProgress.approved = Math.max(0, reviewProgress.approved - 1);
                else if (oldState === "reject") reviewProgress.rejected = Math.max(0, reviewProgress.rejected - 1);

                reviewProgress.edited++;
                m.card.dataset.currentState = "edit";
                m.card.style.opacity = "1.0"; // Ensure bright after auto-fill
              }
            });
            updateReviewProgressUI();
          } else {
            showToast(`Error: ${data.error}`, "error");
          }
        } catch (err) {
          console.error(err);
          showToast(`Batch save failed: ${err.message}`, "error");
        }
      }

      btnApply.disabled = false;
      btnApply.style.display = "flex";
      if (btnCancel) btnCancel.style.display = "none";

      if (isCancelled) {
        showToast(`⚠️ Batch cancelled. Saved ${successCount} out of ${matches.length} items.`, "warning");
      } else {
        showToast(`⚡ Batch saved ${successCount} items automatically!`, "success");
      }

      // Hide bar after success
      if (pWrap && pText) {
        pFill.style.width = `100%`;
        pText.textContent = `Saving: ${successCount} / ${matches.length}`;
        setTimeout(() => {
          pWrap.style.display = "none";
          pText.style.display = "none";
          pFill.style.width = "0%";
        }, 5000);
      }

    } else {
      showToast("Could not find any matching bubble numbers (e.g., '1.') in the pasted text.", "error");
    }
  });
}

// ── Searchable Select ─────────────────────────────────────────────────────────
function initSearchableSelects() {
  const selects = document.querySelectorAll("#source-lang-select");
  selects.forEach(select => {
    if (select.dataset.searchableInit) return;
    select.dataset.searchableInit = "true";

    const wrapper = document.createElement("div");
    wrapper.className = "searchable-select-wrapper";

    const input = document.createElement("input");
    input.type = "text";
    input.className = "searchable-select-input";
    input.placeholder = "Search language...";

    // Set initial text
    const updateInputText = () => {
      const selectedOpt = select.options[select.selectedIndex];
      input.value = selectedOpt ? selectedOpt.text : "";
    };
    updateInputText();

    const chevron = document.createElement("div");
    chevron.className = "searchable-select-chevron";
    chevron.innerHTML = "▼";

    const dropdown = document.createElement("div");
    dropdown.className = "searchable-select-dropdown";

    select.parentNode.insertBefore(wrapper, select);
    wrapper.appendChild(input);
    wrapper.appendChild(chevron);
    wrapper.appendChild(dropdown);

    // Original select is moved inside wrapper but hidden
    wrapper.appendChild(select);
    select.style.display = "none";

    // Build options
    const items = [];
    Array.from(select.options).forEach(opt => {
      const item = document.createElement("div");
      item.className = "searchable-select-item";
      item.textContent = opt.text;
      item.dataset.value = opt.value;
      if (opt.selected) item.classList.add("selected");

      item.addEventListener("click", () => {
        select.value = opt.value;
        updateInputText();
        items.forEach(i => i.classList.remove("selected"));
        item.classList.add("selected");
        dropdown.style.display = "none";
        select.dispatchEvent(new Event("change", { bubbles: true }));
      });
      dropdown.appendChild(item);
      items.push(item);
    });

    input.addEventListener("focus", () => {
      input.value = ""; // clear to let user search easily
      dropdown.style.display = "block";
      items.forEach(item => item.classList.remove("hidden"));
    });

    input.addEventListener("input", () => {
      const filter = input.value.toLowerCase();
      items.forEach(item => {
        if (item.textContent.toLowerCase().includes(filter)) {
          item.classList.remove("hidden");
        } else {
          item.classList.add("hidden");
        }
      });
    });

    document.addEventListener("click", (e) => {
      if (!wrapper.contains(e.target)) {
        dropdown.style.display = "none";
        updateInputText(); // restore text if they clicked away without selecting
      }
    });

    // Update if the original select is changed programmatically
    select.addEventListener("change", () => {
      updateInputText();
      items.forEach(i => {
        i.classList.toggle("selected", i.dataset.value === select.value);
      });
    });
  });
}


// ── Delete Series ─────────────────────────────────────────────────────────────
async function deleteSeries(seriesName) {
  if (!confirm(`Are you sure you want to completely delete the memory for series "${seriesName}"?`)) {
    return;
  }

  try {
    const res = await fetch("/memory/series/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ series: seriesName })
    });

    const data = await res.json();
    if (data.ok) {
      showToast(`Series "${seriesName}" deleted successfully!`);
      setTimeout(() => location.href = "/memory?series=series", 1000);
    } else {
      alert("Error: " + data.error);
    }
  } catch (err) {
    alert("Network error: " + err.message);
  }
}

// ── Delete All Models ─────────────────────────────────────────────────────────
const deleteModelsBtn = document.getElementById("delete-models-btn");
if (deleteModelsBtn) {
  deleteModelsBtn.addEventListener("click", async () => {
    const confirmed = confirm(
      "⚠️ Delete ALL downloaded models?\n\n" +
      "This will permanently remove:\n" +
      "  • ./models/  (detection, OCR, inpainting, translators)\n" +
      "  • ./Pipeline Koharu/  (Koharu pipeline models)\n\n" +
      "Models will need to be re-downloaded before the app can work again.\n\n" +
      "Are you sure?"
    );
    if (!confirmed) return;

    const statusEl = document.getElementById("delete-models-status");
    deleteModelsBtn.disabled = true;
    deleteModelsBtn.textContent = "⏳ Deleting...";
    if (statusEl) { statusEl.style.display = "block"; statusEl.textContent = "Deleting model files..."; statusEl.style.color = "var(--text-muted)"; }

    try {
      const res = await fetch("/models/clear", { method: "POST" });
      const data = await res.json();

      if (data.success) {
        showToast("✅ " + data.message, "success");
        if (statusEl) { statusEl.textContent = "✅ " + data.message; statusEl.style.color = "var(--accent-green)"; }
      } else {
        showToast("❌ " + data.message, "error");
        if (statusEl) { statusEl.textContent = "❌ " + data.message; statusEl.style.color = "var(--accent-red)"; }
      }
    } catch (err) {
      showToast("❌ Network error: " + err.message, "error");
      if (statusEl) { statusEl.textContent = "❌ Network error: " + err.message; statusEl.style.color = "var(--accent-red)"; }
    } finally {
      deleteModelsBtn.disabled = false;
      deleteModelsBtn.textContent = "🗑️ Delete All Downloaded Models";
    }
  });
}

// ── Autodetect (Language Detection) ──────────────────────────────────────────
const DETECT_API_BASE = "http://localhost:8000"; // FastAPI port

function initAutodetect() {
  const engineSel = document.getElementById("autodetect_engine_select");
  if (!engineSel) return;

  const configArea = document.getElementById("engine_config_area");
  const keyInput = document.getElementById("autodetect_api_key");
  const testBtn = document.getElementById("test_engine_btn");
  const saveBtn = document.getElementById("save_autodetect_btn");
  const testResult = document.getElementById("test_result");

  // Load engines
  fetch(`${DETECT_API_BASE}/autodetect/engines`)
    .then(res => res.json())
    .then(engines => {
      engineSel.innerHTML = '<option value="">-- Select Engine --</option>';
      engines.forEach(eng => {
        const opt = document.createElement("option");
        opt.value = eng.name;
        let badge = eng.is_free ? " [FREE]" : (eng.requires_api_key ? " [API KEY]" : " [LOCAL]");
        opt.textContent = eng.name.toUpperCase() + badge;
        if (eng.is_active) opt.selected = true;
        engineSel.appendChild(opt);
      });
      updateEngineUI();
    })
    .catch(err => console.error("Failed to load autodetect engines", err));

  function updateEngineUI() {
    const selected = engineSel.value;
    if (!selected) {
      configArea.style.display = "none";
      return;
    }

    // Check if engine requires API key
    const opt = engineSel.options[engineSel.selectedIndex];
    const text = opt.textContent;
    if (text.includes("[API KEY]")) {
      configArea.style.display = "block";
      // Load from localStorage
      keyInput.value = localStorage.getItem(`autodetect_key_${selected}`) || "";
    } else {
      configArea.style.display = "none";
    }
  }

  if (engineSel) engineSel.addEventListener("change", updateEngineUI);

  if (testBtn) testBtn.addEventListener("click", async () => {
    const engine = engineSel.value;
    const key = keyInput.value;
    testResult.textContent = "⏳ Testing...";
    testResult.className = "mt-4 text-xs text-muted";

    try {
      const res = await fetch(`${DETECT_API_BASE}/autodetect/test/${engine}`, {
        headers: { "X-API-Key": key }
      });
      const data = await res.json();
      if (data.status === "success") {
        testResult.textContent = "✅ Connection successful!";
        testResult.className = "mt-4 text-xs text-success";
      } else {
        testResult.textContent = "❌ Connection failed. Check your API key.";
        testResult.className = "mt-4 text-xs text-danger";
      }
    } catch (err) {
      testResult.textContent = "❌ API unreachable. Make sure api.py is running.";
      testResult.className = "mt-4 text-xs text-danger";
    }
  });

  if (saveBtn) saveBtn.addEventListener("click", async () => {
    const engine = engineSel.value;
    const key = keyInput.value;

    if (!engine) {
      showToast("Please select an engine first.", "warning");
      return;
    }

    try {
      const res = await fetch(`${DETECT_API_BASE}/autodetect/set-engine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine })
      });
      const data = await res.json();
      if (data.status === "success") {
        localStorage.setItem("active_autodetect_engine", engine);
        if (key) {
          localStorage.setItem(`autodetect_key_${engine}`, key);
        }
        showToast(`✅ Language detection engine set to ${engine}`, "success");
      } else {
        showToast(`❌ Error: ${data.detail}`, "error");
      }
    } catch (err) {
      showToast("❌ Failed to save settings. Is the API running?", "error");
    }
  });
}

// ── DOM Content Loaded ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initQueueSSE();
  initDropZone();
  initMITToggle();
  initDeleteOutput();
  initReviewProgress();
  initBubbleReview();
  initMemoryEdit();
  initMemorySearch();
  initTrainButton();
  initTrainProgressSSE();
  initBackupActions();
  initFontUpload();
  initRerender();
  initBulkLLMTranslator();
  initSearchableSelects();
  initAutodetect();
});
