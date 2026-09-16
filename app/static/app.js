const App = (() => {
  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x) => x == null ? "–" : Math.round(x * 100) + "%";
  const money = (x) => x == null ? "–" : "$" + Number(x).toFixed(0);

  async function api(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return r.json();
  }

  // ── rendering ──

  function renderStages(stages) {
    const icons = { pending: "○", running: "◐", done: "✓", skipped: "–", failed: "✗" };
    $("#stages").innerHTML = stages.map((s) =>
      `<li class="${s.status}"><span class="ico">${icons[s.status] || "○"}</span><span>${esc(s.label)}</span>` +
      `<span class="detail">${esc(s.detail || "")}${s.ms != null ? ` <span class="muted">(${s.ms} ms)</span>` : ""}</span></li>`).join("");
  }

  function evidenceChecklist(r) {
    const m = new Set(r.identification_method || []);
    const items = [
      ["Image similarity", m.has("image_similarity")], ["Product metadata", m.has("product_metadata")],
      ["Computer vision", m.has("computer_vision")], ["Web search", m.has("web_search")],
      ["StockX catalog", m.has("stockx_catalog") || m.has("stockx_match")], ["Visual verification", m.has("visual_verification")],
      ["StockX match", m.has("stockx_match")], ["Human review", m.has("manual_review")],
    ];
    return items.map(([l, ok]) => `<div class="${ok ? "ok" : "muted"}">${ok ? "✓" : "○"} ${l}</div>`).join("");
  }

  function renderResult(r, target) {
    const box = target || $("#result");
    box.hidden = false;
    const p = r.product, sx = r.stockx;
    const img = (p && p.images && p.images[0]) || (r.input_images && r.input_images[0]) || "";
    const fails = (r.failure_codes || []).map((f) => `<span class="chip bad">${esc(f)}</span>`).join("");
    const headline = r.status === "high" ? "MATCH FOUND" : r.status === "medium" ? "PROBABLE MATCH — verify" :
      r.status === "low" ? "LOW CONFIDENCE — needs review" : r.status === "error" ? "ERROR" : r.status === "unresolved" ? "UNRESOLVED — manual review" : r.status.toUpperCase();
    box.innerHTML = `
      <h2>Result <span class="status ${r.status}">${esc(r.status)}</span></h2>
      <div class="result-hero">
        <div>${img ? `<img src="${esc(img)}" alt="">` : ""}</div>
        <div>
          <div class="muted">${esc(headline)}</div>
          <h3>${esc(p ? (p.product_name || [p.brand, p.model].join(" ")) : (r.input_title || "No product identified"))}</h3>
          ${p ? `<div>${esc(p.brand)} · ${esc(p.model)} ${p.sub_model ? "· " + esc(p.sub_model) : ""}</div>` : ""}
          <dl class="kv">
            ${p ? `<dt>Colorway</dt><dd>${esc(p.colorway || "–")}</dd><dt>Style code</dt><dd class="big">${esc(p.style_code || "–")}</dd>` : ""}
            ${p && (p.gender || p.size_category) ? `<dt>Gender / sizing</dt><dd>${esc(p.gender || "?")} / ${esc(p.size_category || "?")}</dd>` : ""}
            ${p && p.release_date ? `<dt>Release</dt><dd>${esc(p.release_date)}</dd>` : ""}
            <dt>Confidence</dt><dd class="big">${pct(r.confidence)}</dd>
            ${sx ? `<dt>StockX</dt><dd><a href="${esc(sx.url)}" target="_blank" rel="noopener">${esc(sx.product_name)}</a> <span class="muted">(${esc(sx.style_code)})</span></dd>
                     <dt>Lowest ask</dt><dd class="big">${money(sx.lowest_ask)}</dd><dt>Highest bid</dt><dd class="big">${money(sx.highest_bid)}</dd>
                     ${sx.market_error ? `<dt></dt><dd class="bad">market: ${esc(sx.market_error)}</dd>` : ""}` : `<dt>StockX</dt><dd class="muted">no verified StockX product</dd>`}
            ${r.error ? `<dt>Error</dt><dd class="error">${esc(r.error)}</dd>` : ""}
          </dl>
          <div style="margin-top:10px" class="chips">${fails}</div>
          <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:13.5px">${evidenceChecklist(r)}</div>
          ${r.review_required ? `<p><a href="/review#${r.id}">Open in review queue →</a></p>` : ""}
        </div>
      </div>`;
  }

  function renderCandidates(r, target, onSelect) {
    const box = target || $("#candidates");
    const cands = r.candidates || [];
    box.hidden = cands.length === 0;
    if (!cands.length) return;
    const comp = (c) => Object.entries(c.components || {}).map(([k, v]) => `<div title="${esc(c.component_notes?.[k] || "")}">${esc(k)}: ${Math.round(v * 100)}</div>`).join("");
    box.innerHTML = `<h2>${onSelect ? "Possible matches" : "Candidates considered"}</h2>
      <table class="cands"><thead><tr><th></th><th>Style code</th><th>Product</th><th>Score</th><th>Signals</th><th>Verifier</th><th>Contradictions</th>${onSelect ? "<th></th>" : ""}</tr></thead><tbody>
      ${cands.map((c) => `<tr class="${c.rejected ? "rejected" : ""}">
        <td>${c.image_url ? `<img src="${esc(c.image_url)}" alt="">` : ""}</td>
        <td><b>${esc(c.style_code_display || c.style_code)}</b>${c.stockx_url ? `<br><a href="${esc(c.stockx_url)}" target="_blank" rel="noopener">StockX</a>` : ""}</td>
        <td>${esc(c.name || [c.brand, c.model].join(" "))}<br><small class="muted">${esc(c.colorway || "")} ${c.gender ? "· " + esc(c.gender) : ""} ${c.size_category ? "· " + esc(c.size_category) : ""}</small></td>
        <td><div class="bar"><i style="width:${Math.round(c.score * 100)}%"></i></div>${pct(c.score)}<div class="muted" style="font-size:11px">${comp(c)}</div></td>
        <td><small>${(c.sources || []).map((s) => esc(s.kind)).filter((v, i, a) => a.indexOf(v) === i).join(", ")}${c.embedding_similarity != null ? `<br>cos ${c.embedding_similarity.toFixed(3)}` : ""}</small></td>
        <td>${c.verification ? `<span class="${c.verification.verdict === "same" ? "ok" : c.verification.verdict === "different" ? "bad" : "muted"}">${esc(c.verification.verdict)}</span><br><small class="muted">${esc((c.verification.reasoning || "").slice(0, 140))}</small>` : "<span class='muted'>–</span>"}</td>
        <td><small class="${c.rejected ? "bad" : ""}">${(c.contradictions || []).map(esc).join("<br>") || "–"}</small></td>
        ${onSelect ? `<td><button data-code="${esc(c.style_code_display || c.style_code)}">Select</button></td>` : ""}
      </tr>`).join("")}</tbody></table>`;
    if (onSelect) box.querySelectorAll("button[data-code]").forEach((b) => b.onclick = () => onSelect(b.dataset.code));
  }

  function renderEvidence(r, target) {
    const box = target || $("#evidence");
    const e = r.evidence || {};
    box.hidden = false;
    const det = e.detected || {}, vis = e.vision || {}, page = e.page || {};
    const boxes = [
      ["Detected attributes", `brand: ${esc(det.brand || "–")}<br>model: ${esc(det.model || "–")}<br>colorways: ${esc((det.colorways || []).join(" | ") || "–")}<br>gender/size: ${esc(det.gender || "?")}/${esc(det.size_category || "?")}<br>page codes: ${esc((det.labeled_page_codes || []).join(", ") || "–")}<br>tag code: ${esc(det.tag_code || "–")}`],
      ["Vision analysis", vis.brand ? `${esc(vis.brand)} ${esc(vis.model)} — ${esc(vis.colorway)} ${vis.official_colorway_guess ? "(" + esc(vis.official_colorway_guess) + ")" : ""}<br>conf: brand ${pct(vis.brand_confidence)}, model ${pct(vis.model_confidence)}, colorway ${pct(vis.colorway_confidence)}<br>likely: ${esc((vis.likely_release_names || []).slice(0, 3).join(" | "))}<br>${esc((vis.distinctive_features || []).slice(0, 5).join("; "))}` : "<span class='muted'>not run</span>"],
      ["Product page", page.url ? `${esc(page.domain)} · ${esc(page.title || "")}<br>price ${esc(page.price ?? "–")} ${esc(page.currency || "")} · sources: ${esc((page.sources || []).join(", "))}<br>identifiers: ${esc(JSON.stringify(page.identifiers || {}))}` : "<span class='muted'>no URL</span>"],
      ["Web discovery", (e.web_candidates || []).slice(0, 5).map((w) => `${esc(w.style_code)} — ${w.agreement} domain(s): ${esc(w.domains.join(", "))}`).join("<br>") || "<span class='muted'>none</span>"],
      ["Visual neighbours", (e.visual_candidates || []).slice(0, 5).map((v) => `${esc(v.style_code)} ${v.similarity} ${esc(v.name || "")}`).join("<br>") || "<span class='muted'>none</span>"],
      ["StockX catalog hits", (e.stockx_candidates || []).slice(0, 5).map((s) => `${esc(s.style_id)} — ${esc(s.title)} (${esc(s.colorway)})`).join("<br>") || "<span class='muted'>none</span>"],
      ["Resolution", e.resolution ? `${esc(e.resolution.selected_style_code || "–")} → ${esc(e.resolution.status)} @ ${pct(e.resolution.confidence)}<br>StockX confirmed: ${e.resolution.stockx_confirmed}<br>${esc((e.resolution.notes || []).join(" · "))}` : "–"],
      ["Errors", Object.entries(e.errors || {}).map(([k, v]) => `<b>${esc(k)}</b>${v.map(esc).join("<br>")}`).join("<br>") || "<span class='muted'>none</span>"],
    ];
    box.innerHTML = `<h2>Evidence</h2><div class="evidence">${boxes.map(([t, h]) => `<div class="box"><b>${t}</b>${h}</div>`).join("")}</div>
      <details style="margin-top:12px"><summary>Full evidence graph (JSON)</summary><pre>${esc(JSON.stringify(e, null, 2))}</pre></details>`;
  }

  async function poll(id, onUpdate) {
    for (;;) {
      const r = await api(`/api/identify/${id}`);
      onUpdate(r);
      if (!["queued", "running"].includes(r.status)) return r;
      await new Promise((res) => setTimeout(res, 1200));
    }
  }

  async function loadRecent() {
    try {
      const rows = await api("/api/identifications?limit=15");
      $("#recent").innerHTML = rows.map((x) => `<li data-id="${x.id}">${x.input_images[0] ? `<img src="${esc(x.input_images[0])}">` : "<span style='width:48px'></span>"}
        <div><b>${esc(x.top_candidate || "–")}</b> <span class="status ${x.status}">${esc(x.status)}</span><small>${esc(x.input_title || x.input_url || "")} · ${new Date(x.created_at + "Z").toLocaleString()}</small></div></li>`).join("") || "<li class='muted'>none yet</li>";
      $("#recent").querySelectorAll("li[data-id]").forEach((li) => li.onclick = () => showExisting(li.dataset.id));
    } catch (e) { $("#recent").innerHTML = `<li class="error">${esc(e.message)}</li>`; }
  }

  async function showExisting(id) {
    const r = await poll(id, (r) => renderStages(r.stages));
    renderResult(r); renderCandidates(r); renderEvidence(r);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ── pages ──

  function initIdentifyPage() {
    const files = $("#files"), drop = $("#drop"), thumbs = $("#thumbs");
    let picked = [];
    const showThumbs = () => { thumbs.innerHTML = picked.map((f) => `<img src="${URL.createObjectURL(f)}">`).join(""); };
    drop.onclick = () => files.click();
    files.onchange = () => { picked = [...files.files].slice(0, 6); showThumbs(); };
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { picked = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/") || /\.heic$/i.test(f.name)).slice(0, 6); showThumbs(); });

    $("#form").onsubmit = async (e) => {
      e.preventDefault();
      const fd = new FormData();
      picked.forEach((f) => fd.append("files", f));
      ["product_url", "image_url", "title", "description"].forEach((k) => { const v = $("#" + k).value.trim(); if (v) fd.append(k, v); });
      $("#go").disabled = true; $("#formmsg").textContent = "Starting…";
      ["#result", "#candidates", "#evidence"].forEach((s) => $(s).hidden = true);
      try {
        const { id } = await api("/api/identify", { method: "POST", body: fd });
        $("#formmsg").textContent = `id ${id.slice(0, 8)}`;
        const r = await poll(id, (r) => renderStages(r.stages));
        renderResult(r); renderCandidates(r); renderEvidence(r); loadRecent();
      } catch (err) { $("#formmsg").innerHTML = `<span class="error">${esc(err.message)}</span>`; }
      $("#go").disabled = false;
    };

    api("/health").then((h) => {
      const flag = (ok, l) => `<span class="${ok ? "ok" : "bad"}">${ok ? "●" : "○"} ${l}</span>`;
      $("#healthbox").innerHTML = [flag(h.database, "database"), flag(h.stockx_configured, "StockX API"),
        flag(h.vision_configured, `vision (${h.vision_model})`), flag(!!h.search_provider, `web search (${h.search_provider || "none"})`)].join(" &nbsp; ");
    }).catch(() => {});
    loadRecent();
    if (location.hash.length > 1) showExisting(location.hash.slice(1));
  }

  function initReviewPage() {
    let current = null;
    const msg = (t, bad) => { $("#actionmsg").innerHTML = bad ? `<span class="error">${esc(t)}</span>` : esc(t); };

    async function loadQueue(selectId) {
      const rows = await api("/api/review/queue");
      $("#qcount").textContent = `(${rows.length})`;
      $("#queue").innerHTML = rows.map((x) => `<li data-id="${x.id}" class="${x.id === selectId ? "active" : ""}">${x.input_images[0] ? `<img src="${esc(x.input_images[0])}">` : "<span style='width:48px'></span>"}
        <div><b>${esc(x.top_candidate || "no candidate")}</b> <span class="status ${x.status}">${esc(x.status)}</span><small>${esc(x.input_title || x.input_url || "")}</small><small>${(x.failure_codes || []).join(", ")}</small></div></li>`).join("") || "<li class='muted'>Queue is empty.</li>";
      $("#queue").querySelectorAll("li[data-id]").forEach((li) => li.onclick = () => show(li.dataset.id));
    }

    async function show(id) {
      const r = await api(`/api/review/${id}`);
      current = r;
      location.hash = id;
      $("#queue").querySelectorAll("li").forEach((li) => li.classList.toggle("active", li.dataset.id === id));
      const page = (r.evidence || {}).page || {};
      $("#detail").innerHTML = `<h2>Retailer product <span class="status ${r.status}">${esc(r.status)}</span></h2>
        <div class="row">${(r.input_images || []).map((u) => `<img src="${esc(u)}" style="width:160px;height:160px;object-fit:cover;border-radius:10px;border:1px solid var(--line)">`).join("")}</div>
        <dl class="kv"><dt>Title</dt><dd>${esc(r.input_title || page.title || "–")}</dd><dt>URL</dt><dd>${r.input_url ? `<a href="${esc(r.input_url)}" target="_blank" rel="noopener">${esc(r.input_url)}</a>` : "–"}</dd>
        <dt>Price</dt><dd>${esc(page.price ?? "–")}</dd><dt>System pick</dt><dd>${esc(r.product?.style_code || "none")} @ ${pct(r.confidence)}</dd>
        <dt>Flags</dt><dd class="chips">${(r.failure_codes || []).map((f) => `<span class="chip bad">${esc(f)}</span>`).join("") || "–"}</dd></dl>`;
      renderCandidates(r, $("#matches"), (code) => decide({ action: "select", style_code: code }));
      $("#matches").hidden = false;
      $("#actions").hidden = false;
      $("#btn-confirm").disabled = !r.product;
      renderEvidence(r, $("#evidence"));
      msg("");
    }

    async function decide(body) {
      if (!current) return;
      msg("Saving…");
      try {
        const r = await api(`/api/review/${current.id}/decision`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        msg(`Saved: ${body.action}${r.product ? " → " + r.product.style_code : ""}. Query photos stored as labeled reference images.`);
        await loadQueue();
        renderCandidates(r, $("#matches"));
        $("#detail").querySelector(".status").className = `status ${r.status}`;
      } catch (e) { msg(e.message, true); }
    }

    $("#btn-confirm").onclick = () => decide({ action: "confirm" });
    $("#btn-reject").onclick = () => decide({ action: "reject" });
    $("#btn-manual").onclick = () => { const v = $("#manual_sku").value.trim(); if (v) decide({ action: "manual_sku", style_code: v }); };
    $("#btn-rerun").onclick = async () => {
      if (!current) return;
      msg("Re-running…");
      try { const { id } = await api(`/api/identify/${current.id}/rerun`, { method: "POST" }); msg(`Started ${id.slice(0, 8)} — open it on the Identify page`); }
      catch (e) { msg(e.message, true); }
    };
    loadQueue(location.hash.slice(1)).then(() => { if (location.hash.length > 1) show(location.hash.slice(1)); });
  }

  return { initIdentifyPage, initReviewPage };
})();
