#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local web app for phrasebook processing tasks.
"""

from __future__ import annotations

import cgi
import html
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSING_SCRIPT = REPO_ROOT / "tools" / "guidedeconversationphrasebook_processing.py"
RAPIDFUZZ_SCRIPT = REPO_ROOT / "tools" / "merge_all_languages_rapidfuzz.py"
GENERIC_DOCX_SCRIPT = REPO_ROOT / "tools" / "generic_docx_batch.py"
DEFAULT_BASE_PATH = r"G:\My Drive\Mbú'ŋwɑ̀'nì"
LANGUAGE_PATHS = {
    "Basaa": r"Livres Basaa\Basaa_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Chichewa": r"Livres Chichewa\Chichewa_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "DualaDouala": r"Livres Duala\DualaDouala_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "EweTogo": r"Livres EweTogo\EweTogo_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Ewondo": r"Livres Ewondo\Ewondo_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "FulfuldeBenin": r"Livres Fulfulde\FulfuldeBenin_Benin_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "FulfuldeNigeria": r"Livres Fulfulde_Nigeria\FulfuldeNigeria_Nigeria_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Ghomala": r"Livres Ghomala\Ghomala_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Hausa": r"Livres Hausa\Hausa_Benin_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Igbo": r"Livres Igbo\Igbo_Nigeria_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Kikongo": r"Livres Kikongo\Kikongo_RDCongo_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Kinyarwanda": r"Livres Kinyarwanda\Kinyarwanda_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Lingala": r"Livres Lingala\Lingala_RDCongo_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Medumba": r"Livres Medumba\Medumba_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Nufi": r"Livres Nufi\Nufi_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Shupamom": r"Livres Bamoun\Bamoun_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Swahili": r"Livres Swahili\Swahili_TanzaniaKenya_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Tshiluba": r"Livres Tshiluba\Tshiluba_RDCongo_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Twi": r"Livres Twi\Twi_Ghana_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Wolof": r"Livres Wolof\Wolof_Senegal_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Yemba": r"Livres Yemba\Yemba_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Yoruba": r"Livres Yoruba\Yoruba_Nigeria_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
    "Zulu": r"Livres Zulu\Zulu_Phrasebook_ExpressionsUsuelles_GuideConversation.docx",
}
LANGUAGES = sorted(LANGUAGE_PATHS)
ALL_FILES_LABEL = "All files"

UPLOAD_STATE: dict[str, object] = {
    "source_base": None,
    "output_base": None,
    "languages": [],
    "files": [],
    "mode": None,
}
SERVER_VERSION = str(int(time.time() * 1000))
SELF_PATH = Path(__file__).resolve()

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phrasebook Studio</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root { --bg:#0f1117; --surface:#171b24; --surface2:#1e2330; --border:rgba(255,255,255,.07); --border2:rgba(255,255,255,.12); --accent:#4f9cf9; --green:#3ecf8e; --red:#ef4444; --text:#e8eaf0; --muted:#7a8299; --mono:'IBM Plex Mono', monospace; --sans:'IBM Plex Sans', sans-serif; }
  :root[data-theme="light"] { --bg:#f3f6fb; --surface:#ffffff; --surface2:#eaf0f8; --border:rgba(15,23,42,.08); --border2:rgba(15,23,42,.16); --accent:#2f7cf6; --green:#159f67; --red:#dc2626; --text:#132033; --muted:#5f6c85; }
  body { font-family:var(--sans); background:var(--bg); color:var(--text); font-size:14px; line-height:1.5; min-height:100vh; transition:background .2s ease, color .2s ease; }
  .app { display:grid; grid-template-columns:220px 1fr; grid-template-rows:56px 1fr; min-height:100vh; }
  .topbar { grid-column:1 / -1; background:var(--surface); border-bottom:1px solid var(--border); display:flex; align-items:center; gap:12px; padding:0 20px; }
  .logo { font-family:var(--mono); font-size:13px; font-weight:500; color:var(--accent); letter-spacing:.04em; display:flex; align-items:center; gap:8px; }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 8px var(--accent); }
  .topbar-right { margin-left:auto; display:flex; align-items:center; gap:10px; }
  .status-pill { display:flex; align-items:center; gap:6px; background:rgba(62,207,142,.1); border:1px solid rgba(62,207,142,.25); border-radius:20px; padding:6px 12px; font-family:var(--mono); font-size:11px; color:var(--green); }
  .pip { width:6px; height:6px; border-radius:50%; background:var(--green); box-shadow:0 0 8px var(--green); }
  .theme-toggle { background:var(--surface2); color:var(--text); border:1px solid var(--border2); border-radius:20px; padding:6px 12px; font-family:var(--mono); font-size:11px; cursor:pointer; transition:background .15s ease, border-color .15s ease, color .15s ease; }
  .theme-toggle:hover { border-color:var(--accent); color:var(--accent); }
  .sidebar { background:var(--surface); border-right:1px solid var(--border); padding:16px 0; }
  .nav-section { padding:0 12px; }
  .nav-label { font-size:10px; font-weight:500; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; padding:0 8px; margin-bottom:10px; }
  .nav-item { display:flex; align-items:center; gap:10px; padding:12px 14px; border-radius:10px; cursor:pointer; font-size:13px; color:var(--muted); border:1px solid transparent; margin-bottom:6px; }
  .nav-item:hover { color:var(--text); background:var(--surface2); }
  .nav-item.active { color:var(--accent); background:rgba(79,156,249,.08); border-color:rgba(79,156,249,.15); }
  .main { padding:28px 32px; overflow-y:auto; position:relative; }
  .page { display:none; } .page.active { display:block; }
  h1 { font-size:18px; font-weight:600; color:var(--text); margin-bottom:4px; }
  .page-sub { font-size:13px; color:var(--muted); margin-bottom:28px; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:20px; margin-bottom:16px; }
  .card-title { font-size:11px; font-weight:500; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:16px; }
  .field { margin-bottom:14px; } .field label { display:block; font-size:12px; color:var(--muted); margin-bottom:5px; font-weight:500; }
  .field input, .field select { width:100%; background:var(--bg); border:1px solid var(--border2); border-radius:8px; color:var(--text); font-family:var(--mono); font-size:13px; padding:10px 12px; outline:none; }
  .field input:focus, .field select:focus { border-color:var(--accent); }
  .language-picker { position:relative; }
  .language-trigger { width:100%; background:var(--bg); border:1px solid var(--border2); border-radius:8px; color:var(--text); font-family:var(--mono); font-size:13px; padding:10px 12px; text-align:left; cursor:pointer; }
  .language-trigger:hover, .language-trigger.open { border-color:var(--accent); }
  .language-menu { position:absolute; top:calc(100% + 8px); left:0; right:0; background:var(--surface2); border:1px solid var(--border2); border-radius:10px; padding:10px; box-shadow:0 16px 40px rgba(0,0,0,.35); display:none; z-index:20; }
  .language-menu.open { display:block; }
  .language-search { width:100%; background:var(--bg); border:1px solid var(--border2); border-radius:8px; color:var(--text); font-family:var(--mono); font-size:12px; padding:9px 10px; margin-bottom:10px; outline:none; }
  .language-options { max-height:220px; overflow-y:auto; border:1px solid var(--border); border-radius:8px; background:rgba(15,17,23,.65); }
  .language-option { display:flex; align-items:center; gap:10px; padding:9px 10px; font-size:12px; color:var(--text); border-bottom:1px solid var(--border); }
  .language-option:last-child { border-bottom:none; }
  .language-option input { width:auto; margin:0; accent-color:var(--accent); }
  .language-option.hidden { display:none; }
  .picker-actions { display:flex; gap:8px; margin-top:10px; }
  .picker-actions button { flex:1; }
  .fields-row { display:grid; grid-template-columns:1.2fr .8fr .8fr; gap:14px; }
  .btn-group { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
  .sticky-actions { margin:20px auto 16px; padding:20px 24px; max-width:860px; background:rgba(23,27,36,.98); border:1px solid var(--border); border-radius:14px; box-shadow:0 10px 24px rgba(0,0,0,.22); backdrop-filter:blur(10px); }
  .sticky-actions .card-title { text-align:center; }
  .actions-grid { display:grid; grid-template-columns:repeat(3, minmax(180px, 1fr)); gap:12px; align-items:stretch; }
  .actions-grid .btn { width:100%; min-height:44px; justify-content:center; text-align:center; }
  .actions-note { margin-top:12px; font-size:12px; color:var(--muted); line-height:1.6; text-align:center; }
  .btn { display:inline-flex; align-items:center; gap:6px; padding:9px 14px; border-radius:8px; font-size:12px; font-weight:500; cursor:pointer; border:none; font-family:var(--sans); white-space:nowrap; }
  .btn-primary { background:var(--accent); color:#0a1628; } .btn-primary:hover { background:#6aacff; }
  .btn-secondary { background:var(--surface2); color:var(--text); border:1px solid var(--border2); }
  .btn-danger { background:rgba(239,68,68,.12); color:var(--red); border:1px solid rgba(239,68,68,.2); }
  .btn-ghost { background:transparent; color:var(--muted); border:1px solid var(--border); }
  .status-block { background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:14px 16px; display:flex; align-items:center; gap:12px; }
  .status-indicator { width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 8px var(--green); }
  .status-text { font-size:13px; color:var(--text); font-family:var(--mono); }
  .status-sub, .small { font-size:11px; color:var(--muted); margin-top:2px; font-family:var(--mono); }
  .info-row { display:flex; gap:6px; margin-bottom:10px; font-size:12px; font-family:var(--mono); }
  .info-row .k { color:var(--muted); min-width:170px; } .info-row .v { color:var(--text); }
  .log-box { background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:14px 16px; font-family:var(--mono); font-size:12px; color:var(--muted); min-height:340px; max-height:560px; overflow-y:auto; white-space:pre-wrap; user-select:text; }
  .progress-actions { display:flex; gap:8px; margin-top:14px; flex-wrap:wrap; }
  .tag { display:inline-flex; align-items:center; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:500; font-family:var(--mono); background:rgba(79,156,249,.1); color:var(--accent); border:1px solid rgba(79,156,249,.2); }
  .divider { border:none; border-top:1px solid var(--border); margin:20px 0; }
  .dropzone { border:1px dashed rgba(79,156,249,.35); border-radius:10px; padding:18px; background:rgba(79,156,249,.04); color:var(--muted); text-align:center; margin-bottom:12px; }
  .dropzone.drag { border-color:var(--accent); background:rgba(79,156,249,.10); color:var(--text); }
  .rule-table { width:100%; border-collapse:collapse; } .rule-table th, .rule-table td { padding:8px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; } .rule-table th { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; } .rule-table input { width:100%; background:var(--bg); border:1px solid var(--border2); border-radius:6px; color:var(--text); font-family:var(--mono); font-size:12px; padding:8px 10px; }
  .two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:980px){ .app{grid-template-columns:1fr; grid-template-rows:56px auto 1fr;} .sidebar{border-right:none; border-bottom:1px solid var(--border);} .fields-row,.two-col,.actions-grid{grid-template-columns:1fr;} .sticky-actions{max-width:none;} }
</style>
</head>
<body>
<div class="app">
  <div class="topbar"><div class="logo"><span class="dot"></span>Phrasebook Studio</div><div class="topbar-right"><button id="themeToggle" class="theme-toggle" type="button" onclick="toggleTheme()">Theme: Dark</button><div class="status-pill"><span class="pip"></span><span id="topStatus">Ready</span></div></div></div>
  <div class="sidebar"><div class="nav-section"><div class="nav-label">Workspace</div><div class="nav-item active" onclick="showPage('actions', this)">Actions</div><div class="nav-item" onclick="showPage('progress', this)">Progress</div></div></div>
  <div class="main">
    <div class="page active" id="page-actions">
      <h1>Run Tasks</h1><div class="page-sub">Configure your source or output folder, upload documents if needed, then run a task.</div>
      <div class="card">
        <div class="card-title">Configuration</div>
        <div class="fields-row">
          <div class="field"><label>Base Path / Output Folder</label><input id="basePath" type="text" value="__BASE_PATH__"></div>
          <div class="field">
            <label>Pick Your Language(s)</label>
            <div class="language-picker" id="languagePicker">
              <button type="button" id="languageTrigger" class="language-trigger" onclick="toggleLanguageMenu()">Basaa</button>
              <div id="languageMenu" class="language-menu">
                <input id="languageSearch" class="language-search" type="text" placeholder="Search languages..." oninput="filterLanguageOptions()">
                <div id="languageOptions" class="language-options">__LANGUAGE_OPTIONS__</div>
                <div class="picker-actions">
                  <button type="button" class="btn btn-ghost" onclick="setAllLanguagesChecked(true)">Select All</button>
                  <button type="button" class="btn btn-ghost" onclick="setAllLanguagesChecked(false)">Clear</button>
                </div>
              </div>
            </div>
          </div>
          <div class="field"><label>RapidFuzz Threshold</label><input id="threshold" type="number" value="85" min="0" max="100"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Upload Phrasebooks</div>
        <div id="uploadDropZone" class="dropzone">Drop .docx files here or use the picker below. Files upload immediately when dropped. When uploads are active, the path above becomes the output folder.</div>
        <div class="field" style="margin-bottom:0;">
          <label>Generic Upload Source Folder</label>
          <input id="genericUploadSourceDir" type="text" placeholder="Optional. Example: C:\Users\tcham\Downloads">
        </div>
        <div class="btn-group">
          <input id="uploadInput" type="file" accept=".docx" multiple style="display:none">
          <button class="btn btn-secondary" onclick="document.getElementById('uploadInput').click()">Choose .docx Files</button>
          <button class="btn btn-secondary" onclick="uploadDocs()">Upload Selected Docs</button>
          <button class="btn btn-ghost" onclick="clearUploads()">Clear Uploaded Docs</button>
        </div>
        <div id="selectedUploadFiles" class="small" style="margin-top:10px;">No files selected.</div>
        <div id="uploadError" class="small" style="margin-top:10px; color:#ef4444;"></div>
        <div id="uploadStatus" class="small" style="margin-top:12px;">No uploaded documents are active.</div>
      </div>
      <div class="two-col">
        <div class="card">
          <div class="card-title">In-Place Replacement Dictionary</div>
          <div id="docCsvDropZone" class="dropzone">Drop a CSV file here for doc replacements.</div>
          <div class="btn-group"><input id="docCsvInput" type="file" accept=".csv" style="display:none"><button class="btn btn-secondary" onclick="document.getElementById('docCsvInput').click()">Choose CSV</button><button class="btn btn-ghost" onclick="addRuleRow('doc')">Add Row</button><button class="btn btn-ghost" onclick="clearRules('doc')">Clear</button></div>
          <table class="rule-table"><thead><tr><th>Replace</th><th>By</th><th></th></tr></thead><tbody id="docRulesBody"></tbody></table>
        </div>
        <div class="card">
          <div class="card-title">Normalization Replacements</div>
          <div id="normCsvDropZone" class="dropzone">Drop a CSV file here for normalization replacements.</div>
          <div class="btn-group"><input id="normCsvInput" type="file" accept=".csv" style="display:none"><button class="btn btn-secondary" onclick="document.getElementById('normCsvInput').click()">Choose CSV</button><button class="btn btn-ghost" onclick="addRuleRow('norm')">Add Row</button><button class="btn btn-ghost" onclick="clearRules('norm')">Clear</button></div>
          <table class="rule-table"><thead><tr><th>Replace</th><th>By</th><th></th></tr></thead><tbody id="normRulesBody"></tbody></table>
        </div>
      </div>
      <div class="sticky-actions">
        <div class="card-title">Actions</div>
        <div class="actions-grid">
          <button class="btn btn-primary" onclick="runTask('nbsp_inplace')">Run</button>
          <button class="btn btn-secondary" onclick="runTask('main_rapidfuzz')">Run and merge</button>
          <button class="btn btn-ghost" onclick="runTask('rapidfuzz_only')">Merge-only</button>
        </div>
        <div class="actions-note">Use Run to apply in-place replacements only. Use Run and merge for the full workflow. Use Merge-only when the processing workbook already exists and you only want to rerun the final merge.</div>
      </div>
      <div class="card">
        <div class="card-title">Current Status</div>
        <div class="status-block"><div class="status-indicator"></div><div><div class="status-text" id="statusTextMain">Ready</div><div class="status-sub" id="statusSubMain">No task is currently running</div></div></div>
        <hr class="divider">
        <div class="info-row"><span class="k">NBSP Test Modes</span><span class="v">Do not modify source document unless using the explicit in-place action</span></div>
        <div class="info-row"><span class="k">RapidFuzz Outputs</span><span class="v"><span class="tag">African_Languages_dataframes_merged.csv</span> &nbsp;<span class="tag">African_Languages_dataframes_rapidfuzz_merged.xlsx</span></span></div>
        <div class="info-row"><span class="k">Recommended Flow</span><span class="v">Run main processing first, then RapidFuzz only if needed</span></div>
      </div>
    </div>
    <div class="page" id="page-progress"><h1>Task Progress</h1><div class="page-sub">Live command output for the running task. Running tasks continue server-side unless stopped.</div><div class="card"><div class="card-title">Output Log</div><div id="logBox" class="log-box">Waiting for task to start...</div><div class="progress-actions"><button class="btn btn-secondary" onclick="copyLogs()">Copy Logs</button><button class="btn btn-secondary" onclick="downloadLogs()">Download Logs</button><button class="btn btn-danger" onclick="stopTask()">Stop Current Task</button></div></div></div>
  </div>
</div>
<script>
const THEME_STORAGE_KEY = 'phrasebook-theme';
function getPreferredTheme(){
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  if(savedTheme === 'light' || savedTheme === 'dark'){ return savedTheme; }
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}
function updateThemeToggle(theme){
  const button = document.getElementById('themeToggle');
  if(button){ button.textContent = `Theme: ${theme === 'light' ? 'Light' : 'Dark'}`; }
}
function applyTheme(theme){
  document.documentElement.dataset.theme = theme;
  updateThemeToggle(theme);
}
function toggleTheme(){
  const nextTheme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
  applyTheme(nextTheme);
}
function syncActionsBar(){
  const actionsPage = document.getElementById('page-actions');
  const actionsBar = actionsPage ? actionsPage.querySelector('.sticky-actions') : null;
  if(!actionsPage || !actionsBar){ return; }
  actionsBar.style.position = '';
  actionsBar.style.top = '';
  actionsBar.style.left = '';
  actionsBar.style.width = '';
  actionsPage.style.paddingTop = '';
}
function showPage(page, el){ document.querySelectorAll('.page').forEach(p=>p.classList.remove('active')); document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active')); document.getElementById('page-'+page).classList.add('active'); el.classList.add('active'); syncActionsBar(); }
function rulesBody(kind){ return document.getElementById(kind==='doc' ? 'docRulesBody' : 'normRulesBody'); }
function escapeHtml(value){ return String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;'); }
function addRuleRow(kind, replaceValue='', byValue=''){ const row=document.createElement('tr'); row.innerHTML=`<td><input data-role="replace" value="${escapeHtml(replaceValue)}"></td><td><input data-role="by" value="${escapeHtml(byValue)}"></td><td><button class="btn btn-ghost" type="button" style="padding:8px 10px;">Remove</button></td>`; row.querySelector('button').addEventListener('click',()=>row.remove()); rulesBody(kind).appendChild(row); }
function clearRules(kind){ rulesBody(kind).innerHTML=''; addRuleRow(kind); addRuleRow(kind); }
function collectRules(kind){ return Array.from(rulesBody(kind).querySelectorAll('tr')).map(row=>({ replace: row.querySelector('[data-role="replace"]').value, by: row.querySelector('[data-role="by"]').value })).filter(row=>row.replace.trim() || row.by.trim()); }
function selectedLanguages(){ return Array.from(document.querySelectorAll('.language-checkbox:checked')).map(node=>node.value); }
function updateLanguageTrigger(){ const values=selectedLanguages(); const trigger=document.getElementById('languageTrigger'); if(!values.length){ trigger.textContent='Choose language(s)'; return; } if(values.includes('__ALL_FILES__')){ trigger.textContent='All files'; return; } if(values.length<=2){ trigger.textContent=values.join(', '); return; } trigger.textContent=`${values.length} languages selected`; }
function collectPayload(action){ return { action, base_path: document.getElementById('basePath').value, languages: selectedLanguages(), threshold: document.getElementById('threshold').value, doc_replacements: collectRules('doc'), normalization_replacements: collectRules('norm') }; }
function parseCsvText(text){ const rows=text.replace(/\r/g,'').split('\n').filter(Boolean).map(line=>line.split(',').map(part=>part.trim().replace(/^"|"$/g,''))); if(!rows.length) return []; const header=rows[0].map(v=>v.toLowerCase()); const useHeader=header.includes('replace')||header.includes('by')||header.includes('old')||header.includes('new'); const dataRows=useHeader ? rows.slice(1) : rows; const replaceIndex=useHeader ? (header.indexOf('replace')>=0 ? header.indexOf('replace') : Math.max(header.indexOf('old'),0)) : 0; const byIndex=useHeader ? (header.indexOf('by')>=0 ? header.indexOf('by') : Math.max(header.indexOf('new'),1)) : 1; return dataRows.map(row=>({ replace: row[replaceIndex]||'', by: row[byIndex]||'' })).filter(row=>row.replace||row.by); }
let pendingUploadFiles = [];
let pendingUploadSourceDir = '';
let pauseLogRefresh = false;
let pendingLogText = null;
let knownServerVersion = null;
function syncAllFilesState(changed){ const boxes=Array.from(document.querySelectorAll('.language-checkbox')); const allBox=document.querySelector('.language-checkbox[value="__ALL_FILES__"]'); if(!allBox) return; if(changed === allBox && allBox.checked){ boxes.forEach(box=>{ if(box !== allBox) box.checked=false; }); } else if(changed !== allBox && changed && changed.checked){ allBox.checked=false; } if(!boxes.some(box=>box.checked)){ allBox.checked=true; } updateLanguageTrigger(); }
function toggleLanguageMenu(){ document.getElementById('languageMenu').classList.toggle('open'); document.getElementById('languageTrigger').classList.toggle('open'); }
function closeLanguageMenu(){ document.getElementById('languageMenu').classList.remove('open'); document.getElementById('languageTrigger').classList.remove('open'); }
function filterLanguageOptions(){ const query=document.getElementById('languageSearch').value.trim().toLowerCase(); document.querySelectorAll('.language-option').forEach(node=>{ const label=(node.dataset.label||'').toLowerCase(); node.classList.toggle('hidden', !!query && !label.includes(query)); }); }
function setAllLanguagesChecked(selectAll){ const boxes=Array.from(document.querySelectorAll('.language-checkbox')); const allBox=document.querySelector('.language-checkbox[value="__ALL_FILES__"]'); if(selectAll){ boxes.forEach(box=>{ box.checked = box !== allBox; }); if(allBox) allBox.checked=false; } else { boxes.forEach(box=>{ box.checked=false; }); if(allBox) allBox.checked=true; } updateLanguageTrigger(); filterLanguageOptions(); }
function initLanguagePicker(){ document.querySelectorAll('.language-checkbox').forEach(box=>{ box.addEventListener('change', event=>syncAllFilesState(event.target)); }); document.addEventListener('click', event=>{ const picker=document.getElementById('languagePicker'); if(picker && !picker.contains(event.target)){ closeLanguageMenu(); } }); updateLanguageTrigger(); }
function dirname(path){ return path.replace(/[\\/]+$/, '').replace(/[\\/][^\\/]+$/, ''); }
function knownLanguageNames(){ return Array.from(document.querySelectorAll('.language-checkbox')).map(node=>node.value).filter(value=>value && value !== '__ALL_FILES__'); }
function looksLikePhrasebookUpload(fileName){
  const lowerName = String(fileName || '').toLowerCase();
  return knownLanguageNames().some(language => lowerName.includes(language.toLowerCase()));
}
function inferUploadSourceDir(files){
  const firstFile = Array.isArray(files) && files.length ? files[0] : null;
  if(!firstFile){ return ''; }
  if(typeof firstFile.path === 'string' && firstFile.path && !firstFile.path.toLowerCase().includes('fakepath')){
    return dirname(firstFile.path);
  }
  return '';
}
function syncGenericSourceField(){
  const input = document.getElementById('genericUploadSourceDir');
  if(!input){ return; }
  if(pendingUploadSourceDir){
    input.value = pendingUploadSourceDir;
  }
}
function setPendingUploadFiles(files){
  pendingUploadFiles = Array.from(files || []);
  pendingUploadSourceDir = inferUploadSourceDir(pendingUploadFiles);
  syncGenericSourceField();
  const label = document.getElementById('selectedUploadFiles');
  if(!pendingUploadFiles.length){
    label.textContent = 'No files selected.';
    return;
  }
  label.textContent = 'Pending upload: ' + pendingUploadFiles.map(file => file.name).join(', ');
  if(pendingUploadSourceDir){
    label.textContent += ' | Source folder: ' + pendingUploadSourceDir;
  }
}
function syncBasePathField(upload){
  const basePathInput = document.getElementById('basePath');
  if(!basePathInput){ return; }
  if(upload && upload.files && upload.files.length && upload.source_base){
    if(!basePathInput.dataset.manualValue){
      basePathInput.dataset.manualValue = basePathInput.value;
    }
    basePathInput.value = upload.source_base;
    basePathInput.title = 'Using uploaded files from this folder';
    return;
  }
  if(basePathInput.dataset.manualValue){
    basePathInput.value = basePathInput.dataset.manualValue;
    delete basePathInput.dataset.manualValue;
  }
  basePathInput.title = '';
}
async function importCsv(kind, file){ if(!file) return; const text=await file.text(); const rows=parseCsvText(text); if(!rows.length){ alert('No replacement rows found in that CSV.'); return; } rulesBody(kind).innerHTML=''; rows.forEach(row=>addRuleRow(kind,row.replace,row.by)); }
async function uploadDocs(){ const errorBox=document.getElementById('uploadError'); errorBox.textContent=''; if(!pendingUploadFiles.length){ errorBox.textContent='Choose or drop one or more .docx files first.'; return false; } const form=new FormData(); const basePathInput=document.getElementById('basePath'); const genericSourceInput=document.getElementById('genericUploadSourceDir'); form.append('output_base', basePathInput.dataset.manualValue || basePathInput.value); const chosenUploadSourceDir = pendingUploadSourceDir || (genericSourceInput ? genericSourceInput.value.trim() : ''); if(chosenUploadSourceDir){ form.append('upload_source_dir', chosenUploadSourceDir); } pendingUploadFiles.forEach(file=>form.append('files', file, file.name)); const response=await fetch('/upload',{ method:'POST', body:form }); const data=await response.json(); if(!response.ok){ errorBox.textContent=data.error || 'Upload failed'; refreshStatus(); return false; } pendingUploadFiles = []; pendingUploadSourceDir = ''; document.getElementById('uploadInput').value=''; setPendingUploadFiles([]); syncBasePathField(data.upload); refreshStatus(); return true; }
async function clearUploads(){ document.getElementById('uploadError').textContent=''; pendingUploadFiles = []; pendingUploadSourceDir = ''; document.getElementById('uploadInput').value=''; setPendingUploadFiles([]); const genericSourceInput=document.getElementById('genericUploadSourceDir'); if(genericSourceInput){ genericSourceInput.value=''; } const response=await fetch('/clear-uploads',{ method:'POST' }); const data=await response.json(); if(!response.ok) document.getElementById('uploadError').textContent=data.error || 'Failed to clear uploads'; refreshStatus(); }
async function runTask(action){ if(pendingUploadFiles.length){ const uploaded = await uploadDocs(); if(!uploaded) return; } const response=await fetch('/run',{ method:'POST', headers:{ 'Content-Type':'application/json' }, body:JSON.stringify(collectPayload(action)) }); const data=await response.json(); if(!response.ok){ alert(data.error || 'Failed to start task'); return; } document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active')); document.querySelectorAll('.page').forEach(p=>p.classList.remove('active')); document.getElementById('page-progress').classList.add('active'); document.querySelectorAll('.nav-item')[1].classList.add('active'); refreshStatus(); }
async function stopTask(){ const response=await fetch('/stop',{ method:'POST' }); const data=await response.json(); if(!response.ok) alert(data.error || 'Failed to stop task'); refreshStatus(); }
async function copyLogs(){ const text=document.getElementById('logBox').textContent || ''; if(!text.trim()) return; try { await navigator.clipboard.writeText(text); } catch (error) { const area=document.createElement('textarea'); area.value=text; document.body.appendChild(area); area.select(); document.execCommand('copy'); area.remove(); } }
function downloadLogs(){ const text=document.getElementById('logBox').textContent || ''; const blob=new Blob([text], { type:'text/plain;charset=utf-8' }); const url=URL.createObjectURL(blob); const link=document.createElement('a'); const stamp=new Date().toISOString().replace(/[:.]/g,'-'); link.href=url; link.download=`phrasebook-log-${stamp}.txt`; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url); }
function applyLogText(text){ document.getElementById('logBox').textContent = text || 'Waiting for task to start...'; }
function updateLogSelectionState(){ const selection=window.getSelection(); const logBox=document.getElementById('logBox'); const insideLog = !!selection && selection.rangeCount > 0 && logBox.contains(selection.anchorNode); pauseLogRefresh = insideLog && !selection.isCollapsed; if(!pauseLogRefresh && pendingLogText !== null){ applyLogText(pendingLogText); pendingLogText = null; } }
async function checkServerVersion(){ try { const response = await fetch('/version'); const data = await response.json(); if(!knownServerVersion){ knownServerVersion = data.server_version; return; } if(data.server_version !== knownServerVersion){ window.location.reload(); } } catch (error) {} }
async function refreshStatus(){ const response=await fetch('/status'); const data=await response.json(); document.getElementById('topStatus').textContent=data.running ? 'Running' : (data.status || 'Ready'); document.getElementById('statusTextMain').textContent=data.running ? 'Running' : (data.status || 'Ready'); document.getElementById('statusSubMain').textContent=data.message || ''; const nextLog = data.log || 'Waiting for task to start...'; if(pauseLogRefresh){ pendingLogText = nextLog; } else { applyLogText(nextLog); pendingLogText = null; } const uploadStatus=document.getElementById('uploadStatus'); if(data.upload && data.upload.files && data.upload.files.length){ syncBasePathField(data.upload); if(data.upload.mode === 'generic'){ uploadStatus.textContent = 'Using uploaded generic .docx files from: ' + (data.upload.source_base || '') + ' | Output folder: ' + (data.upload.output_base || '') + ' | Generic mode is active.'; } else { uploadStatus.textContent = 'Using uploaded phrasebook files from: ' + (data.upload.source_base || '') + ' | Output folder: ' + (data.upload.output_base || ''); } } else if (pendingUploadFiles.length) { uploadStatus.textContent='Files are selected locally but not uploaded yet. Running a task will upload them first.'; } else { syncBasePathField(null); uploadStatus.textContent='No uploaded documents are active.'; } }
function wireDropzone(id, handler){ const zone=document.getElementById(id); zone.addEventListener('dragover', event=>{ event.preventDefault(); zone.classList.add('drag'); }); zone.addEventListener('dragleave', ()=>zone.classList.remove('drag')); zone.addEventListener('drop', event=>{ event.preventDefault(); zone.classList.remove('drag'); handler(Array.from(event.dataTransfer.files || [])); }); }
document.getElementById('docCsvInput').addEventListener('change', event=>importCsv('doc', event.target.files[0]));
document.getElementById('normCsvInput').addEventListener('change', event=>importCsv('norm', event.target.files[0]));
document.getElementById('uploadInput').addEventListener('change', async event=>{ setPendingUploadFiles(event.target.files); refreshStatus(); });
wireDropzone('docCsvDropZone', files=>{ if(files[0]) importCsv('doc', files[0]); });
wireDropzone('normCsvDropZone', files=>{ if(files[0]) importCsv('norm', files[0]); });
wireDropzone('uploadDropZone', async files=>{ setPendingUploadFiles(files); refreshStatus(); if(files && files.length){ await uploadDocs(); } });
document.addEventListener('selectionchange', updateLogSelectionState);
window.addEventListener('resize', syncActionsBar);
applyTheme(getPreferredTheme());
clearRules('doc'); clearRules('norm'); setInterval(refreshStatus, 1200); setInterval(checkServerVersion, 1500); checkServerVersion(); refreshStatus();
initLanguagePicker();
syncActionsBar();
</script>
</body>
</html>
"""


class TaskManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.log = deque(maxlen=10000)
        self.status = "Ready"
        self.message = "No task is currently running."

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "running": self.process is not None and self.process.poll() is None,
                "status": self.status,
                "message": self.message,
                "log": "".join(self.log),
                "upload": dict(UPLOAD_STATE),
            }

    def start(self, cmd: list[str], label: str, cwd: str | None = None) -> tuple[bool, str]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False, "A task is already running."
            self.status = "Running"
            self.message = f"Running: {label}"
            self.log.clear()
            self.log.append(f"=== {label} ===\n")
            self.log.append(" ".join(cmd) + "\n\n")

        def runner() -> None:
            try:
                env = dict(os.environ)
                env["PYTHONIOENCODING"] = "utf-8"
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd or str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                with self.lock:
                    self.process = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    with self.lock:
                        self.log.append(line)
                code = proc.wait()
                with self.lock:
                    self.process = None
                    self.status = "Ready" if code == 0 else "Failed"
                    self.message = (
                        f"Finished successfully: {label}"
                        if code == 0
                        else f"Task failed with exit code {code}: {label}"
                    )
            except Exception as exc:
                with self.lock:
                    self.process = None
                    self.status = "Failed"
                    self.message = f"Failed to start task: {exc}"
                    self.log.append(f"\nFailed to start task: {exc}\n")

        threading.Thread(target=runner, daemon=True).start()
        return True, "Started"

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return False, "No running task to stop."
            self.process.terminate()
            self.status = "Stopping"
            self.message = "Stop requested."
            self.log.append("\nStop requested.\n")
        return True, "Stopping"


TASKS = TaskManager()


def infer_language_from_filename(filename: str) -> str:
    lower_name = Path(filename).name.lower()
    exact_matches = [lang for lang, rel in LANGUAGE_PATHS.items() if Path(rel).name.lower() == lower_name]
    if len(exact_matches) == 1:
        return exact_matches[0]
    contains_matches = [lang for lang in LANGUAGES if lang.lower() in lower_name]
    if len(contains_matches) == 1:
        return contains_matches[0]
    raise ValueError(
        f"Could not infer a phrasebook language from filename: {filename}. "
        f"Use one of the actual phrasebook .docx files, or include a known language name such as Basaa, Ewondo, or Nufi in the filename."
    )


def uploaded_source_dir(output_dir: Path, mode: str) -> Path:
    if mode == "generic":
        return output_dir / "_uploaded_generic_docx"
    return output_dir / "_uploaded_phrasebooks"


def guess_generic_upload_source_dir(filenames: list[str]) -> str | None:
    candidate_dirs = [
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path.home() / "Documents",
    ]
    matched_dirs: set[Path] = set()
    for filename in filenames:
        clean_name = Path(filename).name
        found_path = None
        for candidate_dir in candidate_dirs:
            candidate_path = candidate_dir / clean_name
            if candidate_path.exists():
                found_path = candidate_path
                break
        if found_path is None:
            return None
        matched_dirs.add(found_path.parent.resolve())
    if len(matched_dirs) == 1:
        return str(next(iter(matched_dirs)))
    return None


def prepare_uploaded_files(
    output_base: str,
    uploads: list[tuple[str, bytes]],
    upload_source_dir: str | None = None,
) -> dict[str, object]:
    uploaded_languages: list[str] = []
    uploaded_files: list[str] = []
    generic_files: list[tuple[str, bytes]] = []
    phrasebook_files: list[tuple[str, bytes, str]] = []
    for filename, content in uploads:
        try:
            language_name = infer_language_from_filename(filename)
            uploaded_languages.append(language_name)
            phrasebook_files.append((filename, content, language_name))
        except ValueError:
            generic_files.append((filename, content))

    if phrasebook_files and generic_files:
        raise ValueError(
            "Do not mix phrasebook uploads and generic .docx uploads in one batch. "
            "Upload either known phrasebook files or arbitrary .docx files."
        )

    mode = "generic" if generic_files else "phrasebook"
    if mode == "generic":
        chosen_output_base = upload_source_dir.strip() if upload_source_dir else ""
        if not chosen_output_base:
            guessed_output_base = guess_generic_upload_source_dir([name for name, _ in generic_files])
            if guessed_output_base:
                chosen_output_base = guessed_output_base
        if not chosen_output_base:
            raise ValueError(
                "Could not detect the local folder for this generic .docx upload. "
                "Enter Generic Upload Source Folder first."
            )
    else:
        chosen_output_base = output_base
    output_dir = Path(chosen_output_base).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_base = uploaded_source_dir(output_dir, mode)
    for candidate in (
        uploaded_source_dir(output_dir, "phrasebook"),
        uploaded_source_dir(output_dir, "generic"),
    ):
        if candidate.exists():
            shutil.rmtree(candidate, ignore_errors=True)
    source_base.mkdir(parents=True, exist_ok=True)

    for filename, content, language_name in phrasebook_files:
        rel_path = Path(LANGUAGE_PATHS[language_name])
        target = source_base / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        uploaded_files.append(str(target))

    for filename, content in generic_files:
        rel_path = Path(filename).name
        target = source_base / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        uploaded_files.append(str(target))

    UPLOAD_STATE["source_base"] = str(source_base)
    UPLOAD_STATE["output_base"] = str(output_dir)
    UPLOAD_STATE["languages"] = sorted(set(uploaded_languages))
    UPLOAD_STATE["files"] = uploaded_files
    UPLOAD_STATE["mode"] = mode
    return dict(UPLOAD_STATE)


def clear_uploaded_files() -> None:
    source_base = UPLOAD_STATE.get("source_base")
    if source_base and Path(str(source_base)).exists():
        shutil.rmtree(str(source_base), ignore_errors=True)
    UPLOAD_STATE["source_base"] = None
    UPLOAD_STATE["output_base"] = None
    UPLOAD_STATE["languages"] = []
    UPLOAD_STATE["files"] = []
    UPLOAD_STATE["mode"] = None


def watch_self_and_reload() -> None:
    baseline_mtime = SELF_PATH.stat().st_mtime_ns
    while True:
        time.sleep(1.0)
        try:
            current_mtime = SELF_PATH.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        if current_mtime == baseline_mtime:
            continue
        with TASKS.lock:
            task_running = TASKS.process is not None and TASKS.process.poll() is None
        if task_running:
            baseline_mtime = current_mtime
            continue
        os.execv(sys.executable, [sys.executable] + sys.argv)


def rows_to_mapping(rows: object) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not isinstance(rows, list):
        return mapping
    for row in rows:
        if not isinstance(row, dict):
            continue
        old = str(row.get("replace", "")).strip()
        new = str(row.get("by", "")).strip()
        if old:
            mapping[old] = new
    return mapping


def write_temp_mapping(mapping: dict[str, str], name: str) -> str | None:
    if not mapping:
        return None
    temp_dir = REPO_ROOT / "tools" / "_webapp_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / name
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_temp_list(items: list[str], name: str) -> str | None:
    if not items:
        return None
    temp_dir = REPO_ROOT / "tools" / "_webapp_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / name
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def build_command(payload: dict[str, object]) -> tuple[list[str], str, str | None]:
    action = str(payload.get("action", ""))
    base_path = str(payload.get("base_path", DEFAULT_BASE_PATH)).strip() or DEFAULT_BASE_PATH
    raw_languages = payload.get("languages")
    selected_from_payload = [str(item).strip() for item in raw_languages if str(item).strip()] if isinstance(raw_languages, list) else []
    if not selected_from_payload:
        selected_from_payload = ["Basaa"]
    all_files_selected = ALL_FILES_LABEL in selected_from_payload
    selected_ui_languages = [] if all_files_selected else [lang for lang in selected_from_payload if lang in LANGUAGE_PATHS]
    single_language = selected_ui_languages[0] if len(selected_ui_languages) == 1 else ""
    threshold = str(payload.get("threshold", "85")).strip() or "85"
    active_upload = bool(UPLOAD_STATE.get("source_base") and UPLOAD_STATE.get("files"))
    effective_base_path = str(UPLOAD_STATE["source_base"]) if active_upload else base_path
    command_cwd = str(Path(base_path).resolve()) if active_upload else str(REPO_ROOT)
    if active_upload:
        upload_languages = [lang for lang in UPLOAD_STATE["languages"] if lang in LANGUAGE_PATHS]
        if selected_ui_languages:
            selected_languages_list = [lang for lang in upload_languages if lang in selected_ui_languages]
        else:
            selected_languages_list = upload_languages
    else:
        selected_languages_list = selected_ui_languages
    selected_languages = ",".join(selected_languages_list)
    upload_mode = str(UPLOAD_STATE.get("mode") or "")
    doc_replacements_path = write_temp_mapping(rows_to_mapping(payload.get("doc_replacements")), "doc_replacements.json")
    norm_replacements_path = write_temp_mapping(rows_to_mapping(payload.get("normalization_replacements")), "normalization_replacements.json")

    if active_upload and upload_mode == "generic":
        input_files_json = write_temp_list([str(path) for path in UPLOAD_STATE.get("files", [])], "generic_input_files.json")
        generic_args = [
            sys.executable,
            str(GENERIC_DOCX_SCRIPT),
            "--input-dir",
            effective_base_path,
            "--output-dir",
            command_cwd,
        ]
        if input_files_json:
            generic_args.extend(["--input-files-json", input_files_json])
        if doc_replacements_path:
            generic_args.extend(["--doc-replacements-json", doc_replacements_path])
        if action == "main":
            return generic_args + ["--report-only"], "Generic DOCX Report", command_cwd
        if action == "nbsp_test":
            return generic_args + ["--report-only"], "Generic DOCX NBSP Report", command_cwd
        if action == "nbsp_test_copy":
            return generic_args + ["--create-fixed-copies"], "Generic DOCX Fixed Copies", command_cwd
        if action == "nbsp_inplace":
            return generic_args + ["--inplace"], "Generic DOCX In-Place Cleanup", command_cwd
        if action in {"main_rapidfuzz", "rapidfuzz_only"}:
            raise ValueError("RapidFuzz merge is only available for phrasebook language documents, not arbitrary .docx files.")

    processing_args = [sys.executable, str(PROCESSING_SCRIPT), "--base-path", effective_base_path]
    if selected_languages:
        processing_args.extend(["--language-names", selected_languages])
    if doc_replacements_path:
        processing_args.extend(["--doc-replacements-json", doc_replacements_path])
    if norm_replacements_path:
        processing_args.extend(["--normalization-replacements-json", norm_replacements_path])

    if action == "main":
        return processing_args, "Run", command_cwd
    if action == "main_rapidfuzz":
        return processing_args + ["--run-rapidfuzz-merge"], "Run and merge", command_cwd
    if action == "rapidfuzz_only":
        return [
            sys.executable, str(RAPIDFUZZ_SCRIPT),
            "--excel", "African_Languages_dataframes.xlsx",
            "--base-path", effective_base_path,
            "--master-language", "Nufi",
            "--text-column", "both",
            "--threshold", threshold,
            "--output", "African_Languages_dataframes_merged.csv",
            "--excel-output", "African_Languages_dataframes_rapidfuzz_merged.xlsx",
            "--checkpoint-every-language",
        ], "Merge-only", command_cwd
    if action == "nbsp_test":
        if all_files_selected and not active_upload:
            return (
                processing_args
                + ["--skip-doc-replacements", "--skip-excel", "--skip-quality-checks", "--skip-rapidfuzz-merge", "--skip-merge-csv"],
                "NBSP Test Report (All Files)",
                command_cwd,
            )
        if active_upload and not selected_languages_list:
            raise ValueError("No uploaded phrasebook languages match the current language selection.")
        if len(selected_languages_list) != 1:
            return (
                processing_args
                + ["--skip-doc-replacements", "--skip-excel", "--skip-quality-checks", "--skip-rapidfuzz-merge", "--skip-merge-csv"],
                "NBSP Test Report",
                command_cwd,
            )
        return processing_args + ["--test-nbsp-language", single_language], "NBSP Test Report", command_cwd
    if action == "nbsp_test_copy":
        if len(selected_languages_list) != 1:
            raise ValueError("NBSP Test + Fixed Doc Copy requires exactly one language selection.")
        return processing_args + ["--test-nbsp-language", single_language, "--create-nbsp-fixed-doc-copy"], "NBSP Test + Fixed Doc Copy", command_cwd
    if action == "nbsp_inplace":
        return processing_args + ["--apply-nbsp-fixes-inplace"], "Run", command_cwd
    raise ValueError(f"Unknown action: {action}")


class AppHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            options = "\n".join([
                f'<label class="language-option" data-label="{html.escape(ALL_FILES_LABEL)}"><input class="language-checkbox" type="checkbox" value="{html.escape(ALL_FILES_LABEL)}"> <span>{html.escape(ALL_FILES_LABEL)}</span></label>'
            ] + [
                f'<label class="language-option" data-label="{html.escape(lang)}"><input class="language-checkbox" type="checkbox" value="{html.escape(lang)}"{(" checked" if lang == "Basaa" else "")}> <span>{html.escape(lang)}</span></label>'
                for lang in LANGUAGES
            ])
            page = (
                INDEX_HTML
                .replace("__BASE_PATH__", html.escape(DEFAULT_BASE_PATH))
                .replace("__LANGUAGE_OPTIONS__", options)
                .replace("__ALL_FILES__", html.escape(ALL_FILES_LABEL))
            )
            self._send_html(page)
            return
        if parsed.path == "/status":
            self._send_json(TASKS.snapshot())
            return
        if parsed.path == "/version":
            self._send_json({"server_version": SERVER_VERSION})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/upload":
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                output_base = form.getfirst("output_base", DEFAULT_BASE_PATH)
                upload_source_dir = form.getfirst("upload_source_dir", "")
                file_items = form["files"] if "files" in form else []
                if not isinstance(file_items, list):
                    file_items = [file_items]
                uploads: list[tuple[str, bytes]] = []
                for item in file_items:
                    if not getattr(item, "filename", ""):
                        continue
                    uploads.append((item.filename, item.file.read()))
                if not uploads:
                    raise ValueError("No files were uploaded.")
                self._send_json({"ok": True, "upload": prepare_uploaded_files(output_base, uploads, upload_source_dir)})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/clear-uploads":
            clear_uploaded_files()
            self._send_json({"ok": True})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b""

        if parsed.path == "/run":
            try:
                payload = json.loads(raw.decode("utf-8"))
                cmd, label, cwd = build_command(payload)
                ok, msg = TASKS.start(cmd, label, cwd)
                self._send_json({"ok": ok, "message": msg}, status=200 if ok else 409)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/stop":
            ok, msg = TASKS.stop()
            self._send_json({"ok": ok, "message": msg}, status=200 if ok else 409)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    threading.Thread(target=watch_self_and_reload, daemon=True).start()
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Phrasebook web app running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


