#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple desktop UI for phrasebook processing tasks.

This wraps the existing CLI scripts so non-technical users can run the
phrasebook workflow without typing commands manually.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSING_SCRIPT = REPO_ROOT / "tools" / "guidedeconversationphrasebook_processing.py"
RAPIDFUZZ_SCRIPT = REPO_ROOT / "tools" / "merge_all_languages_rapidfuzz.py"
DEFAULT_BASE_PATH = r"G:\My Drive\Mbú'ŋwɑ̀'nì"
LANGUAGES = [
    "Basaa", "Chichewa", "DualaDouala", "EweTogo", "Ewondo", "FulfuldeBenin",
    "FulfuldeNigeria", "Ghomala", "Hausa", "Igbo", "Kikongo", "Kinyarwanda",
    "Lingala", "Medumba", "Nufi", "Shupamom", "Swahili", "Tshiluba", "Twi",
    "Wolof", "Yemba", "Yoruba", "Zulu",
]


class PhrasebookApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Phrasebook Processing App")
        self.root.geometry("1100x760")
        self.root.minsize(980, 680)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.current_process: subprocess.Popen[str] | None = None

        self.base_path_var = tk.StringVar(value=DEFAULT_BASE_PATH)
        self.nbsp_language_var = tk.StringVar(value="Basaa")
        self.rapidfuzz_threshold_var = tk.StringVar(value="85")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        top = ttk.Frame(self.root, padding=16)
        top.pack(fill="x")

        title = ttk.Label(top, text="Phrasebook Processing", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(
            top,
            text="Run phrasebook extraction, NBSP fixes, and RapidFuzz merge from a desktop interface.",
        )
        subtitle.pack(anchor="w", pady=(4, 0))

        settings = ttk.LabelFrame(self.root, text="Settings", padding=16)
        settings.pack(fill="x", padx=16, pady=(0, 12))

        ttk.Label(settings, text="Base Path").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.base_path_var, width=95).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0)
        )

        ttk.Label(settings, text="NBSP Test Language").grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Combobox(
            settings,
            textvariable=self.nbsp_language_var,
            values=LANGUAGES,
            state="readonly",
            width=22,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(12, 0))

        ttk.Label(settings, text="RapidFuzz Threshold").grid(row=1, column=2, sticky="w", pady=(12, 0), padx=(16, 0))
        ttk.Entry(settings, textvariable=self.rapidfuzz_threshold_var, width=10).grid(
            row=1, column=3, sticky="w", padx=(8, 0), pady=(12, 0)
        )

        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        actions = ttk.LabelFrame(self.root, text="Actions", padding=16)
        actions.pack(fill="x", padx=16, pady=(0, 12))

        buttons = [
            ("Main Processing", self.run_main_processing),
            ("Main + RapidFuzz Merge", self.run_main_and_rapidfuzz),
            ("RapidFuzz Merge Only", self.run_rapidfuzz_only),
            ("NBSP Test Report", self.run_nbsp_test),
            ("NBSP Test + Fixed Doc Copy", self.run_nbsp_test_with_copy),
            ("Apply NBSP Fixes In Place", self.run_nbsp_inplace),
            ("Stop Current Task", self.stop_process),
        ]

        for idx, (label, command) in enumerate(buttons):
            ttk.Button(actions, text=label, command=command).grid(
                row=idx // 3,
                column=idx % 3,
                sticky="ew",
                padx=6,
                pady=6,
            )

        for col in range(3):
            actions.columnconfigure(col, weight=1)

        status_bar = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        status_bar.pack(fill="x")
        ttk.Label(status_bar, text="Status:").pack(side="left")
        ttk.Label(status_bar, textvariable=self.status_var).pack(side="left", padx=(6, 0))

        log_frame = ttk.LabelFrame(self.root, text="Command Output", padding=8)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.log_text = tk.Text(log_frame, wrap="word", font=("Consolas", 10))
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _poll_log_queue(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
        self.root.after(100, self._poll_log_queue)

    def _validate_threshold(self) -> str | None:
        value = self.rapidfuzz_threshold_var.get().strip()
        try:
            float(value)
        except ValueError:
            messagebox.showerror("Invalid Threshold", "RapidFuzz threshold must be a number.")
            return None
        return value

    def _start_process(self, cmd: list[str], label: str) -> None:
        if self.current_process is not None and self.current_process.poll() is None:
            messagebox.showwarning("Task Running", "A task is already running. Stop it first or wait for it to finish.")
            return

        self._append_log(f"\n=== {label} ===\n")
        self._append_log(" ".join(cmd) + "\n\n")
        self.status_var.set(f"Running: {label}")

        def runner() -> None:
            try:
                env = dict(**__import__("os").environ)
                env["PYTHONIOENCODING"] = "utf-8"
                self.current_process = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                assert self.current_process.stdout is not None
                for line in self.current_process.stdout:
                    self.log_queue.put(line)
                return_code = self.current_process.wait()
                if return_code == 0:
                    self.log_queue.put(f"\nFinished successfully: {label}\n")
                    self.root.after(0, lambda: self.status_var.set("Ready"))
                else:
                    self.log_queue.put(f"\nTask failed with exit code {return_code}: {label}\n")
                    self.root.after(0, lambda: self.status_var.set("Failed"))
            except Exception as exc:
                self.log_queue.put(f"\nFailed to start task: {exc}\n")
                self.root.after(0, lambda: self.status_var.set("Failed"))
            finally:
                self.current_process = None

        threading.Thread(target=runner, daemon=True).start()

    def run_main_processing(self) -> None:
        cmd = [
            sys.executable,
            str(PROCESSING_SCRIPT),
            "--base-path",
            self.base_path_var.get().strip(),
        ]
        self._start_process(cmd, "Main Processing")

    def run_main_and_rapidfuzz(self) -> None:
        threshold = self._validate_threshold()
        if threshold is None:
            return
        cmd = [
            sys.executable,
            str(PROCESSING_SCRIPT),
            "--base-path",
            self.base_path_var.get().strip(),
            "--run-rapidfuzz-merge",
            "--rapidfuzz-threshold",
            threshold,
        ]
        self._start_process(cmd, "Main Processing + RapidFuzz Merge")

    def run_rapidfuzz_only(self) -> None:
        threshold = self._validate_threshold()
        if threshold is None:
            return
        cmd = [
            sys.executable,
            str(RAPIDFUZZ_SCRIPT),
            "--excel",
            "African_Languages_dataframes.xlsx",
            "--base-path",
            self.base_path_var.get().strip(),
            "--master-language",
            "Nufi",
            "--text-column",
            "both",
            "--threshold",
            threshold,
            "--output",
            "African_Languages_dataframes_rapidfuzz_merged.csv",
            "--excel-output",
            "African_Languages_dataframes_rapidfuzz_merged.xlsx",
            "--checkpoint-every-language",
        ]
        self._start_process(cmd, "RapidFuzz Merge Only")

    def run_nbsp_test(self) -> None:
        cmd = [
            sys.executable,
            str(PROCESSING_SCRIPT),
            "--base-path",
            self.base_path_var.get().strip(),
            "--test-nbsp-language",
            self.nbsp_language_var.get(),
        ]
        self._start_process(cmd, "NBSP Test Report")

    def run_nbsp_test_with_copy(self) -> None:
        cmd = [
            sys.executable,
            str(PROCESSING_SCRIPT),
            "--base-path",
            self.base_path_var.get().strip(),
            "--test-nbsp-language",
            self.nbsp_language_var.get(),
            "--create-nbsp-fixed-doc-copy",
        ]
        self._start_process(cmd, "NBSP Test + Fixed Doc Copy")

    def run_nbsp_inplace(self) -> None:
        if not messagebox.askyesno(
            "Confirm In-Place Update",
            "This will update the source Word documents in place. Continue?",
        ):
            return
        cmd = [
            sys.executable,
            str(PROCESSING_SCRIPT),
            "--base-path",
            self.base_path_var.get().strip(),
            "--apply-nbsp-fixes-inplace",
        ]
        self._start_process(cmd, "Apply NBSP Fixes In Place")

    def stop_process(self) -> None:
        if self.current_process is None or self.current_process.poll() is not None:
            messagebox.showinfo("No Running Task", "There is no running task to stop.")
            return
        self.current_process.terminate()
        self.status_var.set("Stopping...")
        self._append_log("\nStop requested.\n")


def main() -> None:
    root = tk.Tk()
    app = PhrasebookApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
