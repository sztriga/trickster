from __future__ import annotations

import json
import queue
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

from trickster.training.eval import (
    evaluate_policies,
    evaluate_policies_parallel,
    evaluate_policy_vs_random,
    evaluate_policy_vs_random_parallel,
)
from trickster.training.model_store import load_slot, save_latest_and_prev, slot_exists
from trickster.training.self_play import train_self_play
from trickster.training.model_spec import ModelSpec, list_model_dirs, model_dir, model_label_from_dir, read_spec, write_spec


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PLOT_BG = "#111315"
PLOT_AXIS = "#9ca3af"
PLOT_GRID = "#262a30"
PLOT_BASELINE = "#6b7280"
PLOT_TEXT = "#e5e7eb"
PLOT_LINE = "#f4a261"
PLOT_POINT = "#fb923c"

ACCENT_BTN_BG = "#ff8a3d"
ACCENT_BTN_BG_HOVER = "#ff9b57"
ACCENT_BTN_BG_ACTIVE = "#ff7a1a"
ACCENT_BTN_FG = "#1f2937"
ACCENT_BTN_DISABLED_BG = "#2b2f36"
ACCENT_BTN_DISABLED_FG = "#9ca3af"

SEG_BG = "#1e2128"
SEG_SEL_BG = ACCENT_BTN_BG
SEG_SEL_FG = ACCENT_BTN_FG
SEG_UNSEL_BG = "#2b2f36"
SEG_UNSEL_FG = "#9ca3af"

PROG_TRACK = "#e5e7eb"
PROG_OUTLINE = "#d1d5db"
PROG_FILL = ACCENT_BTN_BG

PAD = 12  # standard horizontal margin


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------

class AccentButton(tk.Label):
    def __init__(self, master, *, text: str, command, enabled: bool = True) -> None:
        super().__init__(master, text=text, padx=14, pady=6, bd=0, highlightthickness=0)
        self._command = command
        self._enabled = True
        self._hover = False
        self._active = False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.set_enabled(enabled)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._active = self._hover = False
        if self._enabled:
            self.configure(cursor="hand2", bg=ACCENT_BTN_BG, fg=ACCENT_BTN_FG)
        else:
            self.configure(cursor="", bg=ACCENT_BTN_DISABLED_BG, fg=ACCENT_BTN_DISABLED_FG)

    def _refresh(self) -> None:
        if not self._enabled:
            self.configure(bg=ACCENT_BTN_DISABLED_BG, fg=ACCENT_BTN_DISABLED_FG)
        elif self._active:
            self.configure(bg=ACCENT_BTN_BG_ACTIVE, fg=ACCENT_BTN_FG)
        elif self._hover:
            self.configure(bg=ACCENT_BTN_BG_HOVER, fg=ACCENT_BTN_FG)
        else:
            self.configure(bg=ACCENT_BTN_BG, fg=ACCENT_BTN_FG)

    def _on_enter(self, _e) -> None:
        if self._enabled:
            self._hover = True
            self._refresh()

    def _on_leave(self, _e) -> None:
        self._hover = self._active = False
        self._refresh()

    def _on_press(self, _e) -> None:
        if self._enabled:
            self._active = True
            self._refresh()

    def _on_release(self, _e) -> None:
        if not self._enabled:
            return
        was = self._active
        self._active = False
        self._refresh()
        if was and self._hover:
            self._command()


class OrangeProgressBar(tk.Canvas):
    def __init__(self, master, *, height: int = 12) -> None:
        try:
            bg = master.winfo_toplevel().cget("bg")
        except tk.TclError:
            bg = "SystemButtonFace"
        super().__init__(master, height=height, highlightthickness=0, bd=0, bg=bg)
        self.maximum: int = 1
        self.value: int = 0
        self._height = height
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_maximum(self, m: int) -> None:
        self.maximum = max(1, int(m))
        if self.value > self.maximum:
            self.value = self.maximum
        self._redraw()

    def set_value(self, v: int) -> None:
        self.value = max(0, min(int(v), self.maximum))
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w = int(self.winfo_width() or 1)
        h = int(self.winfo_height() or self._height)
        self.create_rectangle(0, 0, w, h, outline=PROG_OUTLINE, fill=PROG_TRACK)
        fw = int(w * (self.value / self.maximum)) if self.maximum > 0 else 0
        if fw > 0:
            self.create_rectangle(0, 0, fw, h, outline="", fill=PROG_FILL)


class SegmentedControl(tk.Frame):
    def __init__(self, master, options: list[str], *, command=None) -> None:
        super().__init__(master, bg=SEG_BG)
        self._var = tk.StringVar(value=options[0])
        self._command = command
        self._labels: list[tuple[str, tk.Label]] = []
        self._enabled = True
        for opt in options:
            lbl = tk.Label(self, text=opt, padx=14, pady=4, bd=0, highlightthickness=0, cursor="hand2")
            lbl.pack(side="left", padx=(0, 1))
            lbl.bind("<Button-1>", lambda _e, o=opt: self._select(o))
            self._labels.append((opt, lbl))
        self._refresh()

    def _select(self, opt: str) -> None:
        if not self._enabled:
            return
        self._var.set(opt)
        self._refresh()
        if self._command:
            self._command(opt)

    def get(self) -> str:
        return self._var.get()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        for _, lbl in self._labels:
            lbl.configure(cursor="hand2" if enabled else "")
        self._refresh()

    def _refresh(self) -> None:
        sel = self._var.get()
        for opt, lbl in self._labels:
            if opt == sel:
                lbl.configure(bg=SEG_SEL_BG, fg=SEG_SEL_FG)
            else:
                lbl.configure(bg=SEG_UNSEL_BG, fg=SEG_UNSEL_FG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field(parent, var, label: str, row: int, col: int, width: int = 8):
    """Label above an Entry, packed via grid. Returns the Entry widget."""
    ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 4))
    ent = ttk.Entry(parent, textvariable=var, width=width)
    ent.grid(row=row + 1, column=col, sticky="w", padx=(0, 10))
    return ent


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class UltiCardApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Trickster")
        self.geometry("1024x640")
        self.minsize(920, 520)

        self._train_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._train_thread: Optional[threading.Thread] = None
        self._eval_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._eval_thread: Optional[threading.Thread] = None
        self._model_label_to_dir: dict[str, Path] = {}

        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True)
        self.train_tab = ttk.Frame(self._nb)
        self.eval_tab = ttk.Frame(self._nb)
        self.models_tab = ttk.Frame(self._nb)
        self._nb.add(self.train_tab, text="Train")
        self._nb.add(self.eval_tab, text="Evaluate")
        self._nb.add(self.models_tab, text="Models")

        self._build_train_tab()
        self._build_eval_tab()
        self._build_models_tab()
        self.after(100, self._poll_train_queue)
        self.after(100, self._poll_eval_queue)
        self._refresh_all_model_dropdowns()

    # ================================================================
    #  TRAIN TAB
    # ================================================================
    def _build_train_tab(self) -> None:
        root = self.train_tab

        # -- Row 1: Architecture + Method side by side --
        row1 = ttk.Frame(root, padding=(PAD, PAD, PAD, 0))
        row1.pack(fill="x")

        arch = ttk.LabelFrame(row1, text="Architecture", padding=(8, 4, 8, 6))
        arch.grid(row=0, column=0, sticky="nw", padx=(0, 8))

        self.seg_arch = SegmentedControl(arch, ["Linear", "MLP"], command=self._on_arch_change)
        self.seg_arch.pack(anchor="w")

        self._mlp_panel = ttk.Frame(arch)
        self.var_hidden_units = tk.IntVar(value=64)
        self.var_hidden_layers = tk.IntVar(value=1)
        self.var_activation = tk.StringVar(value="relu")
        self.ent_hidden = _field(self._mlp_panel, self.var_hidden_units, "Hidden", 0, 0)
        self.ent_layers = _field(self._mlp_panel, self.var_hidden_layers, "Layers", 0, 1, width=4)
        ttk.Label(self._mlp_panel, text="Activation").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.cmb_activation = ttk.Combobox(
            self._mlp_panel, textvariable=self.var_activation,
            values=["relu", "tanh"], width=6, state="readonly",
        )
        self.cmb_activation.grid(row=1, column=2, sticky="w")

        meth = ttk.LabelFrame(row1, text="Training Method", padding=(8, 4, 8, 6))
        meth.grid(row=0, column=1, sticky="nw")

        ttk.Label(meth, text="Direct self-play").pack(anchor="w")

        self._direct_panel = ttk.Frame(meth)
        self.var_eps0 = tk.DoubleVar(value=0.2)
        self.var_eps1 = tk.DoubleVar(value=0.02)
        self.ent_eps0 = _field(self._direct_panel, self.var_eps0, "Eps start", 0, 0)
        self.ent_eps1 = _field(self._direct_panel, self.var_eps1, "Eps end", 0, 1)
        self._direct_panel.pack(anchor="w", pady=(4, 0))

        # -- Row 2: Parameters + Start button --
        row2 = ttk.Frame(root, padding=(PAD, 8, PAD, 0))
        row2.pack(fill="x")

        self.var_episodes = tk.IntVar(value=20000)
        self.var_seed = tk.IntVar(value=0)
        self.var_lr = tk.DoubleVar(value=0.05)
        self.var_l2 = tk.DoubleVar(value=1e-6)

        self.ent_episodes = _field(row2, self.var_episodes, "Episodes", 0, 0)
        self.ent_seed = _field(row2, self.var_seed, "Seed", 0, 1)
        self.ent_lr = _field(row2, self.var_lr, "LR", 0, 2)
        self.ent_l2 = _field(row2, self.var_l2, "L2", 0, 3)

        self.btn_train = AccentButton(row2, text="Start training", command=self._start_training)
        self.btn_train.grid(row=1, column=4, sticky="w", padx=(8, 0))

        # -- Status + progress --
        self.lbl_train_status = ttk.Label(root, text="Idle")
        self.lbl_train_status.pack(anchor="w", padx=PAD, pady=(8, 0))
        self.prog = OrangeProgressBar(root, height=12)
        self.prog.pack(fill="x", padx=PAD, pady=(4, 6))

        # -- Log --
        log = ttk.Frame(root, padding=(PAD, 0, PAD, PAD))
        log.pack(fill="both", expand=True)
        self.txt_train = tk.Text(log, height=8, wrap="word")
        sb = ttk.Scrollbar(log, orient="vertical", command=self.txt_train.yview)
        self.txt_train.configure(yscrollcommand=sb.set)
        self.txt_train.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.txt_train.insert("end", "Tip: For expert iteration training, use the CLI: python scripts/train.py --method expert\n")
        self.txt_train.configure(state="disabled")

    # -- Dynamic panels --

    def _on_arch_change(self, arch: str) -> None:
        if arch == "MLP":
            self._mlp_panel.pack(anchor="w", pady=(4, 0))
            if abs(float(self.var_lr.get()) - 0.05) < 1e-12:
                self.var_lr.set(0.01)
        else:
            self._mlp_panel.pack_forget()
            if abs(float(self.var_lr.get()) - 0.01) < 1e-12:
                self.var_lr.set(0.05)

    def _current_train_spec(self) -> ModelSpec:
        kind = self.seg_arch.get().lower()
        if kind == "mlp":
            return ModelSpec(
                kind="mlp",
                params={
                    "hidden_units": int(self.var_hidden_units.get()),
                    "hidden_layers": int(self.var_hidden_layers.get()),
                    "activation": str(self.var_activation.get()),
                },
                method="direct",
            )
        return ModelSpec(kind="linear", params={}, method="direct")

    def _set_train_controls_enabled(self, enabled: bool) -> None:
        st = "normal" if enabled else "disabled"
        self.seg_arch.set_enabled(enabled)
        for w in (
            self.ent_hidden, self.ent_layers, self.cmb_activation,
            self.ent_eps0, self.ent_eps1,
            self.ent_episodes, self.ent_seed,
            self.ent_lr, self.ent_l2,
        ):
            try:
                w.configure(state=st)
            except tk.TclError:
                pass
        self.btn_train.set_enabled(enabled)

    def _append_train_log(self, msg: str) -> None:
        self.txt_train.configure(state="normal")
        self.txt_train.insert("end", msg + "\n")
        self.txt_train.see("end")
        self.txt_train.configure(state="disabled")

    def _start_training(self) -> None:
        if self._train_thread is not None and self._train_thread.is_alive():
            return
        episodes = int(self.var_episodes.get())
        seed = int(self.var_seed.get())
        lr = float(self.var_lr.get())
        l2 = float(self.var_l2.get())
        spec = self._current_train_spec()
        mdir = model_dir(spec, root="models")
        write_spec(spec, root="models")

        self._set_train_controls_enabled(False)
        self.prog.set_maximum(max(1, episodes))
        self.prog.set_value(0)
        self.lbl_train_status.configure(text="Training...")

        self._append_train_log(
            f"Direct: {mdir.name}  ep={episodes} "
            f"eps={self.var_eps0.get()}->{self.var_eps1.get()} lr={lr} l2={l2}"
        )

        _last_emit = [0]

        def on_progress(done, stats):
            if done < _last_emit[0] + 250 and done < episodes:
                return
            _last_emit[0] = done
            self._train_queue.put(("progress", (done, stats)))

        _spec, _episodes = spec, episodes
        _seed, _lr, _l2, _mdir = seed, lr, l2, mdir

        def worker():
            try:
                policy, stats = train_self_play(
                    spec=_spec, episodes=_episodes,
                    seed=_seed, lr=_lr, l2=_l2,
                    epsilon_start=float(self.var_eps0.get()),
                    epsilon_end=float(self.var_eps1.get()),
                    on_progress=on_progress,
                )
                save_latest_and_prev(policy, models_dir=_mdir)
                info = {"episodes": stats.episodes, "method": "direct", "seed": _seed, "lr": _lr, "l2": _l2}
                (_mdir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
                self._train_queue.put(("done", (_mdir, policy, stats)))
            except Exception as e:
                self._train_queue.put(("error", e))

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()

    def _poll_train_queue(self) -> None:
        try:
            while True:
                kind, payload = self._train_queue.get_nowait()
                if kind == "progress":
                    done, _ = payload
                    self.prog.set_value(done)
                    self.lbl_train_status.configure(text=f"Training... {done}/{self.prog.maximum}")
                elif kind == "done":
                    mdir2, _, stats = payload
                    self.prog.set_value(self.prog.maximum)
                    self.lbl_train_status.configure(text=f"Done. {mdir2}/latest.pkl")
                    self._append_train_log(f"Done: {stats.episodes} episodes")
                    self._set_train_controls_enabled(True)
                    self._refresh_all_model_dropdowns()
                    self._refresh_models_list()
                elif kind == "error":
                    self.lbl_train_status.configure(text=f"Error: {payload!r}")
                    self._append_train_log(f"ERROR: {payload!r}")
                    self._set_train_controls_enabled(True)
        except queue.Empty:
            pass
        self.after(100, self._poll_train_queue)

    # ================================================================
    #  EVALUATE TAB
    # ================================================================
    def _build_eval_tab(self) -> None:
        root = self.eval_tab

        # -- Compare vs random --
        vs_rand = ttk.LabelFrame(root, text="Evaluate vs random", padding=(8, 4, 8, 8))
        vs_rand.pack(fill="x", padx=PAD, pady=(PAD, 0))

        self.vr_var_model = tk.StringVar(value="")
        self.vr_var_games = tk.IntVar(value=2000)
        self.vr_var_seed = tk.IntVar(value=0)
        self.vr_var_workers = tk.IntVar(value=2)

        ttk.Label(vs_rand, text="Model").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.cmb_vr_model = ttk.Combobox(vs_rand, textvariable=self.vr_var_model, width=30, state="readonly")
        self.cmb_vr_model.grid(row=1, column=0, sticky="w", padx=(0, 10))

        self.ent_vr_games = _field(vs_rand, self.vr_var_games, "Games", 0, 1)
        self.ent_vr_seed = _field(vs_rand, self.vr_var_seed, "Seed", 0, 2)
        self.ent_vr_workers = _field(vs_rand, self.vr_var_workers, "Workers", 0, 3, width=5)

        self.btn_vr = AccentButton(vs_rand, text="Run", command=self._start_vs_random)
        self.btn_vr.grid(row=1, column=4, sticky="w", padx=(6, 0))

        self.lbl_vr_result = ttk.Label(vs_rand, text="")
        self.lbl_vr_result.grid(row=2, column=0, columnspan=5, sticky="w", pady=(4, 0))

        # -- Head-to-head compare --
        cmp = ttk.LabelFrame(root, text="Head-to-head compare", padding=(8, 4, 8, 8))
        cmp.pack(fill="x", padx=PAD, pady=(8, 0))

        self.cmp_var_a = tk.StringVar(value="")
        self.cmp_var_b = tk.StringVar(value="")
        self.cmp_var_games = tk.IntVar(value=2000)
        self.cmp_var_seed = tk.IntVar(value=0)
        self.cmp_var_workers = tk.IntVar(value=2)

        ttk.Label(cmp, text="Model A").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.cmb_cmp_a = ttk.Combobox(cmp, textvariable=self.cmp_var_a, width=26, state="readonly")
        self.cmb_cmp_a.grid(row=1, column=0, sticky="w", padx=(0, 6))

        ttk.Label(cmp, text="vs").grid(row=1, column=1, padx=2)

        ttk.Label(cmp, text="Model B").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.cmb_cmp_b = ttk.Combobox(cmp, textvariable=self.cmp_var_b, width=26, state="readonly")
        self.cmb_cmp_b.grid(row=1, column=2, sticky="w", padx=(0, 6))

        self.ent_cmp_games = _field(cmp, self.cmp_var_games, "Games", 0, 3, width=6)
        self.ent_cmp_seed = _field(cmp, self.cmp_var_seed, "Seed", 0, 4, width=6)
        self.ent_cmp_workers = _field(cmp, self.cmp_var_workers, "Workers", 0, 5, width=5)

        self.btn_compare = AccentButton(cmp, text="Compare", command=self._start_compare)
        self.btn_compare.grid(row=1, column=6, sticky="w", padx=(6, 0))

        self.lbl_compare_result = ttk.Label(cmp, text="")
        self.lbl_compare_result.grid(row=2, column=0, columnspan=7, sticky="w", pady=(4, 0))

        # -- Status + progress --
        self.eval_status = ttk.Label(root, text="")
        self.eval_status.pack(anchor="w", padx=PAD, pady=(8, 0))
        self.eval_prog = OrangeProgressBar(root, height=12)
        self.eval_prog.pack(fill="x", padx=PAD, pady=(4, 4))

        # -- Log --
        log = ttk.Frame(root, padding=(PAD, 0, PAD, PAD))
        log.pack(fill="both", expand=True)
        self.txt_eval = tk.Text(log, height=6, wrap="word")
        sb = ttk.Scrollbar(log, orient="vertical", command=self.txt_eval.yview)
        self.txt_eval.configure(yscrollcommand=sb.set)
        self.txt_eval.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.txt_eval.configure(state="disabled")

    # -- Eval helpers --

    def _refresh_all_model_dropdowns(self) -> None:
        dirs = list_model_dirs(root="models")
        self._model_label_to_dir = {model_label_from_dir(d): d for d in dirs}
        labels = sorted(self._model_label_to_dir.keys())

        first = labels[0] if labels else ""
        for cmb, var in [
            (self.cmb_vr_model, self.vr_var_model),
            (self.cmb_cmp_a, self.cmp_var_a),
            (self.cmb_cmp_b, self.cmp_var_b),
        ]:
            cmb["values"] = labels
            if labels and var.get() not in labels:
                var.set(first)

    def _set_eval_controls_enabled(self, enabled: bool) -> None:
        st = "normal" if enabled else "disabled"
        for w in (
            self.ent_vr_games, self.ent_vr_seed, self.ent_vr_workers,
            self.ent_cmp_games, self.ent_cmp_seed, self.ent_cmp_workers,
        ):
            w.configure(state=st)
        for cmb in (self.cmb_vr_model, self.cmb_cmp_a, self.cmb_cmp_b):
            cmb.configure(state="readonly" if enabled else "disabled")
        self.btn_vr.set_enabled(enabled)
        self.btn_compare.set_enabled(enabled)

    def _append_eval_log(self, msg: str) -> None:
        self.txt_eval.configure(state="normal")
        self.txt_eval.insert("end", msg + "\n")
        self.txt_eval.see("end")
        self.txt_eval.configure(state="disabled")

    # -- Vs random --

    def _start_vs_random(self) -> None:
        if self._eval_thread is not None and self._eval_thread.is_alive():
            return
        mdir = self._model_label_to_dir.get(self.vr_var_model.get())
        if not mdir:
            self.lbl_vr_result.configure(text="Select a model.")
            return
        if not slot_exists("latest", models_dir=mdir):
            self.lbl_vr_result.configure(text="No latest.pkl found.")
            return

        games = int(self.vr_var_games.get())
        seed = int(self.vr_var_seed.get())
        workers = max(1, int(self.vr_var_workers.get()))
        label = self.vr_var_model.get()

        self._set_eval_controls_enabled(False)
        self.lbl_vr_result.configure(text="")
        self.eval_status.configure(text=f"Evaluating {label} vs random...")
        self.eval_prog.set_maximum(1)
        self.eval_prog.set_value(0)

        def worker():
            try:
                pol = load_slot("latest", models_dir=mdir)
                if workers <= 1:
                    st = evaluate_policy_vs_random(pol, games=games, seed=seed)
                else:
                    st = evaluate_policy_vs_random_parallel(pol, games=games, seed=seed, workers=workers)
                self._eval_queue.put(("vr_done", (label, st)))
            except Exception as e:
                self._eval_queue.put(("eval_error", e))

        self._eval_thread = threading.Thread(target=worker, daemon=True)
        self._eval_thread.start()

    # -- Head-to-head compare --

    def _start_compare(self) -> None:
        if self._eval_thread is not None and self._eval_thread.is_alive():
            return
        a_dir = self._model_label_to_dir.get(self.cmp_var_a.get())
        b_dir = self._model_label_to_dir.get(self.cmp_var_b.get())
        if not a_dir or not b_dir:
            self.lbl_compare_result.configure(text="Select two models.")
            return
        if not slot_exists("latest", models_dir=a_dir) or not slot_exists("latest", models_dir=b_dir):
            self.lbl_compare_result.configure(text="One or both models missing latest.pkl.")
            return

        games = int(self.cmp_var_games.get())
        seed = int(self.cmp_var_seed.get())
        workers = max(1, int(self.cmp_var_workers.get()))
        a_label, b_label = self.cmp_var_a.get(), self.cmp_var_b.get()

        self._set_eval_controls_enabled(False)
        self.lbl_compare_result.configure(text="")
        self.eval_status.configure(text=f"Comparing A vs B...")
        self.eval_prog.set_maximum(1)
        self.eval_prog.set_value(0)

        def worker():
            try:
                pa = load_slot("latest", models_dir=a_dir)
                pb = load_slot("latest", models_dir=b_dir)
                if workers <= 1:
                    st = evaluate_policies(pa, pb, games=games, seed=seed)
                else:
                    st = evaluate_policies_parallel(pa, pb, games=games, seed=seed, workers=workers)
                self._eval_queue.put(("compare_done", (a_label, b_label, st)))
            except Exception as e:
                self._eval_queue.put(("eval_error", e))

        self._eval_thread = threading.Thread(target=worker, daemon=True)
        self._eval_thread.start()

    # -- Queue polling --

    def _poll_eval_queue(self) -> None:
        try:
            while True:
                kind, payload = self._eval_queue.get_nowait()
                if kind == "vr_done":
                    label, stats = payload
                    self.lbl_vr_result.configure(
                        text=f"Model {stats.a_points} pts vs Random {stats.b_points} pts  "
                             f"({stats.a_ppd:.2f} vs {stats.b_ppd:.2f} per deal, {stats.deals} deals)"
                    )
                    self._append_eval_log(
                        f"{label}: model={stats.a_points} random={stats.b_points} ppd={stats.a_ppd:.2f} deals={stats.deals}"
                    )
                    self.eval_prog.set_value(1)
                    self.eval_status.configure(text="Done.")
                    self._set_eval_controls_enabled(True)
                elif kind == "compare_done":
                    a_label, b_label, stats = payload
                    self.lbl_compare_result.configure(
                        text=f"A {stats.a_points} pts ({stats.a_ppd:.2f}/deal) | "
                             f"B {stats.b_points} pts ({stats.b_ppd:.2f}/deal)  "
                             f"({stats.deals} deals)"
                    )
                    self._append_eval_log(
                        f"Compare: {a_label} vs {b_label}  A={stats.a_points} B={stats.b_points} deals={stats.deals}"
                    )
                    self.eval_prog.set_value(1)
                    self.eval_status.configure(text="Compare done.")
                    self._set_eval_controls_enabled(True)
                elif kind == "eval_error":
                    self.eval_status.configure(text=f"Error: {payload!r}")
                    self._append_eval_log(f"ERROR: {payload!r}")
                    self._set_eval_controls_enabled(True)
        except queue.Empty:
            pass
        self.after(100, self._poll_eval_queue)

    # ================================================================
    #  MODELS TAB
    # ================================================================
    def _build_models_tab(self) -> None:
        root = self.models_tab

        # -- Top: list + details side by side --
        top = ttk.Frame(root, padding=(PAD, PAD, PAD, 0))
        top.pack(fill="both", expand=True)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(0, weight=1)

        # Listbox with scrollbar
        list_frame = ttk.Frame(top)
        list_frame.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        self.mdl_listbox = tk.Listbox(list_frame, width=30, activestyle="none", exportselection=False)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.mdl_listbox.yview)
        self.mdl_listbox.configure(yscrollcommand=sb.set)
        self.mdl_listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.mdl_listbox.bind("<<ListboxSelect>>", lambda _e: self._on_model_select())

        # Details panel
        detail = ttk.LabelFrame(top, text="Details", padding=(8, 4, 8, 8))
        detail.grid(row=0, column=1, sticky="nsew")

        self.mdl_detail_text = tk.Text(detail, height=8, wrap="word", state="disabled")
        self.mdl_detail_text.pack(fill="both", expand=True)

        # -- Bottom: action buttons --
        btn_bar = ttk.Frame(root, padding=(PAD, 8, PAD, PAD))
        btn_bar.pack(fill="x")

        self.btn_mdl_rename = AccentButton(btn_bar, text="Rename", command=self._rename_model)
        self.btn_mdl_rename.pack(side="left", padx=(0, 6))
        self.btn_mdl_delete = AccentButton(btn_bar, text="Delete", command=self._delete_model)
        self.btn_mdl_delete.pack(side="left", padx=(0, 6))
        self.btn_mdl_refresh = AccentButton(btn_bar, text="Refresh", command=self._refresh_models_list)
        self.btn_mdl_refresh.pack(side="left")

        self.lbl_mdl_status = ttk.Label(btn_bar, text="")
        self.lbl_mdl_status.pack(side="right")

        self._mdl_dirs: list[Path] = []
        self._refresh_models_list()

    def _refresh_models_list(self) -> None:
        self._mdl_dirs = list_model_dirs(root="models")
        self.mdl_listbox.delete(0, "end")
        for d in self._mdl_dirs:
            self.mdl_listbox.insert("end", model_label_from_dir(d))
        self._set_mdl_detail("")
        self.lbl_mdl_status.configure(text=f"{len(self._mdl_dirs)} model(s)")

    def _selected_model_idx(self) -> Optional[int]:
        sel = self.mdl_listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._mdl_dirs):
            return None
        return idx

    def _set_mdl_detail(self, text: str) -> None:
        self.mdl_detail_text.configure(state="normal")
        self.mdl_detail_text.delete("1.0", "end")
        if text:
            self.mdl_detail_text.insert("end", text)
        self.mdl_detail_text.configure(state="disabled")

    def _on_model_select(self) -> None:
        idx = self._selected_model_idx()
        if idx is None:
            self._set_mdl_detail("")
            return
        d = self._mdl_dirs[idx]
        lines: list[str] = [f"Directory: {d.name}"]
        try:
            spec = read_spec(d / "spec.json")
            lines.append(f"Kind: {spec.kind}")
            lines.append(f"Method: {spec.method}")
            lines.append(f"Game: {spec.game}")
            if spec.params:
                for k, v in spec.params.items():
                    lines.append(f"  {k}: {v}")
        except Exception:
            lines.append("(could not read spec.json)")

        # Training info
        train_info_path = d / "train_info.json"
        if train_info_path.exists():
            try:
                ti = json.loads(train_info_path.read_text(encoding="utf-8"))
                lines.append(f"Episodes: {ti.get('episodes', '?')}")
                if "lr" in ti:
                    lines.append(f"LR: {ti['lr']}  L2: {ti.get('l2', '?')}")
                if "seed" in ti:
                    lines.append(f"Seed: {ti['seed']}")
            except Exception:
                pass

        has_latest = (d / "latest.pkl").exists()
        lines.append(f"latest.pkl: {'yes' if has_latest else 'no'}")

        # Disk size
        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        if total > 1_048_576:
            lines.append(f"Size: {total / 1_048_576:.1f} MB")
        else:
            lines.append(f"Size: {total / 1024:.0f} KB")

        self._set_mdl_detail("\n".join(lines))

    def _rename_model(self) -> None:
        idx = self._selected_model_idx()
        if idx is None:
            self.lbl_mdl_status.configure(text="Select a model first.")
            return
        d = self._mdl_dirs[idx]

        dlg = tk.Toplevel(self)
        dlg.title("Rename model")
        dlg.geometry("360x120")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="New directory name:").pack(padx=12, pady=(12, 4), anchor="w")
        var = tk.StringVar(value=d.name)
        ent = ttk.Entry(dlg, textvariable=var, width=40)
        ent.pack(padx=12, fill="x")
        ent.select_range(0, "end")
        ent.focus_set()

        def do_rename():
            new_name = var.get().strip()
            if not new_name or new_name == d.name:
                dlg.destroy()
                return
            new_path = d.parent / new_name
            if new_path.exists():
                self.lbl_mdl_status.configure(text=f"'{new_name}' already exists.")
                dlg.destroy()
                return
            try:
                d.rename(new_path)
                self.lbl_mdl_status.configure(text=f"Renamed to {new_name}")
                self._refresh_models_list()
                self._refresh_all_model_dropdowns()
            except Exception as e:
                self.lbl_mdl_status.configure(text=f"Rename failed: {e}")
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=(8, 0))
        ttk.Button(btn_frame, text="Rename", command=do_rename).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side="left")
        dlg.bind("<Return>", lambda _e: do_rename())
        dlg.bind("<Escape>", lambda _e: dlg.destroy())

    def _delete_model(self) -> None:
        idx = self._selected_model_idx()
        if idx is None:
            self.lbl_mdl_status.configure(text="Select a model first.")
            return
        d = self._mdl_dirs[idx]

        dlg = tk.Toplevel(self)
        dlg.title("Delete model")
        dlg.geometry("360x110")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text=f"Delete '{d.name}' and all its data?").pack(padx=12, pady=(16, 4))
        ttk.Label(dlg, text="This cannot be undone.", foreground="red").pack(padx=12)

        def do_delete():
            try:
                shutil.rmtree(d)
                self.lbl_mdl_status.configure(text=f"Deleted {d.name}")
                self._refresh_models_list()
                self._refresh_all_model_dropdowns()
            except Exception as e:
                self.lbl_mdl_status.configure(text=f"Delete failed: {e}")
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=(8, 0))
        ttk.Button(btn_frame, text="Delete", command=do_delete).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side="left")
        dlg.bind("<Escape>", lambda _e: dlg.destroy())


def main() -> int:
    app = UltiCardApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
