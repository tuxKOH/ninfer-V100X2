#!/usr/bin/env python3
"""PyQt control panel for the NInfer HTTP server.

The GUI only owns the child ``ninfer-serve`` process.  It does not alter CUDA, driver,
IOMMU, or desktop services.  All values are forwarded as explicit command-line options,
and the generated command is shown before/while the server is running.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import queue
import shlex
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path


def _help() -> None:
    print(
        "usage: ninfer_gui.py [--repo PATH] [--binary PATH] [--artifact PATH]\n"
        "\n"
        "Open a PyQt control panel for ninfer-serve. The GUI starts a local child process;\n"
        "it does not configure remote hosts or system services.\n"
        "\n"
        "Environment overrides: NINFER_GUI_REPO, NINFER_GUI_BINARY, NINFER_GUI_ARTIFACT."
    )


try:  # PyQt6 is preferred; PyQt5 is a supported fallback for customer images.
    from PyQt6.QtCore import QProcess, QSettings, QTimer, QUrl
    from PyQt6.QtGui import QDesktopServices
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    QT6 = True
except ImportError:  # selected by the customer's installed Qt binding.
    from PyQt5.QtCore import QProcess, QSettings, QTimer, QUrl
    from PyQt5.QtGui import QDesktopServices
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    QT6 = False


ROOT = Path(__file__).resolve().parents[2]


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


class NInferGui(QWidget):
    def __init__(self, repo: Path, binary: str = "", artifact: str = "", *,
                 settings: QSettings | None = None) -> None:
        super().__init__()
        self.repo = repo
        self.binary_override = binary or os.environ.get("NINFER_GUI_BINARY", "")
        self.artifact_override = artifact or os.environ.get("NINFER_GUI_ARTIFACT", "")
        self.process: QProcess | None = None
        self.health_in_flight = False
        self.health_results: queue.Queue[tuple[str, bool, str]] = queue.Queue()
        self.settings = settings or QSettings("NInfer", "V100X2Server")
        self.output_decoder = codecs.getincrementaldecoder("utf-8")("replace")

        self.setWindowTitle("NInfer V100X2 API 控制台")
        self.resize(1080, 760)
        self._build_widgets()
        self._load_settings()
        self._refresh_command()

        self.health_timer = QTimer(self)
        self.health_timer.setInterval(2000)
        self.health_timer.timeout.connect(self._poll_health)
        self.health_timer.start()
        self.queue_timer = QTimer(self)
        self.queue_timer.setInterval(100)
        self.queue_timer.timeout.connect(self._drain_health_results)
        self.queue_timer.start()

    def _build_widgets(self) -> None:
        self.binary = QLineEdit()
        self.artifact = QLineEdit()
        self.host = QLineEdit("127.0.0.1")
        self.port = self._spin(6200, 1, 65535, 1)
        self.tp = QComboBox()
        self.tp.addItems(["4", "2", "1"])
        self.devices = QLineEdit("0,1,2,3")
        self.max_context = self._spin(180000, 128, 1048576, 128)
        self.kv_capacity = QLineEdit("180000")
        self.prefill_chunk = self._spin(2560, 128, 1048576, 128)
        self.kv_dtype = QComboBox()
        self.kv_dtype.addItems(["int8", "bf16"])
        self.spec = QComboBox()
        self.spec.addItems(["mtp", "none", "dflash"])
        self.draft_tokens = self._spin(3, 0, 15, 1)
        self.lm_head_draft = QCheckBox("使用优化草稿头")
        self.lm_head_draft.setChecked(True)
        self.cuda_graph = QCheckBox("启用 CUDA Graph")
        self.cuda_graph.setChecked(True)
        self.prefix_reuse = QCheckBox("启用前缀缓存")
        self.prefix_reuse.setChecked(True)
        self.max_concurrency = self._spin(1, 1, 8, 1)
        self.default_max_tokens = self._spin(65536, 1, 1048576, 1024)
        self.pending_timeout_ms = self._spin(0, 0, 86400000, 1000)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password if QT6 else QLineEdit.Password)
        self.temperature = QLineEdit()
        self.top_p = QLineEdit()
        self.top_k = QLineEdit()
        self.min_p = QLineEdit()
        self.presence_penalty = QLineEdit()
        self.frequency_penalty = QLineEdit()
        self.seed = QLineEdit()
        for field in (self.temperature, self.top_p, self.top_k, self.min_p,
                      self.presence_penalty, self.frequency_penalty, self.seed):
            field.setPlaceholderText("留空 = 模型默认 / 由请求设置")
        self.no_thinking = QCheckBox("默认关闭 thinking")
        self.preserve_thinking = QCheckBox("保留 thinking")
        self.greedy = QCheckBox("强制 greedy")
        self.cors = QCheckBox("允许 CORS")

        self.profile = QComboBox()
        self.profile.addItems(
            [
                "TP4 · NVFP4 · MTP3 · 180K",
                "TP4 · Native FP8 · MTP3 · 180K",
                "TP2 · NVFP4 · DFlash7 · 98K",
                "TP2 · NVFP4 · MTP3 · 180K",
                "TP2 · Q4_K_M · MTP3 · 180K",
                "自定义",
            ]
        )
        self.profile.currentIndexChanged.connect(self._apply_profile)

        self.status = QLabel("未启动")
        self.status.setStyleSheet("color: #777; font-weight: bold")
        self.command = QLineEdit()
        self.command.setReadOnly(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)

        browse_binary = QPushButton("浏览")
        browse_artifact = QPushButton("浏览")
        browse_binary.clicked.connect(lambda: self._browse(self.binary, executable=True))
        browse_artifact.clicked.connect(lambda: self._browse(self.artifact, executable=False))

        paths = QFormLayout()
        paths.addRow("ninfer-serve", self._with_button(self.binary, browse_binary))
        paths.addRow("模型 artifact", self._with_button(self.artifact, browse_artifact))

        network = QFormLayout()
        network.addRow("Host", self.host)
        network.addRow("Port", self.port)
        network.addRow("API key（可选）", self.api_key)

        runtime = QFormLayout()
        runtime.addRow("TP", self.tp)
        runtime.addRow("设备 IDs", self.devices)
        runtime.addRow("最大上下文", self.max_context)
        runtime.addRow("KV 容量（数字/auto）", self.kv_capacity)
        runtime.addRow("Prefill chunk", self.prefill_chunk)
        runtime.addRow("KV dtype", self.kv_dtype)
        runtime.addRow("推理后端", self.spec)
        runtime.addRow("草稿 token 数", self.draft_tokens)
        runtime.addRow("草稿头", self.lm_head_draft)
        runtime.addRow("CUDA Graph", self.cuda_graph)
        runtime.addRow("前缀缓存", self.prefix_reuse)

        service = QFormLayout()
        service.addRow("最大并发请求", self.max_concurrency)
        service.addRow("默认最大输出", self.default_max_tokens)
        service.addRow("排队超时 ms（0 = 不限）", self.pending_timeout_ms)
        service.addRow("其它开关", self._checks())

        sampling = QFormLayout()
        for label, field in (("Temperature", self.temperature), ("Top P", self.top_p),
                             ("Top K", self.top_k), ("Min P", self.min_p),
                             ("Presence penalty", self.presence_penalty),
                             ("Frequency penalty", self.frequency_penalty), ("Seed", self.seed)):
            sampling.addRow(label, field)

        left = QVBoxLayout()
        left.addWidget(self._group("路径", paths))
        left.addWidget(self._group("网络", network))
        left.addWidget(self._group("推理参数", runtime))
        left.addWidget(self._group("服务与高级参数", service))
        left.addWidget(self._group("采样默认值（请求参数可覆盖）", sampling))
        left.addStretch(1)

        profile_bar = QHBoxLayout()
        profile_bar.addWidget(QLabel("预设"))
        profile_bar.addWidget(self.profile, 1)
        profile_bar.addWidget(self.status)
        profile_bar.addStretch(1)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("启动 API")
        self.stop_button = QPushButton("停止 API")
        self.stop_button.setEnabled(False)
        self.health_button = QPushButton("检查 health")
        self.open_button = QPushButton("打开 API 地址")
        self.copy_button = QPushButton("复制命令")
        self.start_button.clicked.connect(self.start_server)
        self.stop_button.clicked.connect(self.stop_server)
        self.health_button.clicked.connect(lambda: self._check_health(force=True))
        self.open_button.clicked.connect(self.open_api)
        self.copy_button.clicked.connect(self.copy_command)
        for button in (self.start_button, self.stop_button, self.health_button,
                       self.open_button, self.copy_button):
            buttons.addWidget(button)
        buttons.addStretch(1)

        right = QVBoxLayout()
        right.addLayout(profile_bar)
        right.addWidget(QLabel("启动命令"))
        right.addWidget(self.command)
        right.addLayout(buttons)
        right.addWidget(QLabel("服务日志"))
        right.addWidget(self.log, 1)

        root = QGridLayout(self)
        controls = QWidget()
        controls.setLayout(left)
        self.controls = controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(380)
        root.addWidget(scroll, 0, 0)
        root.addLayout(right, 0, 1)
        root.setColumnStretch(0, 0)
        root.setColumnStretch(1, 1)
        root.setRowStretch(0, 1)
        for widget in self.findChildren(QLineEdit):
            if widget is not self.command:
                widget.textChanged.connect(self._refresh_command)
        for widget in self.findChildren(QSpinBox):
            widget.valueChanged.connect(self._refresh_command)
        for widget in (self.tp, self.spec, self.kv_dtype):
            widget.currentTextChanged.connect(self._refresh_command)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self._refresh_command)
        self.spec.currentTextChanged.connect(self._spec_changed)

    @staticmethod
    def _spin(value: int, minimum: int, maximum: int, step: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    @staticmethod
    def _double_spin(value: float, minimum: float, maximum: float, step: float):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(1)
        widget.setValue(value)
        return widget

    @staticmethod
    def _with_button(widget: QWidget, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget, 1)
        layout.addWidget(button)
        return row

    def _checks(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for check in (self.no_thinking, self.preserve_thinking, self.greedy, self.cors):
            layout.addWidget(check)
        return row

    @staticmethod
    def _group(title: str, layout) -> QGroupBox:
        group = QGroupBox(title)
        group.setLayout(layout)
        return group

    def _browse(self, target: QLineEdit, executable: bool) -> None:
        if executable:
            path, _ = QFileDialog.getOpenFileName(self, "选择 ninfer-serve", str(self.repo))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择模型 artifact", str(self.repo), "NInfer artifact (*.ninfer);;All files (*)"
            )
        if path:
            target.setText(path)

    def _path_defaults(self) -> tuple[Path, Path, Path]:
        model_dirs = [
            self.repo / "models",
            Path("/Models/ninfer-V100X2"),
            Path.home() / "Models/ninfer",
            Path("/Models/ninfer"),
        ]
        if self.binary_override:
            binary = Path(self.binary_override).expanduser()
        else:
            four_cards = any((directory / "qwen3_8_27b_fp8.ninfer").is_file()
                             for directory in model_dirs)
            builds = ["build-v100-tp4", "build-v100"] if four_cards else ["build-v100", "build-v100-tp4"]
            binary = _first_existing(
                [
                    *(self.repo / f"{build}/apps/ninfer-serve" for build in builds),
                    self.repo / "build/apps/ninfer-serve",
                ]
            )
        artifact = Path(self.artifact_override).expanduser() if self.artifact_override else _first_existing(
            [directory / "qwen3_8_27b_nvfp4.ninfer" for directory in model_dirs])
        fp8 = _first_existing([directory / "qwen3_8_27b_fp8.ninfer" for directory in model_dirs])
        return binary, artifact, fp8

    def _apply_profile(self, index: int) -> None:
        if index == 5:
            return
        binary, nvfp4, fp8 = self._path_defaults()
        self.binary.setText(str(binary))
        if index == 0:  # TP4 NVFP4 MTP3
            self.artifact.setText(str(nvfp4))
            self.tp.setCurrentText("4")
            self.devices.setText("0,1,2,3")
            self.max_context.setValue(180000)
            self.kv_capacity.setText("180000")
            self.prefill_chunk.setValue(2560)
            self.kv_dtype.setCurrentText("int8")
            self.spec.setCurrentText("mtp")
            self.draft_tokens.setValue(3)
            self.lm_head_draft.setChecked(True)
        elif index == 1:  # TP4 native FP8 MTP3
            self.artifact.setText(str(fp8))
            self.tp.setCurrentText("4")
            self.devices.setText("0,1,2,3")
            self.max_context.setValue(180000)
            self.kv_capacity.setText("180000")
            self.prefill_chunk.setValue(2560)
            self.kv_dtype.setCurrentText("int8")
            self.spec.setCurrentText("mtp")
            self.draft_tokens.setValue(3)
            self.lm_head_draft.setChecked(True)
        elif index == 2:  # TP2 DFlash7
            self.artifact.setText(str(nvfp4))
            self.tp.setCurrentText("2")
            self.devices.setText("0,1")
            self.max_context.setValue(98304)
            self.kv_capacity.setText("98304")
            self.prefill_chunk.setValue(1024)
            self.kv_dtype.setCurrentText("int8")
            self.spec.setCurrentText("dflash")
            self.draft_tokens.setValue(7)
            self.lm_head_draft.setChecked(False)
        elif index in (3, 4):
            self.artifact.setText(str(nvfp4 if index == 3 else nvfp4.parent / "qwen3_8_27b_q4_k_m.ninfer"))
            self.tp.setCurrentText("2")
            self.devices.setText("0,1")
            self.max_context.setValue(180000)
            self.kv_capacity.setText("180000")
            self.prefill_chunk.setValue(2560)
            self.kv_dtype.setCurrentText("int8")
            self.spec.setCurrentText("mtp")
            self.draft_tokens.setValue(3)
            self.lm_head_draft.setChecked(True)

    def _load_settings(self) -> None:
        binary, artifact, _ = self._path_defaults()
        fresh = not self.settings.contains("tp")
        if fresh:
            # Prefer the verified four-card profile whenever its local artifact is present.
            # A fresh two-card checkout falls back to the TP2 MTP preset.
            has_nvfp4 = any((directory / "qwen3_8_27b_nvfp4.ninfer").is_file()
                            for directory in (self.repo / "models", Path("/Models/ninfer-V100X2"),
                                               Path.home() / "Models/ninfer", Path("/Models/ninfer")))
            has_fp8 = any((directory / "qwen3_8_27b_fp8.ninfer").is_file()
                          for directory in (self.repo / "models", Path("/Models/ninfer-V100X2"),
                                             Path.home() / "Models/ninfer", Path("/Models/ninfer")))
            default_profile = 1 if has_fp8 else (0 if has_nvfp4 else 3)
            self._apply_profile(default_profile)
        self.binary.setText(str(self.binary_override or self.settings.value("binary", str(binary))))
        self.artifact.setText(str(self.artifact_override or self.settings.value("artifact", str(artifact))))
        for name, widget in (("host", self.host), ("devices", self.devices),
                             ("kv_capacity", self.kv_capacity), *self._sampling_fields()):
            value = self.settings.value(name)
            if value is not None:
                widget.setText(str(value))
        for name, widget in (("port", self.port), ("max_context", self.max_context),
                             ("prefill_chunk", self.prefill_chunk), ("draft_tokens", self.draft_tokens),
                             ("max_concurrency", self.max_concurrency),
                             ("default_max_tokens", self.default_max_tokens),
                             ("pending_timeout_ms", self.pending_timeout_ms)):
            value = self.settings.value(name)
            if value is not None:
                widget.setValue(int(value))
        for name, widget in (("tp", self.tp), ("kv_dtype", self.kv_dtype), ("spec", self.spec)):
            value = self.settings.value(name)
            if value is not None:
                widget.setCurrentText(str(value))
        for name, widget in (("lm_head_draft", self.lm_head_draft), ("cuda_graph", self.cuda_graph),
                             ("prefix_reuse", self.prefix_reuse), ("no_thinking", self.no_thinking),
                             ("preserve_thinking", self.preserve_thinking), ("greedy", self.greedy),
                             ("cors", self.cors)):
            value = self.settings.value(name)
            if value is not None:
                widget.setChecked(str(value).lower() in {"1", "true", "yes"})
        self.profile.setCurrentIndex(default_profile if fresh else 5)

    def _save_settings(self) -> None:
        for name, widget in (("binary", self.binary), ("artifact", self.artifact), ("host", self.host),
                             ("devices", self.devices), ("kv_capacity", self.kv_capacity),
                             *self._sampling_fields()):
            self.settings.setValue(name, widget.text())
        for name, widget in (("port", self.port), ("max_context", self.max_context),
                             ("prefill_chunk", self.prefill_chunk), ("draft_tokens", self.draft_tokens),
                             ("max_concurrency", self.max_concurrency),
                             ("default_max_tokens", self.default_max_tokens),
                             ("pending_timeout_ms", self.pending_timeout_ms)):
            self.settings.setValue(name, widget.value())
        self.settings.remove("api_key")
        for name, widget in (("tp", self.tp), ("kv_dtype", self.kv_dtype), ("spec", self.spec)):
            self.settings.setValue(name, widget.currentText())
        for name, widget in (("lm_head_draft", self.lm_head_draft), ("cuda_graph", self.cuda_graph),
                             ("prefix_reuse", self.prefix_reuse), ("no_thinking", self.no_thinking),
                             ("preserve_thinking", self.preserve_thinking), ("greedy", self.greedy),
                             ("cors", self.cors)):
            self.settings.setValue(name, widget.isChecked())

    def _validate(self) -> str | None:
        artifact = Path(self.artifact.text()).expanduser()
        binary = Path(self.binary.text()).expanduser()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            return f"找不到可执行 ninfer-serve：{binary}"
        if not artifact.is_file():
            return f"找不到模型 artifact：{artifact}"
        devices = [item.strip() for item in self.devices.text().split(",") if item.strip()]
        tp = int(self.tp.currentText())
        if len(devices) != tp or len(set(devices)) != len(devices) or not all(x.isdecimal() for x in devices):
            return f"TP{tp} 必须填写 {tp} 个不同的设备 ID"
        spec = self.spec.currentText()
        if tp == 4 and spec == "dflash":
            return "当前产品契约不支持 TP4 DFlash；请使用 TP2 DFlash7 预设"
        if spec == "none" and self.lm_head_draft.isChecked():
            return "优化草稿头只能与 MTP/DFlash 一起使用"
        if spec == "mtp" and not 1 <= self.draft_tokens.value() <= 5:
            return "MTP 最大草稿数必须为 1..5（实际接受 0 个仍有效）"
        if spec == "dflash" and not 1 <= self.draft_tokens.value() <= 7:
            return "27B DFlash 草稿数必须为 1..7"
        if not self.host.text().strip():
            return "Host 不能为空"
        if self.prefill_chunk.value() % 128:
            return "Prefill chunk 必须为 128 的正整数倍"
        capacity = self.kv_capacity.text().strip()
        if capacity and capacity != "auto" and (not capacity.isdecimal() or int(capacity) < self.max_context.value()):
            return "KV 容量请填 auto，或不小于最大上下文的整数；留空则跟随最大上下文"
        if artifact.name == "qwen3_8_27b_fp8.ninfer" and tp != 4:
            return "原生 FP8 产物要求 TP4；请使用四卡 FP8 预设"
        import math
        for name, field in self._sampling_fields():
            if not field.text().strip():
                continue
            try:
                value = float(field.text())
                lo, hi = {"temperature": (0, 2), "top_p": (0, 1), "min_p": (0, 1),
                          "presence_penalty": (-2, 2), "frequency_penalty": (-2, 2),
                          "top_k": (0, 2147483647), "seed": (0, 18446744073709551615)}[name]
                if not math.isfinite(value) or not lo <= value <= hi:
                    raise ValueError()
                if name in {"top_k", "seed"} and not field.text().strip().isdecimal():
                    raise ValueError()
            except ValueError:
                return f"采样参数 {name} 的值或范围无效"
        return None

    def _args(self) -> list[str]:
        args = [str(Path(self.artifact.text().strip()).expanduser()), "--host", self.host.text().strip(),
                "--port", str(self.port.value()), "--tp", self.tp.currentText(),
                "--devices", self.devices.text().strip(), "--max-context", str(self.max_context.value()),
                "--kv-dtype", self.kv_dtype.currentText(),
                "--prefill-chunk", str(self.prefill_chunk.value()), "--max-concurrency",
                str(self.max_concurrency.value()), "--default-max-tokens", str(self.default_max_tokens.value()),
                "--pending-timeout-ms", str(self.pending_timeout_ms.value())]
        spec = self.spec.currentText()
        if self.kv_capacity.text().strip():
            args += ["--kv-capacity", self.kv_capacity.text().strip()]
        if spec != "none":
            args += ["--spec", spec, "--draft-tokens", str(self.draft_tokens.value())]
        if spec != "none" and self.lm_head_draft.isChecked():
            args.append("--lm-head-draft")
        if not self.cuda_graph.isChecked():
            args.append("--no-cuda-graph")
        if not self.prefix_reuse.isChecked():
            args.append("--no-prefix-reuse")
        if self.api_key.text():
            args += ["--api-key", self.api_key.text()]
        if self.no_thinking.isChecked():
            args.append("--no-thinking")
        if self.preserve_thinking.isChecked():
            args.append("--preserve-thinking")
        if self.greedy.isChecked():
            args.append("--greedy")
        if self.cors.isChecked():
            args.append("--cors")
        for name, field in self._sampling_fields():
            if field.text().strip():
                args += ["--" + name.replace("_", "-"), field.text().strip()]
        return args

    def _command(self) -> list[str]:
        binary = Path(self.binary.text()).expanduser()
        return [str(binary), *self._args()]

    def _set_command(self, command: list[str]) -> None:
        redacted = list(command)
        if "--api-key" in redacted:
            redacted[redacted.index("--api-key") + 1] = "<API_KEY>"
        self.command.setText(shlex.join(redacted))

    def _sampling_fields(self):
        return [(name, getattr(self, name)) for name in
                ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed")]

    def _refresh_command(self, *_args) -> None:
        self._set_command(self._command())

    def _spec_changed(self, spec: str) -> None:
        self.draft_tokens.setEnabled(spec != "none")
        self.lm_head_draft.setEnabled(spec != "none")
        if spec == "none" or spec == "dflash":
            self.lm_head_draft.setChecked(False)

    def start_server(self) -> None:
        error = self._validate()
        if error:
            QMessageBox.warning(self, "参数检查失败", error)
            return
        command = self._command()
        self._save_settings()
        self._set_command(command)
        if self.process is not None:
            return
        self.log.clear()
        self._append_log("$ " + self.command.text())
        self.output_decoder.reset()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repo))
        mode = QProcess.ProcessChannelMode.MergedChannels if QT6 else QProcess.MergedChannels
        self.process.setProcessChannelMode(mode)
        environment = os.environ.copy()
        library_parts = [self.repo / "build/_deps/install/lib", Path("/usr/local/cuda-12.8/lib64")]
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = ":".join(
            [str(path) for path in library_parts if path.is_dir()] + ([existing] if existing else [])
        )
        process_environment = self.process.processEnvironment()
        for key, value in environment.items():
            process_environment.insert(key, value)
        self.process.setProcessEnvironment(process_environment)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._process_finished)
        self.process.start(str(command[0]), command[1:])
        self.status.setText("启动中…")
        self.status.setStyleSheet("color: #b06a00; font-weight: bold")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.controls.setEnabled(False)

    def stop_server(self) -> None:
        if self.process is None:
            return
        self._append_log("stopping server…")
        process = self.process
        self.process.terminate()
        QTimer.singleShot(5000, lambda: self._kill_if_running(process))

    def _kill_if_running(self, process: QProcess) -> None:
        if self.process is not process:
            return
        not_running = QProcess.ProcessState.NotRunning if QT6 else QProcess.NotRunning
        if self.process.state() != not_running:
            self._append_log("server did not exit after 5s; terminating child")
            self.process.kill()

    def _read_output(self) -> None:
        if self.process is not None:
            data = self.output_decoder.decode(bytes(self.process.readAllStandardOutput()))
            if data:
                self._append_log(data.rstrip("\n"))

    def _process_finished(self, exit_code: int, _status) -> None:
        self._read_output()
        self._append_log(f"server exited with code {exit_code}")
        process = self.process
        self.process = None
        if process:
            process.deleteLater()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status.setText("已停止")
        self.status.setStyleSheet("color: #777; font-weight: bold")
        self.controls.setEnabled(True)

    def _process_error(self, error) -> None:
        self._append_log(f"process error: {self.process.errorString() if self.process else error}")
        failed = QProcess.ProcessError.FailedToStart if QT6 else QProcess.FailedToStart
        if error == failed:
            self._process_finished(-1, None)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _poll_health(self) -> None:
        if self.health_in_flight:
            return
        if self.process is None and not self.host.text().strip():
            return
        self.health_in_flight = True
        url = self._health_url()

        def worker() -> None:
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(url, timeout=1.5) as response:
                    body = response.read().decode("utf-8", errors="replace")
                self.health_results.put((url, json.loads(body) == {"status": "ok"}, body))
            except (OSError, ValueError) as error:
                self.health_results.put((url, False, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def _check_health(self, force: bool = False) -> None:
        if force:
            self._poll_health()

    def _drain_health_results(self) -> None:
        try:
            url, ok, detail = self.health_results.get_nowait()
        except queue.Empty:
            return
        self.health_in_flight = False
        if url != self._health_url():
            return
        if ok:
            self.status.setText("运行中 · health OK" if self.process else "已有 API · 非本窗口启动")
            self.status.setStyleSheet("color: #188038; font-weight: bold")
        elif self.process is not None:
            self.status.setText("进程运行中 · 等待 health")
            self.status.setStyleSheet("color: #b06a00; font-weight: bold")
        else:
            self.status.setText("未启动 · health 不可用")
            self.status.setStyleSheet("color: #777; font-weight: bold")

    def _health_url(self) -> str:
        host = self.host.text().strip()
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port.value()}/health"

    def open_api(self) -> None:
        QDesktopServices.openUrl(QUrl(self._health_url()))

    def copy_command(self) -> None:
        QApplication.clipboard().setText(self.command.text())
        self.status.setText("命令已复制")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name.
        self._save_settings()
        if self.process is not None:
            yes = QMessageBox.StandardButton.Yes if QT6 else QMessageBox.Yes
            no = QMessageBox.StandardButton.No if QT6 else QMessageBox.No
            choice = QMessageBox.question(self, "退出控制台", "关闭窗口将停止此窗口启动的 API。继续？", yes | no, no)
            if choice != yes:
                event.ignore()
                return
            process = self.process
            process.terminate()
            if not process.waitForFinished(3000):
                process.kill()
                process.waitForFinished(1000)
        self.health_timer.stop()
        self.queue_timer.stop()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(os.environ.get("NINFER_GUI_REPO", ROOT)))
    parser.add_argument("--binary", default="")
    parser.add_argument("--artifact", default="")
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    window = NInferGui(args.repo.expanduser().resolve(), args.binary, args.artifact)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
