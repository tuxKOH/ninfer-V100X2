#!/usr/bin/env python3
"""Terminal launcher for one owned ninfer-serve child; Python standard library only."""
from __future__ import annotations

import argparse
from collections import deque
import curses
from dataclasses import dataclass
import json
import locale
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import textwrap
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path('/Models/ninfer-V100X2')
MODELS = {
    'quasar': MODEL_ROOT / 'quasar-v3/qwen3_8_27b_nvfp4.ninfer',
    'nvfp4': MODEL_ROOT / 'qwen3_8_27b_nvfp4.ninfer',
    'q4': MODEL_ROOT / 'qwen3_8_27b_q4_k_m.ninfer',
    'fp8': MODEL_ROOT / 'qwen3_8_27b_fp8.ninfer',
}


def local_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Profile:
    label: str
    model: str = 'quasar'
    tp: int = 2
    context: int = 180000
    chunk: int = 2560
    backend: str = 'mtp'
    drafts: int = 3
    optimized: bool = True
    long_attention: bool = False
    greedy: bool = False
    prefix_reuse: bool = True


PROFILES = {
    'quasar-mtp3': Profile('QUASAR / TP2 / MTP3 / 180K'),
    'quasar-mtp4': Profile('QUASAR / TP2 / MTP4 / 180K', drafts=4,
                          greedy=True, prefix_reuse=False),
    'quasar-long': Profile('QUASAR / TP2 / MTP4 / experimental attention',
                           drafts=4, long_attention=True, greedy=True, prefix_reuse=False),
    'quasar-dflash7': Profile('QUASAR / TP2 / DFlash7 / 98K', context=98304,
                              chunk=1024, backend='dflash', drafts=7, optimized=False),
    'nvfp4-tp4': Profile('Official NVFP4 / TP4 / MTP3 / 180K', model='nvfp4', tp=4),
    'fp8-tp4': Profile('Native FP8 / TP4 / MTP3 / 180K', model='fp8', tp=4),
    'q4-mtp3': Profile('Q4_K_M / TP2 / MTP3 / 180K', model='q4', chunk=4096),
}


@dataclass
class Settings:
    binary: str
    artifact: str
    tp: int = 2
    devices: str = '0,1'
    context: int = 180000
    chunk: int = 2560
    backend: str = 'mtp'
    drafts: int = 3
    optimized: bool = True
    long_attention: bool = False
    graphs: bool = True
    prefix_reuse: bool = True
    thinking: bool = False
    greedy: bool = False
    host: str = '127.0.0.1'
    port: int = 8080
    max_output: int = 65536
    pending_timeout_ms: int = 0
    log: str = str(ROOT / 'profiles/tui-server.jsonl')

    def apply_profile(self, name: str) -> None:
        preset = PROFILES[name]
        self.artifact = str(MODELS[preset.model])
        self.tp, self.context, self.chunk = preset.tp, preset.context, preset.chunk
        self.devices = ','.join(map(str, range(preset.tp)))
        self.backend, self.drafts = preset.backend, preset.drafts
        self.optimized, self.long_attention = preset.optimized, preset.long_attention
        self.greedy, self.prefix_reuse = preset.greedy, preset.prefix_reuse

    def command(self) -> list[str]:
        args = [str(local_path(self.binary)), str(local_path(self.artifact)),
                '--host', self.host, '--port', str(self.port), '--tp', str(self.tp),
                '--devices', self.devices, '--max-context', str(self.context),
                '--kv-dtype', 'int8', '--prefill-chunk', str(self.chunk),
                '--max-concurrency', '1', '--default-max-tokens', str(self.max_output),
                '--pending-timeout-ms', str(self.pending_timeout_ms),
                '--request-log-jsonl', str(local_path(self.log)),
                '--log-stats-interval-ms', '1000']
        if self.backend != 'none':
            args += ['--spec', self.backend, '--draft-tokens', str(self.drafts)]
            if self.optimized:
                args += ['--lm-head-draft']
        for enabled, flag in ((self.long_attention, '--volta-long-attention'),
                              (not self.graphs, '--no-cuda-graph'),
                              (not self.prefix_reuse, '--no-prefix-reuse'),
                              (not self.thinking, '--no-thinking'), (self.greedy, '--greedy')):
            if enabled:
                args.append(flag)
        return args

    def validate(self) -> None:
        binary, artifact = local_path(self.binary), local_path(self.artifact)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f'Executable not found: {binary}')
        if not artifact.is_file() or artifact.suffix != '.ninfer':
            raise ValueError(f'Model .ninfer not found: {artifact}')
        ids = self.devices.split(',')
        if (self.tp not in (1, 2, 4) or len(ids) != self.tp or
                any(not item.strip().isdecimal() for item in ids) or
                len(set(map(int, ids))) != len(ids)):
            raise ValueError('Devices must contain TP distinct logical CUDA IDs (TP = 1/2/4)')
        if not 1 <= self.context <= 262144:
            raise ValueError('Native context capacity must be 1..262144')
        if self.chunk <= 0 or self.chunk % 128:
            raise ValueError('Prefill chunk must be a positive multiple of 128')
        if self.backend not in ('none', 'mtp', 'dflash'):
            raise ValueError('Speculation must be none, mtp or dflash')
        if self.backend == 'mtp' and not 1 <= self.drafts <= 5:
            raise ValueError('MTP draft window must be 1..5; accepting zero is valid')
        if self.backend == 'dflash' and (self.tp == 4 or not 1 <= self.drafts <= 7):
            raise ValueError('27B DFlash requires TP1/TP2 and a draft window of 1..7')
        if self.long_attention and (self.tp != 2 or self.backend == 'dflash'):
            raise ValueError('Experimental long attention requires TP2 None/MTP')
        if not self.host.strip() or not 1 <= self.port <= 65535:
            raise ValueError('Host must be nonempty; port must be 1..65535')
        if self.max_output <= 0:
            raise ValueError('Output limit must be positive; remaining context still limits output')
        if self.pending_timeout_ms < 0:
            raise ValueError('Admission timeout must be >= 0 ms (0 = wait as long as needed)')
        if not self.log.strip():
            raise ValueError('Request log path is required')


def runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    libraries = [ROOT / 'build/_deps/install/lib', Path('/usr/local/cuda-12.8/lib64')]
    parts = [str(path) for path in libraries if path.is_dir()]
    if env.get('LD_LIBRARY_PATH'):
        parts.append(env['LD_LIBRARY_PATH'])
    env['LD_LIBRARY_PATH'] = ':'.join(parts)
    return env


class Server:
    """Owns only the process launched here; no process-name matching or external kills."""
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.status = 'STOPPED'
        self.stop_deadline = 0.0
        self.next_health = 0.0
        self.health = None
        self.prefill = self.decode = 0.0

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, settings: Settings) -> None:
        if self.running:
            return
        settings.validate()
        local_path(settings.log).parent.mkdir(parents=True, exist_ok=True)
        self.prefill = self.decode = 0.0
        self.status, self.stop_deadline = 'LOADING', 0.0
        self.process = subprocess.Popen(settings.command(), cwd=ROOT, env=runtime_environment(),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', errors='replace')
        process = self.process

        def read_output() -> None:
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    self.lines.put(line.rstrip('\n'))
        threading.Thread(target=read_output, daemon=True).start()

    def stop(self) -> None:
        if self.running and not self.stop_deadline:
            self.process.terminate()
            self.status = 'STOPPING'
            self.stop_deadline = time.monotonic() + 5

    def poll(self, settings: Settings) -> None:
        now = time.monotonic()
        if self.stop_deadline and self.running and now >= self.stop_deadline:
            self.process.kill()
            self.stop_deadline = 0.0
        if self.process is not None and not self.running:
            self.status = f'EXIT {self.process.returncode}'
            self.process = None
            self.stop_deadline = 0.0
        if not self.running or self.stop_deadline or self.health or now < self.next_health:
            return
        self.next_health = now + 2
        host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(settings.host, settings.host)
        if ':' in host:
            host = '[' + host.strip('[]') + ']'
        process = self.process

        def check_health() -> None:
            try:
                with urlopen(f'http://{host}:{settings.port}/health', timeout=0.5) as response:
                    online = json.load(response).get('status') == 'ok'
                # /health can report readiness before model upload completes; the
                # listening log is the authoritative transition for our own child.
                if self.process is process and online and self.status == 'ONLINE':
                    self.status = 'ONLINE / HEALTH OK'
            except (OSError, URLError, ValueError):
                pass
            finally:
                self.health = None
        self.health = threading.Thread(target=check_health, daemon=True)
        self.health.start()

    def drain(self) -> list[str]:
        import re
        lines = []
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                break
            lines.append(line)
            if 'listening on http' in line and self.running and not self.stop_deadline:
                self.status = 'ONLINE'
            rates = re.search(r'throughput interval=.*?prefill=([\d.]+)tok/s decode=([\d.]+)tok/s', line)
            if rates:
                self.prefill, self.decode = map(float, rates.groups())
        return lines

    def close(self) -> None:
        if self.running:
            self.stop()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


FIELDS = [
    ('artifact', 'Model .ninfer', None), ('binary', 'ninfer-serve', None),
    ('tp', 'Tensor parallel', (1, 2, 4)), ('devices', 'CUDA devices', None),
    ('context', 'Context capacity', None), ('chunk', 'Prefill chunk', None),
    ('backend', 'Speculation', ('mtp', 'dflash', 'none')), ('drafts', 'Max drafts', None),
    ('optimized', 'Optimized draft head', (False, True)),
    ('long_attention', 'Experimental attention', (False, True)),
    ('graphs', 'CUDA Graphs', (False, True)), ('prefix_reuse', 'Prefix cache', (False, True)),
    ('thinking', 'Default thinking', (False, True)), ('greedy', 'Force greedy', (False, True)),
    ('max_output', 'Max output tokens', None),
    ('pending_timeout_ms', 'Admission timeout ms (0 = unlimited)', None),
    ('host', 'API host', None),
    ('port', 'API port', None), ('log', 'Request JSONL log', None),
]


def put(screen, y: int, x: int, text: str, style: int = 0) -> None:
    height, width = screen.getmaxyx()
    if y < height and x < width:
        try:
            screen.addnstr(y, x, text, max(0, width - x - 1), style)
        except curses.error:
            pass


class Launcher:
    def __init__(self, settings: Settings, profile: str) -> None:
        self.settings, self.profile = settings, profile
        self.server = Server()
        self.selected = 0
        self.logs: deque[str] = deque(maxlen=2000)
        self.notice = ''

    def edit(self, screen, key: str, label: str) -> None:
        value = str(getattr(self.settings, key))
        text = value
        while True:
            self.draw(screen)
            height, width = screen.getmaxyx()
            put(screen, height - 3, 0, ' ' * (width - 1))
            put(screen, height - 3, 1, label + ' (Enter save, Esc cancel): ' + text,
                curses.A_REVERSE)
            screen.refresh()
            try:
                character = screen.get_wch()
            except curses.error:
                continue
            if character == '\x1b':
                return
            if character in ('\n', '\r', curses.KEY_ENTER):
                try:
                    setattr(self.settings, key, int(text) if isinstance(getattr(self.settings, key), int)
                            else text.strip())
                    self.profile = 'custom'
                except ValueError:
                    self.notice = 'Enter an integer'
                return
            if character in ('\b', '\x7f', curses.KEY_BACKSPACE):
                text = text[:-1]
            elif character == '\x15':
                text = ''
            elif isinstance(character, str) and character.isprintable():
                text += character

    def choose(self, screen, title: str, choices: list[tuple[str, str]]) -> str | None:
        index = 0
        while True:
            self.server.poll(self.settings)
            self.logs.extend(self.server.drain())
            screen.erase()
            height, width = screen.getmaxyx()
            put(screen, 0, 1, title + ' / Up Down Enter / Esc cancel', curses.A_BOLD)
            visible = max(1, height - 4)
            start = max(0, index - visible + 1)
            for row, (_, label) in enumerate(choices[start:start + visible]):
                put(screen, row + 2, 1, label, curses.A_REVERSE if row + start == index else 0)
            screen.refresh()
            key = screen.getch()
            if key in (curses.KEY_UP, ord('k')):
                index = (index - 1) % len(choices)
            elif key in (curses.KEY_DOWN, ord('j')):
                index = (index + 1) % len(choices)
            elif key in (10, 13, curses.KEY_ENTER):
                return choices[index][0]
            elif key == 27:
                return None

    def show_command(self, screen) -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        put(screen, 0, 1, 'Exact launch command / any key to return', curses.A_BOLD)
        for index, line in enumerate(textwrap.wrap(shlex.join(self.settings.command()), max(1, width - 4))):
            if index + 2 < height - 1:
                put(screen, index + 2, 2, line)
        screen.refresh()
        screen.timeout(-1)
        screen.getch()
        screen.timeout(100)

    def draw(self, screen) -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 18 or width < 80:
            put(screen, 0, 0, 'Resize terminal to at least 80 x 18; q exits.')
            screen.refresh()
            return
        divider = width // 2
        put(screen, 0, 1, 'NInfer V100 / QUASAR NVFP4 / API launcher', curses.A_BOLD)
        put(screen, 1, 1, 'Profile: ' + PROFILES[self.profile].label if self.profile in PROFILES
            else 'Profile: custom')
        put(screen, 2, 1, f'{self.server.status}   Prefill {self.server.prefill:.1f} tok/s'
            f'   Committed decode {self.server.decode:.1f} tok/s  [1s intervals]')
        put(screen, 3, 1, 'SETTINGS (locked while running)', curses.A_BOLD)
        put(screen, 3, divider + 1, 'LIVE SERVER LOG', curses.A_BOLD)
        visible = height - 9
        start = max(0, self.selected - visible + 1)
        for row, (key, label, _) in enumerate(FIELDS[start:start + visible]):
            value = getattr(self.settings, key)
            value = ('ON' if value else 'OFF') if isinstance(value, bool) else str(value)
            if len(value) > divider - 25:
                value = '…' + value[-max(1, divider - 26):]
            line = f'{label:22} {value}'
            put(screen, 4 + row, 1, line[:divider - 2],
                curses.A_REVERSE if start + row == self.selected else 0)
        wrapped = [part for line in list(self.logs)[-visible:]
                   for part in textwrap.wrap(line, max(1, width - divider - 3))]
        for row, line in enumerate(wrapped[-visible:]):
            put(screen, row + 4, divider + 1, line)
        put(screen, height - 4, 1, self.notice)
        put(screen, height - 3, 1, 'Long attention changes numerical output; DFlash is workload-dependent.')
        put(screen, height - 2, 1, 'Up/Down/Tab field  Enter edit  p preset  m model  c cmd  s start/stop  q quit')
        screen.refresh()

    def run(self, screen) -> None:
        curses.curs_set(0)
        screen.timeout(100)
        screen.keypad(True)
        try:
            while True:
                self.server.poll(self.settings)
                self.logs.extend(self.server.drain())
                self.draw(screen)
                key = screen.getch()
                if key in (ord('q'), 27):
                    return
                if key in (curses.KEY_DOWN, ord('j'), 9):
                    self.selected = (self.selected + 1) % len(FIELDS)
                elif key in (curses.KEY_UP, ord('k'), curses.KEY_BTAB):
                    self.selected = (self.selected - 1) % len(FIELDS)
                elif key == ord('c'):
                    self.show_command(screen)
                elif key == ord('s'):
                    if self.server.running:
                        self.server.stop()
                    else:
                        try:
                            self.server.start(self.settings)
                            self.notice = 'Started owned child; q stops it before exit'
                        except (OSError, ValueError) as error:
                            self.notice = str(error)
                elif not self.server.running:
                    if key == ord('p'):
                        choice = self.choose(screen, 'Select profile',
                                             [(name, preset.label) for name, preset in PROFILES.items()])
                        if choice:
                            self.settings.apply_profile(choice)
                            self.profile = choice
                            self.notice = 'Preset applied; edit paths before starting if needed'
                    elif key == ord('m'):
                        choices = [(str(path), name + ' / ' + str(path))
                                   for name, path in MODELS.items() if path.is_file()]
                        choices.append(('', 'Enter an explicit model path'))
                        choice = self.choose(screen, 'Model artifact', choices)
                        if choice:
                            self.settings.artifact = choice
                            self.profile = 'custom'
                        elif choice == '':
                            self.edit(screen, 'artifact', 'Model .ninfer')
                    elif key in (10, 13, curses.KEY_ENTER, ord(' ')):
                        field, label, choices = FIELDS[self.selected]
                        if choices:
                            value = getattr(self.settings, field)
                            setattr(self.settings, field, choices[(choices.index(value) + 1) % len(choices)])
                            self.profile = 'custom'
                        else:
                            self.edit(screen, field, label)
        finally:
            self.server.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='quasar-mtp3')
    parser.add_argument('--binary', type=Path, help='explicit ninfer-serve executable')
    parser.add_argument('--artifact', type=Path, help='explicit model .ninfer path')
    parser.add_argument('--pending-timeout-ms', type=int,
                        help='admission queue timeout in ms; 0 waits as long as needed (default 0)')
    parser.add_argument('--print-command', action='store_true', help='print command without loading a model')
    args = parser.parse_args()
    candidates = [ROOT / 'build-v100/apps/ninfer-serve', ROOT / 'build-v100-tp4/apps/ninfer-serve']
    binary = args.binary or next((path for path in candidates if path.is_file()), candidates[0])
    settings = Settings(binary=str(binary), artifact='')
    settings.apply_profile(args.profile)
    if args.artifact:
        settings.artifact = str(args.artifact.expanduser().resolve())
    if args.pending_timeout_ms is not None:
        settings.pending_timeout_ms = args.pending_timeout_ms
    if args.print_command:
        print(shlex.join(settings.command()))
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error('TUI requires a terminal; use --print-command for command output')
    locale.setlocale(locale.LC_ALL, '')
    try:
        curses.wrapper(Launcher(settings, args.profile).run)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
