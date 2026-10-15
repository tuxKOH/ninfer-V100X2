"""Launcher command/admission and owned-child lifecycle checks, without CUDA."""
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.v100.ninfer_tui import Server, Settings


class LaunchContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.artifact = Path(self.directory.name) / 'model.ninfer'
        self.artifact.touch()
        self.settings = Settings(binary=sys.executable, artifact=str(self.artifact),
                                 log=str(Path(self.directory.name) / 'request.jsonl'))

    def test_coding_and_dflash_commands(self):
        for profile, drafts, context in [('quasar-mtp3', '3', '180000'),
                                          ('quasar-mtp4', '4', '180000'),
                                          ('quasar-long', '4', '180000'),
                                          ('quasar-dflash7', '7', '98304')]:
            with self.subTest(profile=profile):
                self.settings.apply_profile(profile)
                self.settings.artifact = str(self.artifact)
                self.settings.validate()
                command = self.settings.command()
                self.assertEqual(command[command.index('--draft-tokens') + 1], drafts)
                self.assertEqual(command[command.index('--max-context') + 1], context)
                self.assertEqual('--volta-long-attention' in command, profile == 'quasar-long')
                self.assertEqual('--lm-head-draft' in command, profile != 'quasar-dflash7')
                if profile in ('quasar-mtp4', 'quasar-long'):
                    self.assertIn('--greedy', command)
                    self.assertIn('--no-prefix-reuse', command)

    def test_admission_timeout_is_forwarded(self):
        # 0 means "wait as long as the Engine needs"; it must reach the child verbatim, and a
        # negative value must be refused before launch.
        self.settings.pending_timeout_ms = 0
        command = self.settings.command()
        self.assertEqual(command[command.index('--pending-timeout-ms') + 1], '0')
        self.settings.pending_timeout_ms = -1
        with self.assertRaisesRegex(ValueError, 'timeout'):
            self.settings.validate()

    def test_plain_decode_omits_speculative_options(self):
        self.settings.backend = 'none'
        command = self.settings.command()
        self.assertNotIn('--spec', command)
        self.assertNotIn('--draft-tokens', command)
        self.assertNotIn('--lm-head-draft', command)

    def test_explicit_paths_and_cuda_mapping(self):
        self.settings.artifact = str(Path(self.directory.name) / 'model with spaces.ninfer')
        Path(self.settings.artifact).touch()
        self.settings.devices = '4,6'
        self.settings.validate()
        command = self.settings.command()
        self.assertEqual(command[1], self.settings.artifact)
        self.assertEqual(command[command.index('--devices') + 1], '4,6')
        self.settings.devices = '0,0'
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.settings.validate()

    def test_unsupported_long_attention_and_tp4_dflash(self):
        for tp, backend, long_attention in [(4, 'dflash', False), (2, 'dflash', True),
                                            (1, 'mtp', True)]:
            self.settings.tp = tp
            self.settings.devices = ','.join(map(str, range(tp)))
            self.settings.backend = backend
            self.settings.long_attention = long_attention
            with self.subTest(tp=tp, backend=backend), self.assertRaises(ValueError):
                self.settings.validate()

    def test_owned_child_stop_and_logs(self):
        external = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: external.poll() is None and external.terminate())
        server = Server()
        self.addCleanup(server.close)
        script = "print('owned child ready', flush=True); import time; time.sleep(30)"
        with patch.object(self.settings, 'command', return_value=[sys.executable, '-u', '-c', script]):
            server.start(self.settings)
        child = server.process
        lines = []
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and 'owned child ready' not in lines:
            lines.extend(server.drain())
            time.sleep(0.01)
        self.assertIn('owned child ready', lines)
        server.close()
        self.assertIsNotNone(child.poll())
        self.assertIsNone(external.poll())
        external.terminate()
        external.wait(timeout=3)

    def test_interval_rates_do_not_use_request_average(self):
        server = Server()
        server.lines.put('throughput interval=1.000s prefill=1234.5tok/s decode=143.5tok/s')
        server.drain()
        self.assertEqual((server.prefill, server.decode), (1234.5, 143.5))
        server.lines.put('[req 1] done prefill=900.0tok/s decode=100.0tok/s')
        server.drain()
        self.assertEqual((server.prefill, server.decode), (1234.5, 143.5))


if __name__ == '__main__':
    unittest.main()
