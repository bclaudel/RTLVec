import subprocess
import sys
import unittest
from pathlib import Path


class PyGImportTests(unittest.TestCase):
    def test_entry_points_import_with_future_warnings_as_errors(self):
        result = subprocess.run(
            [sys.executable, '-W', 'error::FutureWarning', '-c',
             'import train, prepare_data, checkpoint'],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipIf(sys.version_info < (3, 14), 'Python 3.14 warning workaround')
    def test_direct_torchscript_warning_remains_visible(self):
        result = subprocess.run(
            [sys.executable, '-W', 'error::FutureWarning', '-c',
             'import pyg_compat; import torch; torch.jit.script(torch.nn.ReLU())'],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FutureWarning: `torch.jit.script` is not supported', result.stderr)
