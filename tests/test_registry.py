"""
Offline unit tests for the model registry.
"""

import os
import subprocess
import sys


def test_import_falls_back_when_capability_tester_is_unavailable(tmp_path):
    """registry must log and continue when capability_testing can't be imported.

    Runs in a subprocess because the fallback executes at module import time.
    """
    code = (
        "import sys\n"
        "sys.modules['model_calling.capability_testing'] = None\n"
        "import model_calling.registry as r\n"
        "assert r.capability_tester is None\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "CapabilityTester not available" in result.stderr
