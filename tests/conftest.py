"""Test environment only: a throwaway cache dir and no network (both seams raise)."""

import os
import tempfile

_CACHE = tempfile.TemporaryDirectory(prefix="insureport-test-")  # removed at interpreter exit
os.environ["INSUREPORT_DATA_DIR"] = _CACHE.name
os.environ["INSUREPORT_OFFLINE"] = "1"
