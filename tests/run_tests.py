"""Tiny runner so the tests work even without pytest installed:  python tests/run_tests.py"""
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
failed = 0
for mod in ["tests.test_oee", "tests.test_modules"]:
    m = importlib.import_module(mod)
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        try:
            getattr(m, name)()
            print("PASS", mod, name)
        except Exception:
            failed += 1
            print("FAIL", mod, name)
            traceback.print_exc()
print("\n%d failed" % failed)
sys.exit(1 if failed else 0)
