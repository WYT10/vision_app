#!/usr/bin/env python3
from pathlib import Path
import runpy
runpy.run_path(str((Path(__file__).resolve().parent / "training_tools/eval_export_cls.py").resolve()), run_name="__main__")
