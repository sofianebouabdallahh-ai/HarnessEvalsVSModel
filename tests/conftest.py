import os

# Tests run on whatever machine is handy; the disk guard must not fail them, it is tested directly.
os.environ.setdefault("HM_MIN_FREE_GB", "0")
os.environ.setdefault("HM_HARD_FLOOR_GB", "0")
