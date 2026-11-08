"""Regenerates figures/three_person_map.png from the three-person flight test.

Runs Skydio_Flight_Experiment/build_charts.py (which reads outputs/people.csv written by run_experiment.py) and copies
its gps_map.png here.
"""
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "Skydio_Flight_Experiment"
runpy.run_path(str(EXP / "build_charts.py"), run_name="__main__")
shutil.copy(EXP / "outputs" / "gps_map.png", HERE / "figures" / "three_person_map.png")
print("wrote three_person_map.png")
