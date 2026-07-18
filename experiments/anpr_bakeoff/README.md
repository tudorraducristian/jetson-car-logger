# ANPR bake-off (v2 Stage A)

Picks the local plate-reading engine by measurement, per
`docs/superpowers/specs/2026-07-18-v2-local-anpr-design.md`.

Machines: LAPTOP = harness dev + fast-alpr runs; JETSON = OpenALPR runs,
real-crop export, ONNX feasibility spike.

## Layout
- `data/<name>/images/*.jpg` + `labels.csv` (`filename,plate_text`,
  filename relative to the dataset dir) + optional `plateless/*.jpg`.
  `data/` is gitignored — regenerate with the scripts below.
- `predictions/<candidate>__<dataset>.csv` — committed evidence.

## Recipes (details in each script's docstring)
1. Public dataset:   `python convert_openalpr_benchmarks.py` (laptop)
2. Real crops:       `python3 export_real_crops.py` (Jetson) + scp back
3. OpenALPR runs:    `python3 run_openalpr.py …` (Jetson)
4. fast-alpr runs:   `python run_fastalpr.py …` (laptop)
5. Score:            `python evaluate.py …` (laptop)
6. Spike (if ONNX wins accuracy): `python3 spike_onnx_jetson.py` (Jetson)

Harness tests: `pytest experiments/anpr_bakeoff -v` (kept out of the app
suite by pytest.ini's `testpaths = tests`).

Verdict: see `RESULTS.md`.
