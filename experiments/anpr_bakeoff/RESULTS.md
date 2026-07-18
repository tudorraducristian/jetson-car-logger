# ANPR bake-off — results

**Date:** 2026-07-18 · **Status:** DRAFT until the Task 9/10 boxes are ticked.

## Decision rule (fixed in the spec BEFORE measuring)
Winner = most accurate on exact-match that runs on the Jetson at
< 2 s/crop and < 500 MB added RAM. ONNX feasibility order: PyPI
onnxruntime 1.9.0 (CPU) → TensorRT 8.2 trtexec → both fail ⇒ OpenALPR
wins by feasibility.

## Accuracy — eu_benchmark (public, N=108)

| candidate | n | exact match | mean CER | read rate | FP rate | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|
| openalpr_eu | 108 | 26.9% | 0.635 | 38.9% | n/a | 1092 | 1374 |
| fastalpr_eu | 108 | 88.9% | 0.028 | 100.0% | n/a | 119 | 192 |
| fastalpr_global | 108 | 93.5% | 0.011 | 100.0% | n/a | 76 | 138 |

Latency caveat: openalpr ms are Jetson wall times paying full process +
model-load per image (in-process will be faster); fastalpr ms are laptop
times, indicative only — the on-device number comes from the Task 9 spike.

### Close-call gate (Task 8 Step 2)
The **engine** question is decisive: both fast-alpr variants beat
OpenALPR by ~62 percentage points on exact-match. The 10-pp gate DOES
trigger *between the two fast-alpr variants* (93.5% vs 88.9%, Δ = 4.6 pp
— same engine, different OCR model), so N=108 alone cannot rank the two
OCR variants with confidence. Resolution (student's call, 2026-07-18):
the Task 9 spike ships BOTH OCR models to the Jetson and lets on-device
feasibility/latency separate the variants; a second public dataset gets
added only if both pass equivalently and the variant choice still
matters for the verdict.

## Accuracy — real_crops (ours, N=9, cloud reads as ground truth)

| candidate | n | exact match | mean CER | read rate | FP rate | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|
| openalpr_eu | 9 | 44.4% | 0.556 | 44.4% | n/a | 951 | 1117 |
| fastalpr_eu | 9 | 88.9% | 0.048 | 100.0% | n/a | 174 | 235 |
| fastalpr_global | 9 | 100.0% | 0.000 | 100.0% | n/a | 64 | 87 |

Notable: the 3 true camera captures (the Dacia, CJ45ARL) were read 3/3
by BOTH fast-alpr variants and 0/3 by OpenALPR. fastalpr_eu's single
miss is event_13 (an EL147AD photo, read as "CLI47A"). FP rate is n/a on
both datasets: the student's plateless triage found no qualifying crops.

## On-device numbers (Jetson)
| candidate | latency/crop | peak RSS | source |
|---|---|---|---|
| openalpr_eu | 1092 ms wall avg (model-only 206 ms) | 128528 KB ≈ 126 MB | measured, alpr CLI (per-process; in-process will be faster) |
| fastalpr_global | ≈337 ms (detector 305 + OCR 32) | 110 MB | Task 9 spike, ORT 1.9 CPU, **SPIKE PASS** |

### Task 9 spike — ONNX-on-Jetson feasibility (ORT 1.9.0 CPU, py3.6)
`SPIKE PASS` for the **global** stack: the cct-xs-v2-global OCR read
`CJ45ARL` correct (conf 1.000) at 32 ms/crop, and the yolo-v9-t detector
ran at 305 ms/crop returning the expected `(1, 7)` output; peak process
RSS 110 MB. Both budgets met (< 2 s/crop, < 500 MB).

Two feasibility facts fixed on the way (both matter for Stage B):
1. **numpy ≥1.19.5 wheels SIGILL on the Tegra X1** unless
   `OPENBLAS_CORETYPE=ARMV8` is set — the Stage B systemd unit must
   export it.
2. **Opset ceiling.** ORT 1.9 (the last onnxruntime with a cp36 aarch64
   wheel) officially supports **opset ≤15**. The models as shipped by
   the hub are higher: yolo-v9-t detector = opset 17, european OCR =
   opset 18. The detector was re-stamped to opset 15 with
   `onnx.version_converter` — output is **bit-identical** to the
   original (verified on the laptop), and it loads+runs on ORT 1.9. The
   **european** OCR could NOT be downgraded (Resize op has no opset-15
   adapter), so it is not usable on this runtime — but it lost on
   accuracy anyway, so the winner is unaffected. Stage B must ship the
   **opset-15 detector** + the **already-opset-15 global OCR**.

This also **resolves the close-call gate**: on-device feasibility itself
separated the two OCR variants — the european model does not even load
on ORT 1.9, while the global model (the accuracy leader, 93.5% / 100%)
runs comfortably. No second public dataset needed.

## Verdict
<filled by Task 10 — winner + why, citing the rule and the tables>

## Confidence calibration notes (for Stage B's min_vehicle_confidence)
<from calib_<winner>__*.csv: at which confidence do wrong reads die out?
1-2 sentences + the threshold the STUDENT picks.>
