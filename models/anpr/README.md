# Local ANPR models (v2 Stage B)

Committed on purpose: clone + pip install = working appliance, no runtime
downloads (student decision, 2026-07-19). ~11 MB total.

| file | role | origin |
|---|---|---|
| `yolo-v9-t-384-license-plates-end2end-opset15.onnx` | plate detector | ankandrew/open-image-models hub model `yolo-v9-t-384-license-plate-end2end`, re-stamped opset 17 → 15 with `onnx.version_converter` (outputs verified bit-identical; the Jetson's onnxruntime 1.9 supports opset ≤ 15) |
| `cct_xs_v2_global.onnx` | OCR — plate text + region head | ankandrew/fast-plate-ocr hub model `cct-xs-v2-global-model` (already opset 15) |
| `cct_xs_v2_global_plate_config.yaml` | OCR preprocessing config | ships with the OCR model |

Both upstream projects are MIT-licensed. Chosen by the Stage A bake-off
(`experiments/anpr_bakeoff/RESULTS.md`): 93.5% exact-match on
eu_benchmark, 100% on our real crops, ~337 ms/crop + 110 MB on the
Jetson CPU. Do NOT swap or upgrade these files without re-running the
bake-off.
