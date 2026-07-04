# Development Approach

How we work on this project — who does what, and in what order.

## Model roles

| Phase | Who | Tools |
|---|---|---|
| Brainstorming & planning | **Claude Opus 4.8** | Superpowers skills (`brainstorming`, `writing-plans`) |
| Execution (writing code) | **Claude Fable** | Direct implementation from the plan |
| Review & testing | **The student** | The 5-step ritual in `CLAUDE.md` |

Opus does the thinking (turn an idea into a design and a concrete plan). Fable
does the typing (implement the plan). The student reads, questions, modifies,
and tests everything before it is committed.

## Workflow

1. **Brainstorm** the idea into a short design — Opus + `superpowers:brainstorming`.
2. **Plan** — turn the design into an implementation plan — Opus + `superpowers:writing-plans`.
3. **Execute** — write the code from the plan — Fable.
4. **Review & test** — the student applies the `CLAUDE.md` accept-a-generation ritual.
5. **Commit** — only when the student approves; commit message reflects understanding.

## Current milestone — Step 1 (isolated test)

Before building the full appliance, test the license-plate reading in isolation:

> Read images from a folder and produce a list of Romanian license plate
> numbers — columns: `filename`, `plate_text`, `confidence`.

- **Chosen tool:** `fast-alpr` + `fast-plate-ocr` (`cct-s-v2-global-model`).
- **Runs on:** an x86 laptop first (Python 3.10+), not the Jetson — see
  `docs/research-lpr.md` for why (the best tools require Python 3.10, and a
  batch accuracy test is device-independent).
- **Not in scope yet:** camera capture, real-time video, the web dashboard,
  on-Jetson native deployment. Those come later per `PLAN.md`.
