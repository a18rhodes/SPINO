# Lab notes

Working notes, experiment journals, and roadmaps used during SPINO development.
Until 2026-05-15 these files lived alongside the source tree as gitignored
`CURRENT_STATUS.md` / `NEXT_STEPS.md` files; they were consolidated here for
provenance and reproducibility purposes when the project began preparing
write-ups for external review.

## Status

These are **internal working notes**, not authoritative documentation.

- They may contain abandoned hypotheses, outdated dataset/checkpoint references,
  experiments that were superseded, and informal commentary.
- Dates reflect when an entry was *written*, not when its claims were last
  re-verified.
- Wording is sometimes loose. Numbers should be cross-checked against the
  published artefacts (`docs/results.md` and the JSON summaries in
  `docs/assets/`) before being treated as canonical.

**Authoritative content is in `docs/`. Cite that.**

## Layout

```
lab/
├── project/
│   ├── project_status.md         — top-level roadmap status (Phase 1–3)
│   └── project_next_steps.md
├── circuit/
│   ├── circuit_status.md         — CS-amp composition status
│   ├── circuit_next_steps.md
│   ├── inverter_chain_status.md  — digital inverter chain (negative result)
│   └── ota_status.md             — 5T OTA composition + PFET triode fine-tune
├── mosfet/
│   ├── mosfet_status.md          — NFET / PFET model development log
│   └── mosfet_next_steps.md
└── diode/
    ├── diode_status.md           — dimensionless diode experiment log
    └── diode_next_steps.md
```

Per-folder convention: `<topic>_status.md` for the running journal,
`<topic>_next_steps.md` for the forward roadmap.
The `inverter_chain_status.md` and `ota_status.md` entries are sub-topics
within the circuit area whose forward roadmap content is embedded directly
in the status file.

## Why keep these in git

1. **Provenance.** The decision history that produced each modelling choice is
   visible to readers of the repo, not just to the author.
2. **Reproducibility.** Some entries reference exact commands, hyperparameters,
   and dataset versions used during specific experiments — useful when
   reconstructing a result.
3. **AI-assisted-work transparency.** The repo discloses AI tooling assistance;
   open lab notes reinforce that the modelling decisions were deliberate.
4. **Backup.** Local-only files are vulnerable to disk loss.

## What you will NOT find here

- Authoritative method or results write-ups → see `docs/`.
- API documentation or developer setup → see `docs/DEVELOPMENT.md`.
- Documentation figures or summary JSONs → see `docs/assets/`.
- Code reviews or PR discussion → see git log / commit messages.
