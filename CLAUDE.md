# Working agreement

## ⛔ A QUESTION IS A QUESTION — NEVER AN ACTION

When the user asks a question, **ANSWER IT. Do not do anything else.**

- A question (e.g. "Why X?", "What about Y?", "Where is Z?", "Can it do W?",
  "Don't you think...?") is a request for an **answer only** — never a license to
  edit files, run commands, reconstruct, plot, commit, or "helpfully" act.
- **Take an action ONLY when the user explicitly tells you to** ("do it", "run it",
  "build it", "commit", "go ahead"). If unsure whether something is an
  instruction, treat it as a question and ask.
- This holds even when the answer obviously suggests a next step. State the next
  step in words and **stop**. Wait for the go-ahead.

## Other standing rules
- Ask questions plainly, in prose — not as multiple-choice "shopping list" menus
  (avoid the AskUserQuestion option-list format). State what you need to know directly.
- Never use the word "honest"/"honestly" — state the thing plainly.
- Don't overclaim: include all the relevant options before any "X beats Y" claim;
  lead with measured numbers, not adjectives.
- Always state the path of any figure you produce (e.g.
  `recoExamples/.../figures/foo.png`) so it can be opened.
- Confirm before writing/building/committing/pushing.

## Status docs — READ THESE to orient on the software state
Dated per-area inventories live in `statusmd/`. Read the relevant one before working
in that area:
- `statusmd/pp_status.md` — RecoCrysp core (`src/`): Joseph projectors, geometry,
  phantoms, PSF, reconstruction (MLEM/OSEM, penalized De Pierro / OSL Huber).
- `statusmd/reco_status.md` — `recoExamples/src/` helpers (PTCRYSP readers,
  attenuation, sinogram scatter/randoms, NEMA ROIs) + the locked recipe.
- `statusmd/reco_spheres_status.md` — `recoExamples/sphere/` MC studies (normalization,
  randoms, attenuation, scatter on the uniform sphere).
- `statusmd/reco_nema_status.md` — `recoExamples/nema/` contrast-phantom studies.
- `statusmd/nema_water_methods_wip.md` — ACTIVE WIP: the nema_la_water method
  comparison (branch `osl-safeguard`), clinical NEMA benchmark, and the next steps.
