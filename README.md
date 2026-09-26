# FinRisk-OPE

FinRisk-OPE is a date-frozen benchmark for financial off-policy evaluation and selective risk certification. This repository contains the source code, frozen public inputs, committed reference outputs, paper evidence, and camera-ready source.

## Repository layout

```text
README.md
LICENSE
src/
data/
outputs/
artifacts/
paper_source/
scripts/
tests/
```

- `src/`: verified experiment runner, supporting modules, exporter, and dependency lock.
- `data/`: 19 frozen Kenneth R. French Data Library ZIP files.
- `outputs/reference_run/`: complete reference run with 50,784 case rows.
- `artifacts/evidence/`: case-derived evidence used by the paper.
- `artifacts/generated_tables/`: generated LaTeX tables.
- `artifacts/docs/`: protocol, data provenance, release checklist, citation metadata, and upload instructions.
- `paper_source/`: camera-ready TeX, bibliography, figures, tables, and pre-upload PDF.
- `scripts/`: reproduction, verification, repository-URL update, and manifest utilities.
- `tests/`: fast repository and release checks.

## Reproduce

The reference environment is Python 3.14.4 on Windows 11. Exact dependency versions are in `src/requirements-lock.txt`.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r src\requirements-lock.txt
.venv\Scripts\python scripts\reproduce.py
```

On Linux or macOS:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r src/requirements-lock.txt
.venv/bin/python scripts/reproduce.py
```

The full CPU workflow reruns all 15 tasks, exports tables and evidence, and compares them with the committed reference outputs. Source hashes and exact structural counts are required; tightly bounded numerical tolerances cover operating-system and BLAS-level floating-point differences.

Fast structural checks do not rerun the experiment:

```bash
python -m unittest discover -s tests
python scripts/verify_release.py \
  --run outputs/reference_run \
  --assets artifacts \
  --reference outputs/reference_run
```

## Public data

The source datasets come from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Exact source hashes and provenance are recorded in `artifacts/docs/DATA_SOURCES.md` and `outputs/reference_run/manifest.json`.

## Camera-ready paper

The repository URL is already written into `paper_source/main.tex` as:

```text
https://github.com/666lyc-123/FinRisk-OPE
```

The committed `paper_source/main_preupload.pdf` is compiled from this source and includes the same current matched-acceptance results as the generated tables. If the repository URL changes, update the TeX source and rebuild the PDF before submission:

```powershell
python scripts\set_repository_url.py https://github.com/OWNER/FinRisk-OPE
cd paper_source
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The resulting PDF must be validated through IEEE PDF eXpress before CPS submission. The checked-in PDF is a local pre-upload candidate; PDF eXpress approval and CPS submission are still required.


## Scope and licensing

The benchmark uses constructed logging policies over empirical public portfolio returns. It does not model real investor behavior, transaction costs, or market impact. Original code and documentation are released under the MIT License. Third-party files under `data/` remain subject to the terms of the Kenneth R. French Data Library.
