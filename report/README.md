# Report build

The final report is `BREACHZERO_Report.pdf`. It is built from Typst sources in this folder.

## Files

| File | What it is |
|---|---|
| `BREACHZERO_Report.typ` | The report text and layout. Edit this for wording. |
| `members.typ` | Group member names and registration numbers. Edit this only for member details. |
| `tables.typ` | Generated. Every table, built from `results/*.csv` and `data/*.csv`. Do not edit. |
| `build_report.py` | Regenerates `tables.typ` and compiles the PDF. |
| `audit_report.py` | Checks the built PDF against the files it quotes. |
| `BREACHZERO_Report.pdf` | The output. |

No number in `tables.typ` is typed by hand. `build_report.py` reads the CSVs, formats every
value to four decimals, and writes the table bodies. Changing a result file and re-running the
build updates the tables.

## Requirements

```bash
pip install typst pandas pypdf
```

The PDF uses Libertinus Serif for the body, DejaVu Sans Mono for file names and code, and
Noto Sans Devanagari for the Hindi-style examples. Typst ships the first two. Install the third
with `sudo apt install fonts-noto-devanagari` if the Devanagari rows come out blank.

## Build

```bash
python report/build_report.py
```

Run it from the repository root. It prints the page count when it finishes.

## Audit

```bash
python report/audit_report.py
```

It checks three things and exits non-zero if any fail:

1. Every value in the main result tables appears in the PDF, and every number in the PDF traces
   back to a results file, a data file, or a constant defined in the project source.
2. No banned words, em dashes, emoji, or bold runs inside paragraphs, in the PDF or in any
   source file here.
3. Page count between 12 and 18, member table present with no placeholders, all required
   sections present, and every numbered table and figure referred to in the body text.

`pdftotext` is used to read the PDF text. On Debian or Ubuntu it comes from `poppler-utils`.
