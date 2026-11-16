# Report

The final report is `BREACHZERO_Report.pdf`. It is built from the Typst sources in this folder.

## Files

| File | What it is |
|---|---|
| `BREACHZERO_Report.typ` | The report text and layout. Edit this for wording. |
| `members.typ` | Group member names and registration numbers. |
| `tables.typ` | Every table, generated from `results/*.csv` and `data/*.csv`. Do not edit by hand. |
| `make_submission_zip.py` | Packs the submission ZIP from committed files. |
| `BREACHZERO_Report.pdf` | The output. |

No number in `tables.typ` is typed by hand. The tables are generated from the result CSVs, so
changing a result file and regenerating updates them.

## Rebuilding

The PDF is compiled with Typst:

```bash
pip install typst
python -c "import typst; typst.compile('report/BREACHZERO_Report.typ', output='report/BREACHZERO_Report.pdf', root='.')"
```

Run it from the repository root. The report uses Libertinus Serif for the body, DejaVu Sans Mono
for file names, and Noto Sans Devanagari for the Hindi-style examples. Typst ships the first two.
Install the third with `sudo apt install fonts-noto-devanagari` if the Devanagari rows come out
blank.

## Submission ZIP

```bash
python report/make_submission_zip.py
```

This packs committed files only, with the PDF at the top level and the project folder beside it.
