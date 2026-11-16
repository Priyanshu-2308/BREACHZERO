# Builds the submission ZIP from committed files only, using git archive.
# Layout inside the ZIP:
#   BREACHZERO_Report.pdf        the report on its own at the top level
#   BREACHZERO/...               the project folder next to it
# Anything untracked stays out, so no .git, venv, model weights or caches.

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "BREACHZERO_submission.zip")
PDF_IN_REPO = "report/BREACHZERO_Report.pdf"


def git(*args):
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout.strip()


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print("warning: tracked files differ from the commit being packed:")
        print(dirty)

    tracked = git("ls-tree", "-r", "--name-only", rev).splitlines()
    if PDF_IN_REPO not in tracked:
        raise SystemExit(f"{PDF_IN_REPO} is not committed at {rev}; commit it first")

    with tempfile.TemporaryDirectory() as tmp:
        tar = os.path.join(tmp, "project.tar")
        git("archive", "--format=tar", "--prefix=BREACHZERO/", "-o", tar, rev)

        stage = os.path.join(tmp, "stage")
        os.makedirs(stage)
        subprocess.run(["tar", "-xf", tar, "-C", stage], check=True)

        # the report also sits at the top level so it is found without digging
        shutil.copy2(os.path.join(stage, "BREACHZERO", PDF_IN_REPO),
                     os.path.join(stage, "BREACHZERO_Report.pdf"))

        files = []
        for dirpath, _, filenames in os.walk(stage):
            for name in sorted(filenames):
                files.append(os.path.join(dirpath, name))
        files.sort()

        if os.path.exists(OUT):
            os.remove(OUT)
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
            for full in files:
                z.write(full, os.path.relpath(full, stage))

    size = os.path.getsize(OUT) / (1024 * 1024)
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
    print(f"wrote {OUT}")
    print(f"packed {rev} ({git('rev-parse', '--short', rev)}): {len(names)} files, {size:.1f} MB")
    print("report at top level:", "BREACHZERO_Report.pdf" in names)


if __name__ == "__main__":
    main()
