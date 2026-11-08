"""QwenFX on one Linux machine with one GPU: fx's exact RV32 binary on the in-weights CPU, with
Qwen3.5-4B as its model.

    uv run python qwenfx.py setup                       # fx's ELF, rv32emu, Qwen3.5-4B
    uv run python qwenfx.py chip                        # build the chip file from the Python circuit
    uv run python qwenfx.py task t1_bug                 # fx fixes a bug, then the task's hidden check
    uv run python qwenfx.py seed ~/fx-sandbox           # the inventory repository, to push as your own
    uv run python qwenfx.py issues --repo you/fx-sandbox
    uv run python qwenfx.py pr 1 --repo you/fx-sandbox  # issue #1 in, a pull request out
    uv run python qwenfx.py pr-local t1_bug             # the same flow with a local bare repository

Every run checks itself: rv32emu runs beside the chip in lockstep, and after the run every chip token
is audited against the reference machine. `task` and `pr` pass any further arguments to
fx/fxio/live_chip.py (for example --no-audit, or --tasks / --reps on task).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
ELF = os.path.join(BUILD, "fx-fastcopy.elf")
ELF_XZ = os.path.join(ROOT, "third_party", "fx", "fx-fastcopy.elf.xz")
ELF_SHA256 = "282e53d497aa113173bc3ad9aa2ab221b6f43e43360457a093aeda7c1a85cbbd"
IMPORTS = os.path.join(ROOT, "third_party", "fx", "imports.json")
DLL = os.path.join(BUILD, "rv32emu.so")
RV32EMU_C = os.path.join(ROOT, "fx", "rv", "rv32emu.c")
LIVE_CHIP = os.path.join(ROOT, "fx", "fxio", "live_chip.py")
PR_DRIVER = os.path.join(ROOT, "fx", "fxio", "pr_driver.py")
PR_LOCAL = os.path.join(ROOT, "fx", "fxio", "pr_local.py")
QWEN = "Qwen/Qwen3.5-4B"
FRAMES = 4096
# the 13 hot-code units of docs/fx/one.md, or none (docs/fx/live.md, docs/fx/pr.md)
CHIP_FILES = {"all+": os.path.join(ROOT, "chip", "fx-live-u13-f4096.safetensors"),
              "": os.path.join(ROOT, "chip", "fx-live-f4096.safetensors")}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def model_dir(download: bool = False) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(QWEN, local_files_only=not download)
    except Exception as e:  # noqa: BLE001
        if download:
            raise
        raise SystemExit(f"no local copy of {QWEN} ({type(e).__name__}): run `qwenfx.py setup` first")


def cmd_setup(a) -> None:
    os.makedirs(BUILD, exist_ok=True)
    if not (os.path.exists(ELF) and sha256(ELF) == ELF_SHA256):
        with lzma.open(ELF_XZ) as src, open(ELF + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        os.replace(ELF + ".part", ELF)
    got = sha256(ELF)
    if got != ELF_SHA256:
        raise SystemExit(f"{ELF}: sha256 {got}, expected {ELF_SHA256}")
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        raise SystemExit("needs a C compiler (gcc) to build rv32emu.so")
    subprocess.run([cc, "-O2", "-shared", "-fPIC", RV32EMU_C, "-o", DLL], check=True)
    m = model_dir(download=not a.no_download)
    print(json.dumps({"elf": ELF, "elf_sha256": got, "rv32emu": DLL, "imports": IMPORTS, "qwen": m}))


def need_setup() -> str:
    if not (os.path.exists(ELF) and os.path.exists(DLL)):
        raise SystemExit("run `uv run python qwenfx.py setup` first")
    return model_dir()


def units_of(a) -> str:
    return "" if a.no_units else "all+"


def chip_file_of(a) -> str | None:
    if a.no_chip_file:
        return None
    p = a.chip_file or CHIP_FILES[units_of(a)]
    if not os.path.exists(p):
        raise SystemExit(f"no chip file at {p}: build it with `qwenfx.py chip{' --no-units' if a.no_units else ''}`,"
                         " or pass --no-chip-file to build the circuit in Python for this run")
    return p


def base_args(m: str, out: str, units: str) -> list[str]:
    args = [DLL, ELF, IMPORTS, m, out, "--full-regs", "--frames", str(FRAMES)]
    return args + (["--units", units] if units else [])


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


def cmd_chip(a) -> None:
    m = need_setup()
    units = units_of(a)
    out = a.out or CHIP_FILES[units]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # a scripted model (not Qwen): writing the file needs this configuration's boot snapshot only
    run([sys.executable, "-u", LIVE_CHIP, *base_args(m, os.path.join(ROOT, "runs", "chipbuild"), units),
         "--tasks", "t1_bug", "--backend", "scripted", "--write-chip-file", out])


def cmd_task(a, rest) -> None:
    m = need_setup()
    cf = chip_file_of(a)
    out = a.out or os.path.join(ROOT, "runs", "live")
    cmd = [sys.executable, "-u", LIVE_CHIP, *base_args(m, out, units_of(a)), "--tasks", *a.tasks,
           "--reps", *map(str, a.reps), "--backend", f"hf:{QWEN}"]
    run(cmd + (["--chip-file", cf] if cf else []) + rest)


def cmd_pr(a, rest) -> None:
    m = need_setup()
    cf = chip_file_of(a)
    cmd = [sys.executable, "-u", PR_DRIVER, "run", str(a.issue), "--dll", DLL, "--elf", ELF, "--imports", IMPORTS,
           "--model", m, "--units", units_of(a), "--frames", str(FRAMES), "--seed", str(a.seed),
           "--out", a.out or os.path.join(ROOT, "runs", "pr")]
    if a.repo:
        cmd += ["--repo", a.repo]
    if a.author:
        cmd += ["--author", a.author]
    if cf:
        cmd += ["--chip-file", cf]
    run(cmd + (["--live-args", *rest] if rest else []))


def cmd_pr_local(a, rest) -> None:
    m = need_setup()
    cf = chip_file_of(a)
    out = a.out or os.path.join(ROOT, "runs", "pr-local")
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} is not empty")
    cmd = [sys.executable, "-u", PR_LOCAL, out, a.task, str(a.seed), DLL, ELF, IMPORTS, m,
           "--full-regs", "--frames", str(FRAMES)] + (["--units", units_of(a)] if units_of(a) else [])
    run(cmd + (["--chip-file", cf] if cf else []) + rest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="fx's ELF (sha256 checked), rv32emu.so, Qwen3.5-4B")
    s.add_argument("--no-download", action="store_true", help="use the local copy of Qwen3.5-4B only")
    c = sub.add_parser("chip", help="build the chip file from the Python circuit")
    c.add_argument("--out", default=None)
    t = sub.add_parser("task", help="fx on the chip with Qwen3.5-4B on a P0c task, then its hidden check")
    t.add_argument("tasks", nargs="+", choices=("t1_bug", "t2_rename", "t3_feature", "t4_config", "t5_parse"))
    t.add_argument("--reps", nargs="+", type=int, default=[1], help="reps; the seed is the rep")
    p = sub.add_parser("pr", help="issue N of your repository in, a pull request out (needs gh)")
    p.add_argument("issue", type=int)
    p.add_argument("--repo", default=None, help="your repository, OWNER/NAME (or QWENFX_REPO)")
    p.add_argument("--author", default=None, help="commit identity 'Name <email>' (default: git config)")
    p.add_argument("--seed", type=int, default=1)
    pl = sub.add_parser("pr-local", help="the issue-to-PR flow against a local bare repository, no GitHub")
    pl.add_argument("task", choices=("t1_bug", "t2_rename", "t3_feature", "t4_config", "t5_parse"))
    pl.add_argument("--seed", type=int, default=1)
    for x in (c, t, p, pl):
        x.add_argument("--no-units", action="store_true", help="the circuit without the 13 hot-code units")
    for x in (t, p, pl):
        x.add_argument("--chip-file", default=None, help="default: chip/fx-live-u13-f4096.safetensors")
        x.add_argument("--no-chip-file", action="store_true", help="build the circuit in Python instead")
        x.add_argument("--out", default=None)
    sd = sub.add_parser("seed", help="write the inventory repository to DIR (git, one commit; nothing pushed)")
    sd.add_argument("dir")
    i = sub.add_parser("issues", help="open one issue per P0c task on your repository (needs gh)")
    i.add_argument("--repo", default=None)
    a, rest = ap.parse_known_args()
    if a.cmd in ("setup", "chip", "seed", "issues") and rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if a.cmd == "setup":
        cmd_setup(a)
    elif a.cmd == "chip":
        cmd_chip(a)
    elif a.cmd == "task":
        cmd_task(a, rest)
    elif a.cmd == "pr":
        cmd_pr(a, rest)
    elif a.cmd == "pr-local":
        cmd_pr_local(a, rest)
    elif a.cmd == "seed":
        run([sys.executable, PR_DRIVER, "seed", a.dir])
    elif a.cmd == "issues":
        run([sys.executable, PR_DRIVER, "issues"] + (["--repo", a.repo] if a.repo else []))


if __name__ == "__main__":
    main()
