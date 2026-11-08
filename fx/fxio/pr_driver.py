"""Issue in, pull request out (docs/fx/pr.md), on a repository you own.

    python pr_driver.py seed DIR                          # the P0c inventory repository, to push as your repo
    python pr_driver.py issues --repo OWNER/NAME          # one issue per P0c task (once)
    python pr_driver.py run N --repo OWNER/NAME --dll .. --elf .. --imports .. --model .. [--chip-file ..]
    python pr_driver.py check SUMMARY.json --repo OWNER/NAME   # the hidden check on the PR branch, again

`run` fetches issue N and main's tree from GitHub with `gh`, writes the job (the prompt, the repository
at that commit, the run id), serves pr_server's handler on 127.0.0.1 inside this process, and runs fx on
the chip (live_chip.py --job) on this machine's GPU. When the run ends it runs the task's hidden check
on the pull request branch as GitHub has it. qwenfx.py fills in the paths.

The handler writes only to --repo (or QWENFX_REPO): a new fx/... branch per run, no force push, no merge.
The commit identity is --author, or your git config's user.name and user.email.

`--ssh HOST --remote-dir DIR` is the two-machine mode the published runs used: the handler stays here,
with `gh`, and fx runs on HOST, which reaches the handler through `ssh -R` to its own 127.0.0.1. HOST
needs DIR set up with this repository, rv32emu.so, the ELF and imports.json in row24/, and a venv.

The prompt is pr_tool.PROMPT: the issue's number, title and body, and one sentence asking fx to open a
pull request with the open_pull_request tool when the change is complete. Nothing else is added.
Credentials stay with `gh`: fx, the job, the prompt and the logs never hold a token.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(FX, "p0c"))
import pr_server as PS  # noqa: E402
import pr_tool as PR  # noqa: E402
import tasks as TK  # noqa: E402

LIVE_CHIP = os.path.join(HERE, "live_chip.py")
SEED_README = ("# fx-sandbox\n\nA tiny inventory library for fx on QwenFX to open pull requests against.\n"
               "Run the tests with `python -m unittest`.\n")

ISSUE_TITLES = {
    "t1_bug": "total() in inventory/money.py gives the wrong sum",
    "t2_rename": "Rename Store.get_count to count",
    "t3_feature": "Add slugify(text) in a new inventory/text.py",
    "t4_config": "Low stock should mean fewer than 10 units, not 5",
    "t5_parse": "parse_line breaks on surrounding whitespace",
}

# the two-machine mode: fx on HOST under nohup, the handler here through ssh -R
BOX_RUN = ("cd {d} && . venv/bin/activate && export HF_HUB_OFFLINE=1 TMPDIR={d}/tmp && "
           "M=$(python -c 'from huggingface_hub import snapshot_download as s; print(s(\"Qwen/Qwen3.5-4B\"))') && "
           "mkdir -p out/pr tmp && "
           "{{ setsid nohup sh -c 'python -u fx/fxio/live_chip.py ./rv32emu.so row24/fx-fastcopy.elf row24/imports.json "
           "\"$0\" out/pr --job jobs/{run}.json --reps {seed} --full-regs --tmp tmp --frames {frames} {units} "
           "{chip} > out/pr/{run}.log 2>&1; echo $? > out/pr/{run}.done' \"$M\" < /dev/null > /dev/null 2>&1 & }} && echo launched")


def gh(*args, inp: dict | None = None) -> dict | list:
    r = subprocess.run(["gh", "api", *args] + (["--input", "-"] if inp is not None else []),
                       input=json.dumps(inp).encode() if inp is not None else None, capture_output=True)
    if r.returncode:
        raise SystemExit(f"gh api {args[0]}: {r.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return json.loads(r.stdout or b"null")


def task_of(body: str) -> str | None:
    """The P0c task whose prompt is this issue's text (its hidden check then applies)."""
    return next((t for t, v in TK.TASKS.items() if v["prompt"].strip() == body.strip()), None)


def seed_files() -> dict[str, bytes]:
    """The repository the issues are written for: P0c's files and a README of its own."""
    files = {p: s.encode() for p, s in TK.COMMON.items()}
    files["README.md"] = SEED_README.encode()
    return files


def cmd_seed(a) -> None:
    """Writes the inventory repository to DIR as a git repository with one commit. Nothing is pushed:
    you create the GitHub repository from it yourself."""
    d = os.path.abspath(a.dir)
    if os.path.exists(d) and os.listdir(d):
        raise SystemExit(f"{d} is not empty")
    for p, b in seed_files().items():
        f = os.path.join(d, p)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "wb").write(b)
    run = lambda *x: subprocess.run(["git", *x], cwd=d, check=True, capture_output=True)  # noqa: E731
    run("init", "-q", "-b", "main")
    run("config", "core.autocrlf", "false")
    run("add", "-A")
    try:
        who = PS.git_identity(None)
    except SystemExit:  # (no git identity configured: a neutral one for this one commit)
        who = ("fx-sandbox", "fx-sandbox@example.invalid")
    run("-c", f"user.name={who[0]}", "-c", f"user.email={who[1]}", "commit", "-qm", "The inventory library")
    print(json.dumps({"seeded": d, "files": len(seed_files()),
                      "next": f"cd {d} && gh repo create OWNER/NAME --private --source . --push"}))


def cmd_issues(a) -> None:
    repo = PS.allowed_repo(a.repo)
    have = {i["title"]: i["number"] for i in gh(f"repos/{repo}/issues?state=all&per_page=100")
            if "pull_request" not in i}
    out = {}
    for t, title in ISSUE_TITLES.items():
        if title in have:
            out[t] = have[title]
            continue
        i = gh("-X", "POST", f"repos/{repo}/issues", inp={"title": title, "body": TK.TASKS[t]["prompt"]})
        out[t] = i["number"]
    print(json.dumps(out))


def ssh(box: str, cmd: str, check: bool = True, timeout: float = 120) -> str:
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", box, cmd], capture_output=True, timeout=timeout)
    if check and r.returncode:
        raise SystemExit(f"ssh {cmd[:60]}: {r.stderr.decode('utf-8', 'replace')[-300:]}")
    return r.stdout.decode("utf-8", "replace")


def hidden_check(git: PS.Git, branch: str, task: str | None, store: dict | None) -> dict:
    """The task's hidden check on the pull request branch, fetched from the remote."""
    git.run("fetch", "-q", git.remote, f"+refs/heads/{branch}:refs/remotes/pr/{branch}")
    head = git.run("rev-parse", f"refs/remotes/pr/{branch}").decode().strip()
    files = git.files(head)
    out = {"branch": branch, "head": head, "files": len(files)}
    if store is not None:
        out["branch_equals_fx_store"] = {p: b.decode("utf-8", "replace") for p, b in files.items()} == store
    if not task:
        out["solved"] = None
        out["why"] = "no hidden check: the issue is not one of the P0c tasks"
        return out
    d = tempfile.mkdtemp(prefix="fxpr-check-")
    for p, b in files.items():
        q = os.path.join(d, p)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        open(q, "wb").write(b)
    r = subprocess.run([sys.executable, "-c", TK.TASKS[task]["check"]], cwd=d, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    out["solved"] = r.returncode == 0
    out["why"] = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    return out


def make_job(issue: dict, number: int, repo: str, base: str, files: dict[str, bytes], port: int, seed: int,
             run_id: str | None = None) -> dict:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = run_id or f"i{number}-{stamp}"
    return {"run_id": run_id, "issue": number, "issue_title": issue["title"], "task": task_of(issue["body"] or ""),
            "repo": repo, "base_sha": base, "prompt": PR.issue_prompt(number, issue["title"], issue["body"] or ""),
            "pr_url": f"http://127.0.0.1:{port}", "seed": seed, "slug": "issue-" + run_id[1:],
            "files": {p: base64.b64encode(b).decode() for p, b in files.items()}}


def live_cmd(a, jp: str, out: str) -> list[str]:
    """live_chip.py on this machine, for the job at jp."""
    cmd = [sys.executable, "-u", LIVE_CHIP, a.dll, a.elf, a.imports, a.model, out, "--job", jp,
           "--reps", str(a.seed), "--full-regs", "--frames", str(a.frames), "--tmp", a.tmp or tempfile.gettempdir()]
    if a.units:
        cmd += ["--units", a.units]
    if a.chip_file:
        cmd += ["--chip-file", a.chip_file]
    return cmd + list(a.live_args or [])


def run_local(job: dict, handler: PS.Handler, cmd: list[str], out: str, port: int) -> int:
    """Serves the handler on 127.0.0.1:port in this process while live_chip runs the job here."""
    os.makedirs(out, exist_ok=True)
    srv = PS.serve(handler, port)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        log = open(os.path.join(out, f"{job['run_id']}.log"), "w")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:  # (the run's log, as it goes)
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return p.wait()
    finally:
        srv.shutdown()


def run_box(a, job: dict, jp: str, handler: PS.Handler, run_id: str) -> None:
    """The two-machine mode: the handler here, fx on a.ssh, reached through ssh -R."""
    port, d = a.port, a.remote_dir
    srv = PS.serve(handler, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def tunnel():
        return subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                                 "-R", f"127.0.0.1:{port}:127.0.0.1:{port}", a.ssh],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    tun = None
    try:
        if '"ok"' in ssh(a.ssh, f"curl -s -m 5 http://127.0.0.1:{port}/health", check=False):
            raise SystemExit(f"{a.ssh}'s port {port} is already forwarded (a stale tunnel?): not starting")
        tun = tunnel()
        for _ in range(30):
            time.sleep(2)
            if tun.poll() is not None:
                raise SystemExit(f"tunnel failed: {tun.stderr.read().decode()[-300:]}")
            if '"ok": true' in ssh(a.ssh, f"curl -s -m 5 http://127.0.0.1:{port}/health", check=False):
                break
        else:
            raise SystemExit("the tunnel did not come up")
        if not a.resume:
            ssh(a.ssh, f"mkdir -p {d}/jobs")
            subprocess.run(["scp", "-q", jp, f"{a.ssh}:{d}/jobs/{run_id}.json"], check=True)
            chip = f"--chip-file {shlex.quote(a.chip_file)}" if a.chip_file else ""
            # (only the job is backgrounded, with its own fds, so this ssh returns at once)
            print(ssh(a.ssh, BOX_RUN.format(d=d, run=run_id, seed=a.seed, chip=chip, frames=a.frames,
                                            units=f"--units {a.units}" if a.units else "")).strip(), flush=True)
        t0 = time.time()
        while True:
            time.sleep(30)
            if tun.poll() is not None:  # the tunnel dropped: bring it back while fx runs
                tun = tunnel()
                print(json.dumps({"tunnel": "reopened", "at_s": round(time.time() - t0)}), flush=True)
            st = ssh(a.ssh, f"cat {d}/out/pr/{run_id}.done 2>/dev/null", check=False).strip()
            if st.isdigit():
                print(json.dumps({"remote_exit": int(st), "wall_s": round(time.time() - t0)}), flush=True)
                break
            if time.time() - t0 > a.timeout:
                raise SystemExit("the run did not finish in time; it is still running there")
    finally:
        if tun is not None and tun.poll() is None:
            tun.terminate()
        srv.shutdown()
    for f in (f"pr_{run_id}.json", f"{run_id}.log"):
        subprocess.run(["scp", "-q", f"{a.ssh}:{d}/out/pr/{f}", os.path.join(a.out, f)], check=False)


def cmd_run(a) -> None:
    repo = PS.allowed_repo(a.repo)
    os.makedirs(a.out, exist_ok=True)
    g = PS.Git(a.cache, f"https://github.com/{repo}.git", PS.GH_CRED, PS.git_identity(a.author))
    if a.resume:  # a run already launched on the other machine: serve, tunnel, wait, check
        job = json.load(open(os.path.join(a.out, f"job_{a.resume}.json")))
        assert job["issue"] == a.issue and job["repo"] == repo, "the job is another issue's"
        a.seed = job["seed"]
    else:
        issue = gh(f"repos/{repo}/issues/{a.issue}")
        if "pull_request" in issue or issue["state"] != "open":
            raise SystemExit(f"#{a.issue} is not an open issue on {repo}")
        base = g.fetch_base()
        job = make_job(issue, a.issue, repo, base, g.files(base), a.port, a.seed)
    run_id = job["run_id"]
    jp = os.path.join(a.out, f"job_{run_id}.json")
    json.dump(job, open(jp, "w"))
    calls = os.path.join(a.out, f"calls_{run_id}.json")
    h = PS.Handler(a.cache, {run_id: {"issue": a.issue, "base": job["base_sha"], "slug": job["slug"]}}, repo,
                   log_path=calls, author=g.author)
    print(json.dumps({"run": run_id, "issue": a.issue, "task": job["task"], "base": job["base_sha"], "repo": repo}),
          flush=True)
    if a.ssh:
        run_box(a, job, jp, h, run_id)
    else:
        rc = run_local(job, h, live_cmd(a, jp, a.out), a.out, a.port)
        print(json.dumps({"live_chip_exit": rc}), flush=True)
    summary = summarize(job, os.path.join(a.out, f"pr_{run_id}.json"), h.calls, g)
    json.dump(summary, open(os.path.join(a.out, f"summary_{run_id}.json"), "w"), indent=1)
    with open(os.path.join(a.out, "runs.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)


def summarize(job: dict, local: str, calls: list[dict], git: PS.Git) -> dict:
    s = {"run_id": job["run_id"], "issue": job["issue"], "task": job["task"], "seed": job["seed"], "repo": job["repo"]}
    s["handler_calls"] = [{k: c.get(k) for k in ("title", "changes", "branch", "commit", "seconds")}
                          | {"ok": c["answer"]["ok"], "text": c["answer"]["text"]} for c in calls]
    opened = [c for c in calls if c["answer"]["ok"]]
    s["pr_url"] = opened[0]["answer"]["url"] if opened else None
    if not os.path.exists(local):
        s["run_file"] = None
        return s
    r = json.load(open(local))
    s.update(stopped=r.get("stopped"), steps=r.get("steps"), store_check=r.get("solved"),
             chip_s=round(sum(v for k, v in r["chip"]["seconds"].items() if k != "text"), 1),
             qwen_s=r["qwen"]["link_s"], chip_tokens=r["chip"]["tokens"], instructions=r["chip"]["instructions"],
             audit={k: r.get("audit", {}).get(k) for k in ("wrong", "chip_tokens", "forced_tokens", "memory_diff",
                                                           "memory_words_compared")},
             emu=r.get("emu"), end=r.get("end"), chip_file=r.get("chip_file"),
             tools_declared=r.get("tools_declared"),
             tool_calls=[x.get("name") if isinstance(x, dict) else x for x in r.get("pr", {}).get("tool_calls_seen", [])],
             pr_tool_calls=r.get("pr", {}).get("calls"), replayed=r.get("pr", {}).get("replayed"),
             elf_sha256=r.get("elf_sha256"))
    if opened:
        s["hidden_check_on_pr_branch"] = hidden_check(git, opened[0]["answer"]["branch"], job["task"], r.get("final_repo"))
    return s


def cmd_check(a) -> None:
    repo = PS.allowed_repo(a.repo)
    s = json.load(open(a.summary))
    br = next(c["branch"] for c in s["handler_calls"] if c["ok"])
    g = PS.Git(a.cache, f"https://github.com/{repo}.git", PS.GH_CRED)
    print(json.dumps(hidden_check(g, br, s["task"], None)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sd = sub.add_parser("seed")
    sd.add_argument("dir")
    i = sub.add_parser("issues")
    r = sub.add_parser("run")
    r.add_argument("issue", type=int)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--author", default=None, help="commit identity 'Name <email>' (default: git config)")
    r.add_argument("--dll", help="rv32emu.so")
    r.add_argument("--elf", help="fx's RV32 ELF (fx-fastcopy.elf)")
    r.add_argument("--imports", help="the ELF's imports.json")
    r.add_argument("--model", help="Qwen3.5-4B's local directory")
    r.add_argument("--units", default="all+", help="hot-code units; '' for none (the chip file must match)")
    r.add_argument("--frames", type=int, default=4096)
    r.add_argument("--chip-file", default=None, help="the chip file for this configuration; none builds the circuit")
    r.add_argument("--tmp", default=None)
    r.add_argument("--live-args", nargs=argparse.REMAINDER, help="more live_chip.py arguments, last")
    r.add_argument("--ssh", default=None, help="two-machine mode: run fx on this ssh host")
    r.add_argument("--remote-dir", default=None, help="two-machine mode: the run directory on that host")
    r.add_argument("--timeout", type=float, default=4 * 3600)
    r.add_argument("--resume", default=None, help="two-machine mode: a RUN_ID already launched")
    r.add_argument("--out", default="runs/pr")
    c = sub.add_parser("check")
    c.add_argument("summary")
    for p in (i, r, c):
        p.add_argument("--repo", default=None, help="your repository, OWNER/NAME (or QWENFX_REPO)")
    for p in (r, c):
        p.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "fxpr-cache.git"))
    a = ap.parse_args(argv)
    if a.cmd == "run" and not a.ssh and not all((a.dll, a.elf, a.imports, a.model)):
        ap.error("run needs --dll --elf --imports --model (qwenfx.py pr fills them in)")
    if a.cmd == "run" and a.ssh and not a.remote_dir:
        ap.error("--ssh needs --remote-dir")
    {"seed": cmd_seed, "issues": cmd_issues, "run": cmd_run, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
