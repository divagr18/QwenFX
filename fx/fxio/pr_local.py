"""The issue-to-PR flow of pr_driver.py with no GitHub (docs/fx/units-file.md): the same driver code and
pr_server's Handler, against a local bare git repository standing in for your GitHub repository, and a
recorded fake in place of `gh api POST /pulls`. Nothing is pushed anywhere and no PR is opened.

    python fx/fxio/pr_local.py OUT_DIR TASK SEED DLL ELF IMPORTS MODEL [live_chip.py arguments ..]

The bare repository holds pr_driver.py seed's files (P0c's inventory library). The issue is the one
pr_driver.py issues writes for TASK: its title from ISSUE_TITLES, the task's prompt as its body. fx runs on
the chip here (live_chip.py --job); afterwards the fx/ branch on the bare repository gets the task's
hidden check, as pr_driver.py run does on GitHub.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.dirname(HERE)
sys.path[:0] = [HERE, FX, os.path.join(FX, "p0c")]

import pr_driver as PD  # noqa: E402
import pr_server as PS  # noqa: E402
import tasks as TK  # noqa: E402

PORT = 8771
REPO = "you/fx-sandbox"  # (a name only: the remote is the local bare repository)


def git(*a, cwd=None) -> str:
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout.decode()


class FakeGitHub:
    """`gh api POST /pulls`, recorded and never sent."""

    def __init__(self) -> None:
        self.prs = []

    def __call__(self, repo, head, base, title, body):
        self.prs.append({"repo": repo, "head": head, "base": base, "title": title, "body": body})
        return {"url": f"https://example.invalid/{repo}/pull/{len(self.prs)} (not opened: pr_local)",
                "number": len(self.prs)}


def bare_remote(out: str, files: dict[str, bytes]) -> tuple[str, str]:
    bare, w = os.path.join(out, "remote.git"), os.path.join(out, "seed")
    git("init", "--bare", "-q", "-b", "main", bare)
    git("init", "-q", "-b", "main", w)
    git("config", "core.autocrlf", "false", cwd=w)
    for p, b in files.items():
        f = os.path.join(w, p)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "wb").write(b)
    git("add", "-A", cwd=w)
    git("-c", "user.name=t", "-c", "user.email=t@t.invalid", "commit", "-qm", "base", cwd=w)
    git("push", "-q", bare, "main", cwd=w)
    return bare, git("rev-parse", "HEAD", cwd=w).strip()


def main() -> None:
    out, task, seed = sys.argv[1:4]
    dll, elf, imports, model = sys.argv[4:8]
    rest = sys.argv[8:]
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    bare, base = bare_remote(out, PD.seed_files())
    issue = {"title": PD.ISSUE_TITLES[task], "body": TK.TASKS[task]["prompt"]}
    job = PD.make_job(issue, 1, REPO, base, PD.seed_files(), PORT, int(seed), run_id="i1-local")
    jp = os.path.join(out, "job.json")
    json.dump(job, open(jp, "w"))
    gh = FakeGitHub()
    author = ("fx-local", "fx@example.invalid")
    h = PS.Handler(os.path.join(out, "cache.git"), {job["run_id"]: {"issue": 1, "base": base, "slug": job["slug"]}},
                   REPO, remote=bare, create_pr=gh, cred=[], log_path=os.path.join(out, "calls.json"), author=author)
    assert h.git.fetch_base() == base
    cmd = [sys.executable, "-u", PD.LIVE_CHIP, dll, elf, imports, model, out, "--job", jp, "--reps", str(seed), *rest]
    rc = PD.run_local(job, h, cmd, out, PORT)
    res = PD.summarize(job, os.path.join(out, f"pr_{job['run_id']}.json"), h.calls, h.git)
    res.update(live_chip_rc=rc, fake_prs=gh.prs)
    res["branches"] = [ln.split()[1] for ln in git("ls-remote", bare, cwd=out).splitlines() if "refs/heads/fx/" in ln]
    res["main_unchanged"] = git("--git-dir", bare, "rev-parse", "refs/heads/main").strip() == base
    if res["branches"]:
        br = res["branches"][0]
        res["branch_diff"] = git("--git-dir", bare, "diff", "--stat", base, br)
        log = git("--git-dir", bare, "log", "-1", "--format=%P|%an|%ae", br).strip().split("|")
        res["branch_commit"] = {"parent_is_base": log[0] == base, "author": log[1:]}
    json.dump(res, open(os.path.join(out, "pr_local.json"), "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("live_chip_rc", "stopped", "steps", "tool_calls", "pr_url", "branches",
                                              "main_unchanged", "hidden_check_on_pr_branch", "audit", "branch_diff")}),
          flush=True)


if __name__ == "__main__":
    main()
