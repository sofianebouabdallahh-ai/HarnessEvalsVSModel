# Disk and Docker budget

A run must never take the host down. This page says what each part of the platform can write, what
bounds it, and what to do when the machine is tight. None of it changes what is measured.

## What a run writes on the host

| Thing | Size | Bound |
|---|---|---|
| `runs/<run_id>/traces.jsonl` | ~110 KB per AgentDojo episode (post-env dump + raw tool I/O), ~5 KB toy, custom suite ~10 KB | `pre_env` is stored as a hash, not a dump; traces are append-only and never duplicated |
| `scores.jsonl`, `cells.json`, `episodes.json` | KBs | — |
| Postgres (`hm load`) | queryable projection only; no traces | one `pgdata` volume, tens of MB per thousand episodes |
| `.venv`, `.tools` | ~0.6 GB + ~0.3 GB | one copy per checkout; delete worktree venvs when a worktree is done |
| Claude Code session transcripts (`~/.claude/projects/<cwd slug>/`, one per `claude_sdk` episode) | ~0.3 MB per episode | removed by the adapter right after the episode; `hm doctor` counts any left behind and `hm clean` removes them (410 / 118 MB were found after the first 400 episodes, before this existed) |

## What a run writes in Docker

Only the custom safety suite creates containers (one per episode). Each container:

- root filesystem **read-only**; the workspace and `/tmp` are **tmpfs** mounts capped at
  `HM_CUSTOM_WORK_MB` (256) and `HM_CUSTOM_TMP_MB` (64) — nothing an agent does can land in Docker's
  writable layers or grow `Docker.raw`
- memory `HM_CUSTOM_MEMORY` (512m, no swap), `HM_CUSTOM_PIDS` (256), `--network none`
- `--log-driver none`: no container logs (tool output is already in the trace)
- labelled `hm.suite=custom`, removed in `snapshot()`, at interpreter exit, and by `hm clean`
- one pulled image, pinned by digest; the suite never builds images, so the build cache never grows

Postgres and Grafana are the only long-lived containers; their images total ~1.4 GB.

Measured on the first full custom-suite run (100 episodes, 2026-10-25): `Docker.raw` actual size
49 GB before and after, host free space 25 GB before and after, 964 KB written under `runs/`, zero
containers and zero Claude Code transcripts left behind.

## Guards

- **Before a run**: `hm run` refuses to start when host free space is under `HM_MIN_FREE_GB`
  (default 5), or when a selected suite needs Docker and the engine does not answer within 20 s.
- **During a run**: when free space drops under `HM_HARD_FLOOR_GB` (default 2) the run stops after the
  current episode, keeps everything written so far, and records `stopped_early` in `manifest.json`.
- `hm doctor` prints host free space, Docker usage by type with reclaimable amounts, and leftover hm
  containers. `hm clean` removes only containers carrying the project label.

## Docker Desktop on macOS

`Docker.raw` is a sparse file whose *apparent* size is the disk-image limit (the whole drive, if no
limit is set) and whose *actual* size only shrinks when Docker reclaims space. Two settings matter:

1. Set a **disk image size** limit in Docker Desktop → Settings → Resources (e.g. 60–80 GB) so Docker
   can never consume the host.
2. Build cache and dangling images from *other* projects are the usual hogs; `docker system df` shows
   the reclaimable amount and `docker builder prune` / `docker image prune` reclaim it. This project
   never runs those for you — they touch other projects' data.

On 2026-10-24 the host was at 202/228 GB before this project existed; Docker held 69 GB of which this
project accounted for ~1.6 GB. The guards above exist so that the custom suite's per-episode containers
can never turn a nearly-full disk into a full one.
