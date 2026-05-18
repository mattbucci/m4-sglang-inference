# qwen36 missing-ecosystems retry — JETSAM struck again, harness caught it

## Context

After the first sweep (`qwen36-missing-ecosystems-JETSAM-2026-05-18/`) had
its server reaped ~10 minutes in, this retry launched on a fresh server
with the newly-hardened `run_rollouts.py` per-instance preflight check.
4 instances queued (flask, pytest, requests, xarray — excluding the
already-verified seaborn).

## What happened

- **14:01:29** smoke.sh started; server boot
- **14:02:05** server ready, opencode dispatch begins
- **14:02:06** instance 1 (`pallets__flask-4045`) starts, preflight passes
- **14:03:38** SGLang server's last log entry — **jetsam fired again,
  this time only 1.5 minutes into flask's rollout**
- **14:02:06 - 14:17:09** flask runs against a dying/dead upstream; 5 tool
  calls captured (1 glob + 2 grep + 2 read) before the death, then opencode
  hangs until 900s timeout
- **14:17:09** flask records as 0-byte/rc=124
- **14:17:09** harness moves to instance 2 (`psf__requests-1963`); calls
  `preflight_canary` → `URLError: Connection refused`
- **14:17:09** SERVER_DEAD: aborts sweep cleanly with `rc=-2` recorded
  for requests, `sys.exit(4)`

## Result

| Instance | Wall | Patch | rc | Real? |
|---|:---:|:----:|:---:|:---:|
| `pallets__flask-4045` | 902 s | 0 B | 124 | ✗ (server died mid-run, 5 tool calls in 1.5 min before death) |
| `psf__requests-1963` | 0 s | 0 B | -2 | ✓ caught by preflight (`server_dead=True`) |

The harness fix works exactly as designed. Flask is still contaminated
because the server died DURING its run, not before — and the preflight
only runs at the boundary between instances. To catch mid-instance
death, you'd need to either:

1. Run a single instance per smoke.sh invocation (per-instance fresh
   server, ~30s overhead each)
2. Run a watchdog alongside opencode that polls `/health` every 30s and
   SIGTERMs opencode on server death (mid-instance escape)
3. Lower CTX to 32K to reduce KV-pool memory pressure (might prevent
   jetsam entirely on this hardware)

## What this tells us

**Recurring jetsam at CTX=131K + qwen36 cross-ecosystem workload is the
floor on this hardware as currently configured.** Both sweeps had
identical pre-flight state (55GB+ free+inactive). Both had the server
reaped within the first 2 instances. Timing was 10 min vs 1.5 min —
highly variable, not a deterministic memory budget. Other system
processes (Firefox/Spotify/Steam observed in `ps aux`) presumably move
the jetsam threshold around per minute.

**Real qwen36 cross-ecosystem rate stays at the 7-ecosystem 16/17 =
94.1% figure of record.** Missing-ecosystem coverage limited to
seaborn 1/1 verified.

## Next iteration ideas

- Run 1-instance-per-smoke.sh wrapper for the 4 missing instances
- Try `CTX=32K` to see if smaller KV pool prevents jetsam
- Close memory-hungry apps (Firefox especially) before sweep launch
- Investigate whether jetsam fires consistently at a specific opencode
  context-token threshold (the launch.log showed 30K tokens at the
  moment of death in this retry; 32K-33K in the first sweep)

## Files

`pallets__flask-4045.diff` is empty (server died mid-run before any
write). Kept for the audit trail.
