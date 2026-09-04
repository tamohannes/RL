# SciProbe checks resource server

This NeMo Gym resource server grades allowlisted SciProbe probes without
exposing checker or reference artifacts to the model's Python sandbox. The
model sees only `/workspace/sciprobe-probe/data`. The verifier maps the opaque
row `probe_id` to a private probe root, verifies the configured checker and data
hashes, copies only `data/`, `checks.py`, and the submission into a fresh temp,
and runs that exact checker in a separate process. It returns reward `1.0` only
when every execution-grounded check passes.
