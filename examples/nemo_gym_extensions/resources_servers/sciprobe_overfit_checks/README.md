# SciProbe overfit checks

This isolated NeMo Gym resource server scores a deliberately trivial GRPO
canary. Every valid trace executes the same two stateful Python calls and then
the model samples exactly `A` or `B`. Only `B` is rewarded. Reward is binary
and deterministic; no LLM judge or verifier-side code execution is used.
