Ask user for permission if the agent try to change the version of torch.

Agent can install or update other python packages free by using `uv`.

Use subagents with caution; only enable them when absolutely necessary.

For FS-Based Decoupled DiLoCo work, read and enforce
`plans/FS_DILOCO_CODEX_EXECUTION_PLAN/AGENTS.md` before changing any file in
the repository. The execution package is the plan control plane; production
code, tests, runtime configs, PBS scripts, and evidence live at the repository
root unless that contract explicitly says otherwise.
