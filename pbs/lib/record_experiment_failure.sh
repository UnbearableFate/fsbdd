FSBDD_EXPERIMENT_FAILURE_RECORDED=0

record_experiment_failure() {
  local exit_code="$1"
  local line_number="$2"
  local failed_command="$3"
  trap - ERR
  set +e

  echo "[ERROR] line=$line_number command=$failed_command" >&2

  local project_root="${PROJECT_ROOT:-$PWD}"
  local python="${PYTHON:-${MAIN_ROOT:-$project_root}/.venv/bin/python}"
  local output_root="${RESULT_ROOT:-${OUTPUT_ROOT:-${LOG_ROOT:-$project_root/runtime_runs/failures/${PBS_JOBID:-unknown}}}}"
  local experiment_id="${RUN_ID:-${PBS_JOBNAME:-unknown-experiment}-${PBS_JOBID:-unknown-job}}"
  local role_id="${PBS_ARRAY_INDEX:-${ROLE_INDEX:-}}"
  local job_id="${PBS_JOBID:-unknown-job}"
  local code_commit="${EXPECTED_COMMIT:-}"
  local -a command=(
    "$python" -m fsbdd.auxiliary.experiment_failure capture
    --output-root "$output_root"
    --experiment-id "$experiment_id"
    --exit-code "$exit_code"
    --line-number "$line_number"
    --command "$failed_command"
    --job-id "$job_id"
  )
  if [[ -n "$role_id" ]]; then
    command+=(--role-id "$role_id")
  fi
  if [[ -n "$code_commit" ]]; then
    command+=(--code-commit "$code_commit")
  fi
  if PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" "${command[@]}" >&2; then
    FSBDD_EXPERIMENT_FAILURE_RECORDED=1
  else
    echo "[ERROR] automatic experiment failure capture failed" >&2
  fi
  exit "$exit_code"
}

record_experiment_exit() {
  local exit_code="$1"
  local line_number="$2"
  local failed_command="$3"
  trap - ERR EXIT
  if [[ "$exit_code" -ne 0 && "$FSBDD_EXPERIMENT_FAILURE_RECORDED" -ne 1 ]]; then
    record_experiment_failure "$exit_code" "$line_number" "$failed_command"
  fi
  exit "$exit_code"
}

install_experiment_failure_traps() {
  trap 'record_experiment_failure "$?" "$LINENO" "$BASH_COMMAND"' ERR
  trap 'record_experiment_exit "$?" "$LINENO" "$BASH_COMMAND"' EXIT
}
