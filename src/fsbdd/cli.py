from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import freeze_config, load_config
from .evidence import EvidencePackage, validate_package
from .requirements import validate_matrix


_ROLES = ("bootstrap", "learner", "syncer", "checker")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fsbdd")
    commands = parser.add_subparsers(dest="command", required=True)
    for role in _ROLES:
        role_parser = commands.add_parser(role)
        role_parser.add_argument("--dry-run", action="store_true", required=True)
    config = commands.add_parser("config")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    resolve = config_commands.add_parser("resolve")
    resolve.add_argument("source", type=Path)
    resolve.add_argument("destination", type=Path)
    requirements = commands.add_parser("requirements")
    requirements.add_argument("loop_card", type=Path)
    requirements.add_argument("matrix", type=Path)
    evidence = commands.add_parser("evidence")
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    initialize = evidence_commands.add_parser("init")
    initialize.add_argument("root", type=Path)
    finalize = evidence_commands.add_parser("finalize")
    finalize.add_argument("root", type=Path)
    finalize.add_argument("manifest", type=Path)
    validate = evidence_commands.add_parser("validate")
    validate.add_argument("root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in _ROLES:
        print(json.dumps({"schema_version": 1, "role": args.command, "dry_run": True}, sort_keys=True))
        return 0
    if args.command == "config":
        config = load_config(args.source)
        freeze_config(config, args.destination)
        print(json.dumps({"status": "frozen", "config_sha256": config.digest}, sort_keys=True))
        return 0
    if args.command == "requirements":
        print(json.dumps(validate_matrix(args.loop_card, args.matrix), sort_keys=True))
        return 0
    if args.command == "evidence":
        if args.evidence_command == "init":
            EvidencePackage.create(args.root)
            print(json.dumps({"status": "building", "root": str(args.root)}, sort_keys=True))
            return 0
        if args.evidence_command == "finalize":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            summary = EvidencePackage(args.root).finalize(manifest)
            print(json.dumps(summary, sort_keys=True))
            return 0
        summary = validate_package(args.root)
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["status"] == "admissible" else 2
    raise AssertionError("unreachable")


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
