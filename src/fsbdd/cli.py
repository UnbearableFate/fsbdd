from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsbdd.auxiliary.contracts.config import freeze_config, load_config
from fsbdd.auxiliary.contracts.evidence import EvidencePackage, validate_package
from fsbdd.auxiliary.contracts.requirements import validate_matrix


_ROLES = ("bootstrap", "learner", "syncer", "checker")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fsbdd")
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--dry-run", action="store_true")
    bootstrap.add_argument("--plan", type=Path)
    bootstrap.add_argument("--storage-root", type=Path)
    for role in _ROLES[1:]:
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
    if args.command == "bootstrap":
        if args.dry_run:
            if args.plan is not None or args.storage_root is not None:
                raise SystemExit(
                    "bootstrap --dry-run cannot be combined with --plan or --storage-root"
                )
            print(
                json.dumps(
                    {"schema_version": 1, "role": args.command, "dry_run": True},
                    sort_keys=True,
                )
            )
            return 0
        if args.plan is None or args.storage_root is None:
            raise SystemExit("bootstrap requires --plan and --storage-root")
        from fsbdd.diloco.protocol.global_state import (
            GlobalStateStore,
            load_bootstrap_plan,
        )
        from fsbdd.diloco.protocol.storage import PosixStorageBackend

        plan = load_bootstrap_plan(args.plan)
        store = GlobalStateStore(
            PosixStorageBackend(args.storage_root),
            identities=plan.identities,
            descriptors=plan.descriptors,
            s_max=plan.s_max,
        )
        result = store.bootstrap(plan.fragments)
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "role": "bootstrap",
                    "plan_digest": plan.digest,
                    "published_indices": result.published_indices,
                    "existing_indices": result.existing_indices,
                    "version_vector": result.snapshot.version_vector,
                    "snapshot_digest": result.snapshot.digest,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command in _ROLES[1:]:
        print(
            json.dumps(
                {"schema_version": 1, "role": args.command, "dry_run": True},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "config":
        config = load_config(args.source)
        freeze_config(config, args.destination)
        print(
            json.dumps(
                {"status": "frozen", "config_sha256": config.digest}, sort_keys=True
            )
        )
        return 0
    if args.command == "requirements":
        print(json.dumps(validate_matrix(args.loop_card, args.matrix), sort_keys=True))
        return 0
    if args.command == "evidence":
        if args.evidence_command == "init":
            EvidencePackage.create(args.root)
            print(
                json.dumps(
                    {"status": "building", "root": str(args.root)}, sort_keys=True
                )
            )
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
