"""Non-concurrency S1-03 RED cases; unsafe detection reuses S0B-02."""

from __future__ import annotations


class NaiveStorage:
    def publish(self) -> None:
        return None

    def read(self) -> bytes:
        return b"partial"

    def list_payloads(self) -> list[str]:
        return ["payload-1", "payload-2"]


failures: list[str] = []
surface = {name for name in dir(NaiveStorage) if not name.startswith("_")}
if surface != {"publish", "read"}:
    failures.append("STOR-01/FS-01/INV-04: naive interface exposes payload history listing")

record = {
    "schema_version": 1,
    "run_identity": "run-a",
    "fragment_map_identity": "map-a",
    "fragment_identity": "fragment-0",
    "version": 1,
    "sequence": 1,
    "dtype": "float32",
    "shape": [4],
    "payload_bytes": 16,
    "payload_sha256": "wrong",
    "base_content_identity": "base-a",
}
payload = NaiveStorage().read()
if payload and record:
    failures.append("FS-02/03/07/PROP-02/A-PROP-01: naive reader accepts partial bytes and wrong checksum")

if failures:
    raise AssertionError("\n".join(failures))
