#!/usr/bin/env python3
"""Validate the custom QHPI operator definition against an installed QAIRT."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from lxml import etree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qairt",
        type=Path,
        default=Path(os.environ.get("QNN_SDK_ROOT", "/opt/qairt")),
    )
    parser.add_argument(
        "--opdef",
        type=Path,
        default=Path(__file__).with_name("NvFp4LinearQhpiHtp.xml"),
    )
    args = parser.parse_args()
    schema_path = (
        args.qairt
        / "lib/python/qti/aisw/op_package_generator/translator/OpDef.xsd"
    )
    schema = etree.XMLSchema(etree.parse(str(schema_path)))
    schema.assertValid(etree.parse(str(args.opdef)))
    print("PASS: NvFp4Linear QHPI OpDef validates against the QAIRT schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
