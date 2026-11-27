"""CLI principal para ejecutar un experimento individual (YAML y/o flags)."""

from __future__ import annotations

import json
from colab.config import build_arg_parser, resolve_config
from colab.experiments import dispatch_task


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = resolve_config(args)
    res = dispatch_task(cfg)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
