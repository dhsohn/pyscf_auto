from __future__ import annotations

from typing import Any

from app_config import load_app_config
from runner.orchestrator import cmd_run_inp as _cmd_run_inp
from runner.orchestrator import cmd_status as _cmd_status


def cmd_run_inp(args: Any) -> int:
    cfg = load_app_config(getattr(args, "config", None))
    return int(
        _cmd_run_inp(
            reaction_dir=args.reaction_dir,
            max_retries=args.max_retries,
            force=args.force,
            json_output=args.json,
            app_config=cfg,
        )
    )


def cmd_status(args: Any) -> int:
    cfg = load_app_config(getattr(args, "config", None))
    return int(
        _cmd_status(
            reaction_dir=args.reaction_dir,
            json_output=args.json,
            app_config=cfg,
        )
    )

