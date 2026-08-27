"""Strict TOML defaults for the command-line applications."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import tomllib
from typing import Any


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common configuration-file option to a root parser."""

    parser.add_argument(
        "--config",
        type=Path,
        help="load CLI defaults from a TOML file; explicit options take precedence",
    )


def reject_legacy_backend_argument(argv: Sequence[str] | None) -> None:
    """Fail with migration guidance for the removed CLI selector."""

    arguments = list(argv) if argv is not None else None
    if arguments is None:
        import sys

        arguments = sys.argv[1:]
    if any(
        argument == "--backend" or argument.startswith("--backend=")
        for argument in arguments
    ):
        raise SystemExit(
            "--backend was removed; PyTorch is now the only compute runtime. "
            "Remove the option and select hardware with --device."
        )


def load_toml_from_argv(
    argv: Sequence[str] | None,
) -> tuple[Path | None, dict[str, Any]]:
    """Read only ``--config`` before the complete parser is evaluated."""

    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    namespace, _ = preliminary.parse_known_args(argv)
    if namespace.config is None:
        return None, {}
    try:
        with namespace.config.open("rb") as stream:
            values = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(
            f"cannot load TOML config {namespace.config}: {error}"
        ) from error
    return namespace.config, values


def table(
    document: dict[str, Any], path: tuple[str, ...]
) -> dict[str, Any]:
    """Return one optional nested TOML table with structural validation."""

    current: Any = document
    for name in path:
        if not isinstance(current, dict):
            raise SystemExit(f"TOML section {'.'.join(path)} must be a table")
        current = current.get(name, {})
    if not isinstance(current, dict):
        raise SystemExit(f"TOML section {'.'.join(path)} must be a table")
    return current


def apply_toml_defaults(
    parser: argparse.ArgumentParser,
    values: dict[str, Any],
    *,
    section: str,
) -> None:
    """Validate and apply flat values to an argparse parser."""

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    scalar_values = {
        key: value
        for key, value in values.items()
        if not isinstance(value, dict)
    }
    if "backend" in scalar_values:
        raise SystemExit(
            f"TOML key {section}.backend was removed; PyTorch is now the only "
            "compute runtime. Remove the key and use device/dtype settings."
        )
    unknown = sorted(set(scalar_values).difference(actions))
    if unknown:
        raise SystemExit(
            f"unknown key(s) in TOML section [{section}]: {', '.join(unknown)}"
        )
    defaults = {
        key: _convert_default(actions[key], value, section)
        for key, value in scalar_values.items()
    }
    parser.set_defaults(**defaults)


def validate_nested_tables(
    values: dict[str, Any], *, allowed: set[str], section: str
) -> None:
    """Reject misspelled or unsupported nested tables."""

    nested = {key for key, value in values.items() if isinstance(value, dict)}
    unknown = sorted(nested.difference(allowed))
    if unknown:
        raise SystemExit(
            f"unknown table(s) below [{section}]: {', '.join(unknown)}"
        )


def validate_root_sections(
    document: dict[str, Any], *, allowed: set[str]
) -> None:
    """Require every top-level value to be a recognized application table."""

    unknown = sorted(set(document).difference(allowed))
    if unknown:
        raise SystemExit(f"unknown TOML root section(s): {', '.join(unknown)}")
    invalid = sorted(
        key for key, value in document.items() if not isinstance(value, dict)
    )
    if invalid:
        raise SystemExit(f"TOML root values must be tables: {', '.join(invalid)}")


def subparser(
    parser: argparse.ArgumentParser, command: str
) -> argparse.ArgumentParser:
    """Return a named argparse subparser."""

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[command]
    raise RuntimeError("parser has no subcommands")


def explicit_subcommand(
    argv: Sequence[str] | None, commands: set[str]
) -> str | None:
    """Locate the required visualization command in the raw argument list."""

    arguments = list(argv) if argv is not None else None
    if arguments is None:
        import sys

        arguments = sys.argv[1:]
    matches = [argument for argument in arguments if argument in commands]
    return matches[-1] if matches else None


def clear_explicit_append_defaults(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> None:
    """Make an explicit repeatable option replace, rather than extend, TOML."""

    arguments = list(argv) if argv is not None else None
    if arguments is None:
        import sys

        arguments = sys.argv[1:]
    for action in parser._actions:
        if isinstance(action, argparse._AppendAction) and any(
            option in arguments for option in action.option_strings
        ):
            parser.set_defaults(**{action.dest: None})


def _convert_default(
    action: argparse.Action, value: Any, section: str
) -> Any:
    label = f"{section}.{action.dest}"
    if isinstance(
        action,
        (argparse._StoreTrueAction, argparse._StoreFalseAction),
    ) or isinstance(action, argparse.BooleanOptionalAction):
        if not isinstance(value, bool):
            raise SystemExit(f"TOML key {label} must be boolean")
        converted = value
    elif action.nargs not in (None, "?") or isinstance(action, argparse._AppendAction):
        if not isinstance(value, list):
            raise SystemExit(f"TOML key {label} must be an array")
        converted = _convert_sequence(action, value, label)
    elif action.type is not None:
        try:
            converted = action.type(value)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"invalid value for TOML key {label}: {error}") from error
    else:
        if action.default is not None and not isinstance(value, type(action.default)):
            expected = type(action.default).__name__
            raise SystemExit(f"TOML key {label} must be {expected}")
        converted = value
    if action.choices is not None and converted not in action.choices:
        choices = ", ".join(str(choice) for choice in action.choices)
        raise SystemExit(f"TOML key {label} must be one of: {choices}")
    return converted


def _convert_sequence(action: argparse.Action, value: list[Any], label: str) -> Any:
    def convert(item: Any) -> Any:
        if action.type is None:
            return item
        try:
            return action.type(item)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"invalid value for TOML key {label}: {error}") from error

    if isinstance(action, argparse._AppendAction) and action.nargs not in (None, "?"):
        if not all(isinstance(row, list) for row in value):
            raise SystemExit(f"TOML key {label} must be an array of arrays")
        if isinstance(action.nargs, int) and any(
            len(row) != action.nargs for row in value
        ):
            raise SystemExit(
                f"each array in TOML key {label} must contain "
                f"{action.nargs} values"
            )
        return [[convert(item) for item in row] for row in value]
    if isinstance(action.nargs, int) and len(value) != action.nargs:
        raise SystemExit(
            f"TOML key {label} must contain {action.nargs} values"
        )
    return [convert(item) for item in value]
