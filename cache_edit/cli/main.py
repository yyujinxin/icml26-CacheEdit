"""Main entry point for the cache-edit CLI."""

import argparse
import sys
from typing import List, Optional

from cache_edit.cli.commands.edit import add_edit_parser, run_edit
from cache_edit.cli.commands.benchmark import add_benchmark_parser, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    """构建 cache-edit 命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="cache-edit",
        description="CacheEdit - Efficient image editing with intelligent "
        "caching for diffusion models (Qwen / Flux Kontext).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  cache-edit edit --model flux --image in.png --prompt 'add hat' "
            "--output out.png\n"
            "  cache-edit benchmark --model qwen --config configs/qwen_default.yaml\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version="cache-edit 0.1.0"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="COMMAND",
        help="Subcommand to run",
    )
    subparsers.required = False  # 允许只打印 help

    add_edit_parser(subparsers)
    add_benchmark_parser(subparsers)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI 入口。

    Args:
        argv: 参数列表（不含程序名）。None 时使用 sys.argv[1:]。

    Returns:
        退出码（0 表示成功）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "edit":
            return run_edit(args)
        elif args.command == "benchmark":
            return run_benchmark(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
