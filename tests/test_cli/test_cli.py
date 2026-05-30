"""Tests for cache_edit.cli (argparse wiring only — no model loads)."""

import pytest

from cache_edit.cli import build_parser, main


class TestBuildParser:
    def test_parser_constructible(self):
        p = build_parser()
        assert p.prog == "cache-edit"

    def test_help_does_not_crash(self, capsys):
        p = build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "cache-edit" in out
        assert "edit" in out
        assert "benchmark" in out

    def test_version_flag(self, capsys):
        p = build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--version"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "0.1.0" in out


class TestEditParser:
    def test_required_args(self):
        p = build_parser()
        args = p.parse_args([
            "edit", "--model", "flux",
            "--image", "in.png", "--prompt", "hello",
        ])
        assert args.command == "edit"
        assert args.model == "flux"
        assert str(args.image) == "in.png"
        assert args.prompt == "hello"
        assert str(args.output) == "output.png"  # default

    def test_invalid_model_rejected(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["edit", "--model", "sdxl", "--image", "x", "--prompt", "y"])

    def test_no_cache_flag(self):
        p = build_parser()
        args = p.parse_args([
            "edit", "--model", "qwen",
            "--image", "in.png", "--prompt", "p",
            "--no-cache",
        ])
        assert args.no_cache is True


class TestBenchmarkParser:
    def test_defaults(self):
        p = build_parser()
        args = p.parse_args([
            "benchmark", "--model", "flux",
            "--image", "x.png", "--prompt", "p",
        ])
        assert args.command == "benchmark"
        assert args.rounds == 3
        assert args.warmup == 1
        assert args.seed == 0


class TestMainEntry:
    def test_no_args_prints_help_returns_0(self, capsys):
        rc = main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "cache-edit" in out

    def test_version_via_main(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
