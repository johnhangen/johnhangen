import pytest

from wingspan_rl.cli import main


def test_demo_command(capsys):
    assert main(["--players", "2", "--seed", "1", "demo", "--agents", "greedy,random"]) == 0
    out = capsys.readouterr().out
    assert "game over" in out
    assert "winner(s)" in out


def test_benchmark_command(capsys):
    assert main(["--players", "2", "--seed", "0", "benchmark",
                 "--agents", "greedy,random", "--games", "3"]) == 0
    out = capsys.readouterr().out
    assert "3 games" in out
    assert "seat 0" in out


def test_unknown_agent_spec_is_rejected():
    with pytest.raises(ValueError):
        main(["demo", "--agents", "nonsense"])
