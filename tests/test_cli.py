import pytest
from unittest.mock import patch
from nemesis_sim.cli import main

def test_cli_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "NEMESIS" in captured.out

def test_cli_basic_run(tmp_path):
    out_file = tmp_path / "out.bin"
    # minimal run: 1ms duration, no noise, embedded ephemeris
    args = ["--dur", "1.0", "--no-noise", "--out", str(out_file), "--summary"]
    
    assert main(args) == 0
    assert out_file.exists()

def test_cli_attacks(tmp_path):
    out_file = tmp_path / "attack.bin"
    # test meaconing
    args = ["--dur", "1.0", "--attack", "meaconing", "--meacon-delay", "1e-4", "--out", str(out_file)]
    assert main(args) == 0
    
    # test slow drift
    args = ["--dur", "1.0", "--attack", "slow_drift", "--drift-rate", "2.0", "--out", str(out_file)]
    assert main(args) == 0
    
    # test adversarial
    args = ["--dur", "1.0", "--attack", "adversarial", "--false-lat", "7.0", "--out", str(out_file)]
    assert main(args) == 0

def test_cli_gui_flag():
    # If we pass --gui, it runs run_gui. We should mock run_gui so it doesn't block.
    with patch("nemesis_sim.gui.server.run_gui") as mock_run_gui:
        assert main(["--gui"]) == 0
        mock_run_gui.assert_called_once()
