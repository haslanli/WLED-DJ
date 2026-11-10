import os
from pathlib import Path
import pwd

from wled_dj.config import ControllerConfig, Settings, app_support_dir, load, logs_dir, save, settings_path


def test_missing_file_gives_defaults(tmp_path):
    assert load(tmp_path / "none.json") == Settings()


def test_round_trip(tmp_path):
    s = Settings(controllers=[ControllerConfig("192.168.1.50", "Wall")], look="sync-check",
                 master=.5, trim_ms={"uid-a": -20.0})
    save(s, tmp_path / "s.json")
    assert load(tmp_path / "s.json") == s


def test_out_of_range_values_are_clamped_and_bad_types_default(tmp_path):
    (tmp_path / "s.json").write_text('{"master": 7, "trim_ms": {"x": 999}, "look": 3}')
    s = load(tmp_path / "s.json")
    assert (s.master, s.trim_ms, s.look) == (1.0, {"x": 150.0}, "sync-check")


def test_corrupt_json_is_preserved(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    assert load(p) == Settings() and (tmp_path / "settings.json.bad").exists()


def test_save_is_atomic_and_private(tmp_path):
    p = tmp_path / "s.json"
    save(Settings(), p)
    assert oct(p.stat().st_mode & 0o777) == "0o600" and not list(tmp_path.glob("*.tmp"))


def test_bad_controller_entries_are_dropped(tmp_path):
    (tmp_path / "s.json").write_text('{"controllers": [{"host": "10.0.0.2"}, {"name": "no host"}, 5]}')
    assert load(tmp_path / "s.json").controllers == [ControllerConfig("10.0.0.2", "")]


def test_valid_host():
    from wled_dj.config import valid_host
    assert all(valid_host(h) for h in ("192.168.1.50", "wled-living.local", "wled", "10.0.0.255"))
    assert not any(valid_host(h) for h in ("192.168.1.50:80", "bad host", "", "http://x", "a" * 300, "-x.local",
                                           "192.168.1.300", " 192.168.1.50", "x..y"))


def test_invalid_saved_hosts_are_dropped(tmp_path):
    (tmp_path / "s.json").write_text('{"controllers": [{"host": "192.168.1.50:80"}, {"host": "10.0.0.2"}]}')
    assert load(tmp_path / "s.json").controllers == [ControllerConfig("10.0.0.2", "")]


def test_stems_setting(tmp_path):
    path = tmp_path / "settings.json"
    assert load(path).stems is True
    save(Settings(stems=False), path)
    assert load(path).stems is False
    path.write_text('{"stems": "yes"}')
    assert load(path).stems is True


def test_no_test_can_reach_the_real_home():
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)       # the account's home, whatever HOME says
    for place in (app_support_dir(), logs_dir(), settings_path()):
        assert real_home not in place.parents, f"{place} is in the real home"


def test_flash_limit_is_saved_and_loaded(tmp_path):
    path = tmp_path / "settings.json"
    assert load(path).flash_limit == 3
    for limit in (0, 1, 2, 3):
        save(Settings(flash_limit=limit), path)
        assert load(path).flash_limit == limit
    for bad in ("4", "-1", "2.5", "true", '"3"', "null"):
        path.write_text('{"flash_limit": %s}' % bad)
        assert load(path).flash_limit == 3, bad
