from __future__ import annotations

from app.input.keyboard import KeyboardDriver


class TestKeyboardDriver:
    def test_dry_run_mode(self):
        driver = KeyboardDriver(dry_run=True)
        assert driver.dry_run

    def test_dry_run_key_down(self):
        driver = KeyboardDriver(dry_run=True)
        assert driver.key_down("A")

    def test_dry_run_key_up(self):
        driver = KeyboardDriver(dry_run=True)
        assert driver.key_up("A")

    def test_dry_run_release_all(self):
        driver = KeyboardDriver(dry_run=True)
        released = driver.release_all()
        assert len(released) == 6

    def test_unknown_key_returns_false(self):
        driver = KeyboardDriver(dry_run=True)
        assert not driver.key_down("X")
        assert not driver.key_up("X")

    def test_dry_run_toggle(self):
        driver = KeyboardDriver(dry_run=True)
        driver.dry_run = False
        assert not driver.dry_run
