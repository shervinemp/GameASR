"""Tests for hotkey dispatcher."""
import unittest


class TestHotkeyDispatcher(unittest.TestCase):
    def test_register_unregister(self):
        from voice_control.hotkey_dispatcher import HotkeyDispatcher
        d = HotkeyDispatcher()
        results = []
        def cb():
            results.append("fired")
        d.register("<ctrl>+r", cb)
        self.assertEqual(len(d.hotkeys), 1)
        d.unregister("<ctrl>+r")
        self.assertEqual(len(d.hotkeys), 0)
        d.stop()

    def test_multiple_registrations(self):
        from voice_control.hotkey_dispatcher import HotkeyDispatcher
        d = HotkeyDispatcher()
        d.register("<ctrl>+a", lambda: None)
        d.register("<ctrl>+b", lambda: None)
        self.assertEqual(len(d.hotkeys), 2)
        d.stop()

    def test_register_invalid_hotkey(self):
        from voice_control.hotkey_dispatcher import HotkeyDispatcher
        d = HotkeyDispatcher()
        with self.assertRaises(ValueError):
            d.register("invalid!!", lambda: None)
        d.stop()
