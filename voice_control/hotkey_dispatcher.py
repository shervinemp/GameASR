from pynput import keyboard

class HotkeyDispatcher:
    def __init__(self):
        self.hotkeys = {}
        self.pressed_keys = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

    def register(self, hotkey_string: str, target):
        """Registers a hotkey and a target object to be toggled."""
        hotkey = frozenset(keyboard.HotKey.parse(hotkey_string))
        self.hotkeys[hotkey] = {"target": target, "active": False}

    def unregister(self, hotkey_string: str):
        """Unregisters a hotkey."""
        hotkey = frozenset(keyboard.HotKey.parse(hotkey_string))
        if hotkey in self.hotkeys:
            del self.hotkeys[hotkey]

    def _on_press(self, key):
        self.pressed_keys.add(key)
        for hotkey, info in self.hotkeys.items():
            if not info["active"] and hotkey.issubset(self.pressed_keys):
                info["target"].enable()
                info["active"] = True

    def _on_release(self, key):
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        for hotkey, info in self.hotkeys.items():
            if info["active"] and not hotkey.issubset(self.pressed_keys):
                info["target"].disable()
                info["active"] = False

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
