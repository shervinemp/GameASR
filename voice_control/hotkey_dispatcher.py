from pynput import keyboard
from typing import Dict, Set, Callable, Union, TypeAlias, ContextManager

HotkeyAction: TypeAlias = Union[
    Callable[[], ContextManager[None]], Callable[[], None]
]


class HotkeyDispatcher:
    def __init__(self) -> None:
        """Initializes the hotkey dispatcher."""
        self.hotkeys: Dict[frozenset, HotkeyAction] = {}
        self.active_contexts: Dict[frozenset, ContextManager[None]] = {}
        self.pressed_keys: Set[keyboard.Key | keyboard.KeyCode] = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

    def register(self, hotkey_string: str, action: HotkeyAction) -> None:
        """Registers a hotkey with a specific action."""
        hotkey = frozenset(keyboard.HotKey.parse(hotkey_string))
        self.hotkeys[hotkey] = action

    def unregister(self, hotkey_string: str) -> None:
        """Unregisters a hotkey."""
        hotkey = frozenset(keyboard.HotKey.parse(hotkey_string))
        if hotkey in self.hotkeys:
            del self.hotkeys[hotkey]

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key is None:
            return
        self.pressed_keys.add(key)
        for hotkey, action in self.hotkeys.items():
            if (
                hotkey.issubset(self.pressed_keys)
                and hotkey not in self.active_contexts
            ):
                context_manager = action()
                if isinstance(context_manager, ContextManager):
                    context_manager.__enter__()
                    self.active_contexts[hotkey] = context_manager

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        for hotkey, context_manager in list(self.active_contexts.items()):
            if not hotkey.issubset(self.pressed_keys):
                context_manager.__exit__(None, None, None)
                del self.active_contexts[hotkey]

        if key in self.pressed_keys:
            self.pressed_keys.remove(key)

    def start(self) -> None:
        """Starts the keyboard listener."""
        self.listener.start()

    def stop(self) -> None:
        """Stops the keyboard listener."""
        self.listener.stop()
        self.listener.join()
