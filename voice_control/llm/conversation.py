from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Dict
from .tools import Tool


@dataclass
class Message:

    class Role(Enum):
        system = "system"
        user = "user"
        assistant = "assistant"
        tool = "tool"

    role: Role
    content: str

    def asdict(self):
        return {"role": self.role.value, "content": self.content}

    @staticmethod
    def from_dict(data: Dict[str, str]):
        return Message(Message.Role(data["role"]), data["content"])


class MessageList(list):
    def __init__(self, other: Iterable | None = None):
        super().__init__()
        if other:
            for item in other:
                self.append(item)

    def append(self, x: Message | Dict[str, str]):
        if isinstance(x, Message):
            super().append(x)
        elif isinstance(x, dict):
            super().append(Message.from_dict(x))
        else:
            raise NotImplementedError

    def __getitem__(
        self, key: int | slice
    ) -> Dict[str, str] | List[Dict[str, str]]:
        if isinstance(key, slice):
            return MessageList(super().__getitem__(key))
        else:
            return Message.asdict(super().__getitem__(key))

    def __setitem__(self, key: int | slice, value: Message | Dict[str, str]):
        if isinstance(key, slice):
            for i in range(len(self))[key]:
                self.__setitem__(i, value)
        else:
            v = (
                value
                if isinstance(Message, value)
                else Message.from_dict(value)
            )
            super().__setitem__(key, v)

    def __iter__(self) -> Iterable[Dict[str, str]]:
        return map(Message.asdict, super().__iter__())

    def __add__(self, other):
        if isinstance(other, list):
            return MessageList(super().__add__(other))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            return MessageList(other.__add__(self))
        return NotImplemented


class Conversation:

    def __init__(self):
        self._messages: MessageList = MessageList()
        self.cutoff_idx: int = 0

        self._system: str = ""
        self._cutoff_idx: int = 0
        self._tools: Dict[str, Tool] = {}
        self._state: Dict[str, Dict] = defaultdict({}, dict)

    def set_system_message(self, content: str):
        self._system = content

    def add_user_message(self, content: str):
        msg = Message(role=Message.Role.user, content=content)
        self._messages.append(msg)

    def add_assistant_message(self, content: str):
        msg = Message(role=Message.Role.assistant, content=content)
        self._messages.append(msg)

    def add_tool_message(self, content: str):
        msg = Message(role=Message.Role.tool, content=content)
        self._messages.append(msg)

    def clear(self):
        self._messages.clear()

    @property
    def system(self) -> Message:
        return Message(role=Message.Role.system, content=self._system)

    @property
    def messages(self) -> MessageList:
        return self._messages[self.cutoff_idx :]

    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools

    @tools.setter
    def tools(self, tools: Dict[str, Tool] | Iterable[Tool]):
        if not isinstance(tools, dict):
            tools = {t.name: t for t in tools}
        self._tools = tools
