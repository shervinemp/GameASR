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


class MessageList(List):

    def __getitem__(self, key: int | slice) -> Dict[str, str] | List[Dict[str, str]]:
        if isinstance(key, slice):
            return list(map(self.__getitem__, range(len(self))[key]))
        else:
            return Message.asdict(super().__getitem__(key))

    def __setitem__(self, key: int | slice, value: Dict[str, str]):
        if isinstance(key, slice):
            for i in range(len(self))[key]:
                self.__setitem__(i, value)
        else:
            self._message_list[key] = Message.from_dict(value)

    def __iter__(self) -> Iterable[Dict[str, str]]:
        return map(Message.asdict, super().__iter__())


class Conversation:

    def __init__(self):
        self._messages: MessageList = MessageList()
        self._cutoff_idx: int = 0
        self._tools: Dict[str, Tool] = {}

    def add_system_message(self, content: str):
        msg = Message(role=Message.Role.system, content=content)
        self._messages.append(msg)

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
    def messages(self) -> MessageList:
        return self._messages[self._cutoff_idx :]

    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools

    @tools.setter
    def tools(self, tools: Dict[str, Tool] | Iterable[Tool]):
        if not isinstance(tools, dict):
            tools = {t.name: t for t in tools}
        self._tools = tools
