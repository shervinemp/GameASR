from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Dict, Tuple
from .tools import Tool


@dataclass
class Message:

    class Role(Enum):
        user = "user"
        assistant = "assistant"
        system = "system"

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
        self.messages = MessageList()

    def add_user_message(self, content: str):
        msg = Message(role=Message.Role.user, content=content)
        self.messages.append(msg)

    def add_assistant_message(self, content: str):
        msg = Message(role=Message.Role.assistant, content=content)
        self.messages.append(msg)

    def add_system_message(self, content: str):
        msg = Message(role=Message.Role.system, content=content)
        self.messages.append(msg)


class SystemPrompt:

    def __init__(
        self,
        prompt: str = "",
        contexts: Tuple[str, ...] = (),
        tools: Dict[str, Tool] | List[Tool] = {},
    ):
        self.prompt = prompt
        self.contexts = contexts
        self.tools = tools

    def __str__(self) -> str:
        parts = []
        if self.prompt:
            parts.append(self.prompt)

        if self.tools:
            tools_string = "\n".join(
                f"<tool>{tool}</tool>" for tool in self.tools.values()
            )
            parts.append(f"\n<tools>\n{tools_string}\n</tools>")

        if self.contexts:
            contexts_string = "\n".join(
                f"<context>{ctx}</context>" for ctx in self.contexts
            )
            parts.append(f"\n{contexts_string}")

        return "\n".join(parts)

    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools

    @tools.setter
    def tools(self, tools: Iterable[Tool] | Dict[str, Tool]):
        if not isinstance(tools, dict):
            tools = {t.name: t for t in tools}
        self._tools = tools
