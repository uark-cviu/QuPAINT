"""Conversation template used by QuPAINT.

Trimmed from the training codebase to the single template the released
checkpoint was tuned with (``qmllm``), in the ChatML / MPT separator style.
"""

import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple


class SeparatorStyle(IntEnum):
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A prompt builder that keeps the exact formatting used during tuning."""

    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: Tuple[str, str] = ("USER", "ASSISTANT")
    messages: List[List[str]] = dataclasses.field(default_factory=list)
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.MPT
    sep: str = "\n"
    sep2: str = None
    stop_str: str = None
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        if self.sep_style is not SeparatorStyle.MPT:
            raise NotImplementedError(f"Unsupported separator style: {self.sep_style}")
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = system_prompt + self.sep
        for role, message in self.messages:
            if message:
                if isinstance(message, tuple):  # (text, image) pairs
                    message = message[0]
                ret += role + message + self.sep
            else:
                ret += role
        return ret

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def copy(self) -> "Conversation":
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self) -> Dict:
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


QUPAINT_SYSTEM_MESSAGE = (
    "You are a helpful and harmless AI assistant developed by the CVIU. You are a "
    "domain-specialized vision-language model for 2D materials flake analysis. Your "
    "primary task is to detect, segment, and verify monolayer flakes of various "
    "materials (e.g., graphene, hBN, MoS₂, WS₂, WSe₂, MoSe₂, BP, etc.) "
    "from microscopy images and auxiliary metadata/signals."
)

conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    if not override and template.name in conv_templates:
        raise AssertionError(f"{template.name} has been registered.")
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    return conv_templates[name].copy()


# The released checkpoint is tuned with `qmllm`; `internvl2_5` is an alias that
# carried the same system message in the training codebase.
for _name in ("qmllm", "internvl2_5"):
    register_conv_template(
        Conversation(
            name=_name,
            system_template="<|im_start|>system\n{system_message}",
            system_message=QUPAINT_SYSTEM_MESSAGE,
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>\n",
        )
    )
