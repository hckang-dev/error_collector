from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


class ScenarioError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScenarioCommand:
    name: str
    args: tuple
    line_no: int


def parse_scenario_text(text: str) -> List[ScenarioCommand]:
    commands: List[ScenarioCommand] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "start":
            commands.append(ScenarioCommand("start", (), line_no))
            continue
        if line == "end":
            commands.append(ScenarioCommand("end", (), line_no))
            continue
        match = re.fullmatch(r"([A-Za-z_]+)\((.*)\)", line)
        if not match:
            raise ScenarioError(f"line {line_no}: invalid command syntax: {line}")
        name = match.group(1)
        arg_text = match.group(2).strip()
        args = _parse_args(name, arg_text, line_no)
        commands.append(ScenarioCommand(name, args, line_no))
    _validate_scenario(commands)
    return commands


def load_scenario_file(path: str | Path) -> List[ScenarioCommand]:
    target = Path(path).expanduser().resolve()
    return parse_scenario_text(target.read_text(encoding="utf-8"))


def _parse_args(name: str, arg_text: str, line_no: int) -> tuple:
    if name == "move":
        parts = [part.strip() for part in arg_text.split(",")]
        if len(parts) != 3:
            raise ScenarioError(f"line {line_no}: move expects 3 integer arguments")
        return tuple(int(part) for part in parts)
    if name == "delaysec":
        if not arg_text:
            raise ScenarioError(f"line {line_no}: delaysec expects one integer argument")
        return (int(arg_text),)
    if name == "record":
        arg = arg_text.strip()
        if arg not in {"start", "end", "reset"}:
            raise ScenarioError(f"line {line_no}: record expects start, end, or reset")
        return (arg,)
    if name == "export":
        if not arg_text:
            raise ScenarioError(f"line {line_no}: export expects one path argument")
        try:
            value = ast.literal_eval(arg_text)
        except Exception as exc:
            raise ScenarioError(f"line {line_no}: invalid export path") from exc
        if not isinstance(value, str):
            raise ScenarioError(f"line {line_no}: export path must be a string")
        return (value,)
    raise ScenarioError(f"line {line_no}: unsupported command: {name}")


def _validate_scenario(commands: List[ScenarioCommand]) -> None:
    if not commands:
        raise ScenarioError("scenario is empty")
    if commands[0].name != "start":
        raise ScenarioError("scenario must start with start")
    if commands[-1].name != "end":
        raise ScenarioError("scenario must end with end")


class ScenarioRunner:
    def __init__(
        self,
        *,
        on_move: Callable[[int, int, int], None],
        on_record: Callable[[str], None],
        on_export: Callable[[str], None],
        on_log: Callable[[str], None],
    ) -> None:
        self.on_move = on_move
        self.on_record = on_record
        self.on_export = on_export
        self.on_log = on_log
        self.commands: List[ScenarioCommand] = []
        self.index = 0
        self.running = False
        self._delay_until: Optional[float] = None

    def load(self, commands: List[ScenarioCommand]) -> None:
        self.commands = list(commands)
        self.index = 0
        self.running = False
        self._delay_until = None

    def start(self) -> None:
        if not self.commands:
            raise ScenarioError("no scenario loaded")
        self.index = 0
        self.running = True
        self._delay_until = None
        self.on_log("Scenario started")

    def stop(self) -> None:
        if self.running:
            self.on_log("Scenario stopped")
        self.running = False
        self._delay_until = None

    def tick(self) -> None:
        if not self.running:
            return
        now = time.time()
        if self._delay_until is not None:
            if now < self._delay_until:
                return
            self._delay_until = None
            self.index += 1

        while self.running and self.index < len(self.commands):
            command = self.commands[self.index]
            if command.name == "start":
                self.index += 1
                continue
            if command.name == "end":
                self.running = False
                self.on_log("Scenario finished")
                return
            if command.name == "move":
                roll, seg1, seg2 = command.args
                self.on_move(int(roll), int(seg1), int(seg2))
                self.on_log(f"Scenario move -> roll={roll} seg1={seg1} seg2={seg2}")
                self.index += 1
                continue
            if command.name == "record":
                (mode,) = command.args
                self.on_record(str(mode))
                self.on_log(f"Scenario record({mode})")
                self.index += 1
                continue
            if command.name == "export":
                (path,) = command.args
                self.on_export(str(path))
                self.on_log(f"Scenario export({path})")
                self.index += 1
                continue
            if command.name == "delaysec":
                (seconds,) = command.args
                self._delay_until = now + max(0, int(seconds))
                self.on_log(f"Scenario delay {seconds}s")
                return
            raise ScenarioError(f"unexpected command: {command.name}")
