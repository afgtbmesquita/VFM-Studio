"""Group raw DIC files into tests, and accept machine data only when the user submits it.

Nothing is read from beside a DIC export. A test is usable only when the user has
submitted both the Vic-3D exports and a machine-data file, and that file has a
row for every submitted DIC frame. Other rows in the machine file are a recording
of frames that were not exported, and they are ignored. ``step_DIC_file_map.csv``
is optional and is used only when the user submits that file too. Everything is
read-only.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from vfmstudio.agents.dic_data.raw_format import RAW_NAME, RawFileInfo, inspect_raw_file
from vfmstudio.io.csv_import import long_path, natural_path_key

STEP_MAP_NAME = "step_DIC_file_map.csv"
SYNC_NAME = "synchronized_DIC_machine_data.csv"

_MACHINE_COLUMN = re.compile(
    r"^Machine_(?:(?P<system>sys\d+)_)?"
    r"(?P<quantity>Fx|Fy|Fz|Tx|Ty|Tz|DeltaX|DeltaY|DeltaZ|DeltaTheta_x|DeltaTheta_y|DeltaTheta_z)"
    r"_(?P<unit>N|Nm|Nmm|mm|deg)_?$"
)
_UNIT_SCALE = {"Nm": 1000.0}
_UNIT_LABEL = {"N": "N", "Nm": "N·mm", "Nmm": "N·mm", "mm": "mm", "deg": "deg"}


@dataclass(frozen=True)
class MachineChannel:
    column: str
    name: str
    kind: str  # force | moment | displacement | rotation
    unit: str
    scale: float


def parse_machine_column(column: str) -> Optional[MachineChannel]:
    match = _MACHINE_COLUMN.match(column.strip())
    if not match:
        return None
    quantity = match.group("quantity")
    system = match.group("system")
    if quantity.startswith("DeltaTheta"):
        kind, short = "rotation", "θ" + quantity[-1]
    elif quantity.startswith("Delta"):
        kind, short = "displacement", "Δ" + quantity[-1].lower()
    elif quantity.startswith("T"):
        kind, short = "moment", quantity
    else:
        kind, short = "force", quantity
    unit = match.group("unit")
    name = f"{system}.{short}" if system else short
    return MachineChannel(
        column=column,
        name=name,
        kind=kind,
        unit=_UNIT_LABEL[unit],
        scale=_UNIT_SCALE.get(unit, 1.0),
    )


@dataclass
class FolderCompanions:
    folder: Path
    step_of_dic: dict[int, int] = field(default_factory=dict)
    time_of_dic: dict[int, float] = field(default_factory=dict)
    channels: list[MachineChannel] = field(default_factory=list)
    machine_rows: dict[int, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    step_map_file: Optional[Path] = None
    sync_file: Optional[Path] = None

    def loads_for(self, dic_number: Optional[int]) -> dict[str, float]:
        """Machine channels at one DIC frame, converted to N, N·mm, mm, deg."""
        if dic_number is None:
            return {}
        return dict(self.machine_rows.get(int(dic_number), {}))

    def channel(self, name: str) -> Optional[MachineChannel]:
        for item in self.channels:
            if item.name == name:
                return item
        return None


def _read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(long_path(path), skipinitialspace=True)
    frame.columns = [str(name).strip().strip('"') for name in frame.columns]
    return frame


def _header_columns(path: Path) -> Optional[tuple[str, ...]]:
    try:
        with open(long_path(path), "r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline()
    except OSError:
        return None
    return tuple(name.strip().strip('"').strip() for name in header.split(","))


def _peek_kind(path: Path) -> Optional[str]:
    """``machine`` or ``step_map`` when the header says so. DIC exports are named, not peeked."""
    columns = _header_columns(path)
    if not columns:
        return None
    names = set(columns)
    if "DIC_file_number" in names and any(parse_machine_column(name) for name in columns):
        return "machine"
    if {"Step", "DIC_file_number"} <= names:
        return "step_map"
    return None


def read_machine_file(path: Path | str) -> FolderCompanions:
    """Read one submitted machine-data file. A bad file is returned with notes, not raised."""
    path = Path(path)
    found = FolderCompanions(folder=path.parent, sync_file=path)
    try:
        table = _read_table(path)
    except Exception as exc:
        found.notes.append(f"{path.name} could not be read ({exc})")
        return found
    if "DIC_file_number" not in table.columns:
        found.notes.append(f"{path.name} has no DIC_file_number column")
        return found
    found.channels = [
        channel for channel in (parse_machine_column(name) for name in table.columns) if channel is not None
    ]
    if not found.channels:
        found.notes.append(f"{path.name} has no machine channel columns (Machine_Tx_Nmm and the like)")
        return found
    numbers = pd.to_numeric(table["DIC_file_number"], errors="coerce")
    keep = numbers.notna() & ~numbers.duplicated()
    converted = pd.DataFrame(
        {
            channel.name: pd.to_numeric(table[channel.column], errors="coerce") * channel.scale
            for channel in found.channels
        }
    ).loc[keep]
    converted.index = numbers[keep].astype(int).to_numpy()
    for number, row in converted.iterrows():
        found.machine_rows[int(number)] = {
            name: float(value) for name, value in row.items() if pd.notna(value)
        }
    if not found.machine_rows:
        found.notes.append(f"{path.name} has no DIC frames with machine values")
    return found


def read_step_map_file(path: Path | str) -> FolderCompanions:
    """Read one submitted step map. Optional: a bad file is noted and does not block a test."""
    path = Path(path)
    found = FolderCompanions(folder=path.parent, step_map_file=path)
    try:
        table = _read_table(path)
    except Exception as exc:
        found.notes.append(f"{path.name} could not be read ({exc})")
        return found
    if not {"Step", "DIC_file_number"} <= set(table.columns):
        found.notes.append(f"{path.name} lacks Step / DIC_file_number columns")
        return found
    numbers = pd.to_numeric(table["DIC_file_number"], errors="coerce")
    steps = pd.to_numeric(table["Step"], errors="coerce")
    times = pd.to_numeric(table["Target_time_s"], errors="coerce") if "Target_time_s" in table.columns else None
    for index in range(len(table)):
        if pd.isna(numbers.iloc[index]) or pd.isna(steps.iloc[index]):
            continue
        number = int(numbers.iloc[index])
        found.step_of_dic[number] = int(steps.iloc[index])
        if times is not None and not pd.isna(times.iloc[index]):
            found.time_of_dic[number] = float(times.iloc[index])
    if not found.step_of_dic:
        found.notes.append(f"{path.name} has no DIC frames")
    return found


def clear_machine(companions: FolderCompanions) -> None:
    companions.channels.clear()
    companions.machine_rows.clear()
    companions.sync_file = None
    companions.notes = [note for note in companions.notes if "machine" not in note.lower() and SYNC_NAME not in note]


def clear_step_map(companions: FolderCompanions) -> None:
    companions.step_of_dic.clear()
    companions.time_of_dic.clear()
    companions.step_map_file = None


def expand_inputs(paths: Iterable[Path | str]) -> list[Path]:
    """Files as given; folders contribute their raw exports (non-recursive)."""
    found: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        path = Path(item)
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(
                (child for child in path.iterdir() if child.is_file() and RAW_NAME.match(child.name)),
                key=natural_path_key,
            )
        else:
            candidates = [path]
        for candidate in candidates:
            key = str(candidate.resolve()).lower()
            if key not in seen:
                seen.add(key)
                found.append(candidate)
    return found


@dataclass
class TestGroup:
    name: str
    files: list[RawFileInfo] = field(default_factory=list)
    companions: FolderCompanions = field(default_factory=lambda: FolderCompanions(folder=Path()))

    @property
    def folders(self) -> list[Path]:
        return sorted({info.path.parent for info in self.files}, key=str)


def order_key(info: RawFileInfo, step_of_dic: Optional[dict[int, int]] = None) -> tuple:
    """Sort by a submitted step map when the caller passes one, otherwise by DIC frame."""
    step = None
    if step_of_dic and info.dic_number is not None:
        step = step_of_dic.get(info.dic_number)
    return (
        0 if step is not None else 1,
        step if step is not None else 0,
        info.dic_number if info.dic_number is not None else 1 << 40,
        natural_path_key(info.path),
    )


def sort_group_files(group: TestGroup) -> None:
    group.files.sort(key=lambda info: order_key(info, group.companions.step_of_dic))


def group_by_folder(infos: Iterable[RawFileInfo]) -> list[TestGroup]:
    groups: dict[Path, TestGroup] = {}
    for info in infos:
        folder = info.path.parent
        group = groups.setdefault(folder, TestGroup(name=folder.name))
        group.files.append(info)
    names: dict[str, int] = {}
    ordered = []
    for folder in sorted(groups, key=lambda item: natural_path_key(item)):
        group = groups[folder]
        group.files.sort(key=order_key)
        count = names.get(group.name, 0)
        names[group.name] = count + 1
        if count:
            group.name = f"{group.name} ({folder.parent.name})"
        ordered.append(group)
    return ordered


def inspect_inputs(paths: Iterable[Path | str]) -> list[RawFileInfo]:
    return [inspect_raw_file(path) for path in expand_inputs(paths)]


def dominant_channel(loads: dict[str, float], companions: FolderCompanions, kind: str) -> Optional[str]:
    names = [item.name for item in companions.channels if item.kind == kind and item.name in loads]
    if not names:
        return None
    return max(names, key=lambda name: abs(loads[name]))


def format_load(info: RawFileInfo, companions: Optional[FolderCompanions] = None) -> str:
    """Short human label of the largest moment and rotation at this frame.

    ``companions`` is the machine file the user submitted for this test. Nothing
    is read from the DIC file's folder.
    """
    if companions is None:
        return ""
    loads = companions.loads_for(info.dic_number)
    if not loads:
        return ""
    parts = []
    for kind in ("moment", "rotation"):
        name = dominant_channel(loads, companions, kind)
        if name is None:
            continue
        channel = companions.channel(name)
        parts.append(f"{name} = {loads[name]:+.3g} {channel.unit if channel else ''}".strip())
    return ", ".join(parts)


def _list_frames(values: Sequence[int], limit: int = 8) -> str:
    shown = [str(value) for value in list(values)[:limit]]
    text = ", ".join(shown)
    extra = len(values) - len(shown)
    if extra:
        text += f" and {extra} more"
    return text


@dataclass(frozen=True)
class FrameMatch:
    """Whether a test's DIC exports and its submitted machine file describe the same frames."""

    ready: bool
    dic_count: int
    machine_count: int
    only_dic: tuple[int, ...] = ()
    only_machine: tuple[int, ...] = ()
    problems: tuple[str, ...] = ()


def frame_match(group: TestGroup) -> FrameMatch:
    """Ready when every submitted DIC frame has a row in the machine file.

    Rows for frames that were not exported are left unread. A DIC export with
    no machine row is still a mismatch.
    """
    companions = group.companions
    dic_numbers = [info.dic_number for info in group.files]
    named = [number for number in dic_numbers if number is not None]
    machine_frames = tuple(sorted(companions.machine_rows))
    problems: list[str] = []
    if any(not info.ok for info in group.files):
        problems.append("DIC exports that failed the format check have to be removed first.")
    if companions.sync_file is None:
        problems.append(
            f"Machine data has not been submitted. Add {SYNC_NAME} for this test."
        )
    elif companions.notes and not companions.machine_rows:
        problems.extend(companions.notes)
    unnamed = [info.path.name for info in group.files if info.dic_number is None]
    if unnamed:
        problems.append(
            "These DIC files have no frame number in the name, so they cannot be matched: "
            + ", ".join(unnamed[:6])
        )
    duplicates = tuple(sorted(number for number, count in Counter(named).items() if count > 1))
    if duplicates:
        problems.append("DIC frame " + _list_frames(duplicates) + " is listed more than once.")
    named_set = set(named)
    only_dic = tuple(sorted(named_set - set(machine_frames)))
    only_machine = tuple(sorted(set(machine_frames) - named_set))
    if companions.sync_file is not None and companions.machine_rows and not unnamed and not duplicates:
        if only_dic:
            problems.append("DIC frames " + _list_frames(only_dic) + " are not in the machine data.")
    if (
        companions.sync_file is not None
        and not only_dic
        and named
        and not duplicates
        and not unnamed
    ):
        moments = [channel.name for channel in companions.channels if channel.kind == "moment"]
        covered = [
            name
            for name in moments
            if all(name in companions.machine_rows.get(number, {}) for number in named)
        ]
        if not covered:
            problems.append("The machine data has no moment channel (Tx, Ty or Tz) on every DIC frame.")
    return FrameMatch(
        ready=not problems and bool(group.files),
        dic_count=len(named),
        machine_count=len(named_set & set(machine_frames)),
        only_dic=only_dic,
        only_machine=only_machine,
        problems=tuple(problems),
    )


@dataclass
class Submission:
    dic_files: list[Path] = field(default_factory=list)
    machine_files: list[Path] = field(default_factory=list)
    step_maps: list[Path] = field(default_factory=list)


def _classify_file(path: Path, *, explicit: bool) -> Optional[str]:
    if RAW_NAME.match(path.name):
        return "dic"
    lowered = path.name.lower()
    if lowered == SYNC_NAME.lower():
        return "machine"
    if lowered == STEP_MAP_NAME.lower():
        return "step_map"
    if path.suffix.lower() == ".csv" and path.is_file():
        peeked = _peek_kind(path)
        if peeked:
            return peeked
    if explicit:
        return "dic"
    return None


def classify_submission(paths: Iterable[Path | str]) -> Submission:
    """Split exactly the paths the user gave. A folder contributes its own files, not its parents'."""
    found = Submission()
    seen: set[str] = set()
    buckets = {"dic": found.dic_files, "machine": found.machine_files, "step_map": found.step_maps}

    def take(path: Path, *, explicit: bool) -> None:
        try:
            key = str(path.resolve()).lower()
        except OSError:
            key = str(path).lower()
        if key in seen:
            return
        kind = _classify_file(path, explicit=explicit)
        if kind is None:
            return
        seen.add(key)
        buckets[kind].append(path)

    for item in paths:
        path = Path(item)
        if path.is_dir():
            children = sorted(
                (child for child in path.iterdir() if child.is_file()),
                key=natural_path_key,
            )
            for child in children:
                take(child, explicit=False)
        else:
            take(path, explicit=True)
    return found


def _dic_frames(group: TestGroup) -> set[int]:
    return {info.dic_number for info in group.files if info.dic_number is not None}


def _single_folder(group: TestGroup) -> Optional[Path]:
    folders = {info.path.parent for info in group.files}
    if len(folders) == 1:
        return next(iter(folders))
    return None


def _copy_machine(dest: FolderCompanions, src: FolderCompanions) -> None:
    dest.channels = list(src.channels)
    dest.machine_rows = {number: dict(values) for number, values in src.machine_rows.items()}
    dest.sync_file = src.sync_file
    dest.folder = src.folder
    dest.notes = list(dict.fromkeys([*dest.notes, *src.notes]))


def _copy_step_map(dest: FolderCompanions, src: FolderCompanions) -> None:
    dest.step_of_dic = dict(src.step_of_dic)
    dest.time_of_dic = dict(src.time_of_dic)
    dest.step_map_file = src.step_map_file
    dest.notes = list(dict.fromkeys([*dest.notes, *src.notes]))


def _pick_index(group: TestGroup, items: Sequence[FolderCompanions], used: set[int], *, machine: bool) -> Optional[int]:
    candidates = [index for index in range(len(items)) if index not in used]
    if not candidates:
        return None
    frames = _dic_frames(group)
    if machine:
        equal = [index for index in candidates if frames and set(items[index].machine_rows) == frames]
    else:
        equal = [index for index in candidates if frames and set(items[index].step_of_dic) == frames]
    if len(equal) == 1:
        return equal[0]
    if equal:
        return None
    if machine and frames:
        covering = [
            index for index in candidates if frames <= set(items[index].machine_rows)
        ]
        if len(covering) == 1:
            return covering[0]
    folder = _single_folder(group)
    same = [index for index in candidates if folder is not None and items[index].folder == folder]
    if len(same) == 1:
        return same[0]
    return None


def assign_companions(
    groups: Sequence[TestGroup],
    machines: Sequence[FolderCompanions],
    step_maps: Sequence[FolderCompanions],
) -> tuple[list[FolderCompanions], list[FolderCompanions]]:
    """Attach submitted files onto tests. Returns the machine files and step maps left over.

    A file already stored on a test is left there. Exact frame-set matches are
    attached first, then a machine file that contains every DIC frame of the
    test (extra rows are ignored), then a file that sits in the test's own
    folder. One leftover file and one test still waiting are attached so a
    mismatch can be shown.
    """
    held_machines = {
        group.companions.sync_file.resolve()
        for group in groups
        if group.companions.sync_file is not None
    }
    held_steps = {
        group.companions.step_map_file.resolve()
        for group in groups
        if group.companions.step_map_file is not None
    }
    fresh_machines = [
        item for item in machines if item.sync_file is None or item.sync_file.resolve() not in held_machines
    ]
    fresh_steps = [
        item for item in step_maps if item.step_map_file is None or item.step_map_file.resolve() not in held_steps
    ]

    def attach(items: list[FolderCompanions], *, machine: bool) -> list[FolderCompanions]:
        used: set[int] = set()
        waiting = [
            group
            for group in groups
            if group.files and (group.companions.sync_file if machine else group.companions.step_map_file) is None
        ]
        for group in waiting:
            choice = _pick_index(group, items, used, machine=machine)
            if choice is None:
                continue
            used.add(choice)
            if machine:
                _copy_machine(group.companions, items[choice])
            else:
                _copy_step_map(group.companions, items[choice])
            sort_group_files(group)
        still = [group for group in waiting if (group.companions.sync_file if machine else group.companions.step_map_file) is None]
        leftover = [index for index in range(len(items)) if index not in used]
        if len(still) == 1 and len(leftover) == 1:
            if machine:
                _copy_machine(still[0].companions, items[leftover[0]])
            else:
                _copy_step_map(still[0].companions, items[leftover[0]])
            sort_group_files(still[0])
            leftover = []
        return [items[index] for index in leftover]

    return attach(fresh_machines, machine=True), attach(fresh_steps, machine=False)


@dataclass
class PreparedSubmission:
    groups: list[TestGroup] = field(default_factory=list)
    unassigned_machines: list[FolderCompanions] = field(default_factory=list)
    unassigned_step_maps: list[FolderCompanions] = field(default_factory=list)

    def refusal_reasons(self) -> list[str]:
        reasons = []
        for machine in self.unassigned_machines:
            name = machine.sync_file.name if machine.sync_file is not None else "A machine file"
            reasons.append(
                f"{name} was submitted but does not belong to one test. "
                "Each test needs its own machine file, with a row for every DIC frame of its exports."
            )
        return reasons


def prepare_submission(paths: Iterable[Path | str]) -> PreparedSubmission:
    """Classify the paths the user gave, group the DIC exports, and attach submitted companions."""
    submission = classify_submission(paths)
    groups = group_by_folder(inspect_raw_file(path) for path in submission.dic_files)
    machines = [read_machine_file(path) for path in submission.machine_files]
    step_maps = [read_step_map_file(path) for path in submission.step_maps]
    loose_machines, loose_steps = assign_companions(groups, machines, step_maps)
    return PreparedSubmission(groups, loose_machines, loose_steps)


def prepare_groups(paths: Iterable[Path | str]) -> list[TestGroup]:
    return prepare_submission(paths).groups
