from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

try:
    import serial
except Exception:
    serial = None  # type: ignore

try:
    from serial.tools import list_ports
except Exception:
    list_ports = None  # type: ignore

try:
    from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler
except Exception:
    GroupSyncRead = None  # type: ignore
    GroupSyncWrite = None  # type: ignore
    PacketHandler = None  # type: ignore
    PortHandler = None  # type: ignore


ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_PROFILE_ACCEL = 108
ADDR_PROFILE_VEL = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
LEN_2 = 2
LEN_4 = 4
TORQUE_ON = 1
TORQUE_OFF = 0
OP_MODE_POSITION = 3
TICK_MAX = 4095
DXL_PRESENT_CURRENT_UNIT_MA = 2.69
DEFAULT_BAUDRATE = 57600
DEFAULT_MOTOR_DIRECTIONS = (1, -1, -1, 1)

IMU_PATTERN = re.compile(
    r"IMU(?P<idx>[12])\s*>\s*Q\(\s*"
    r"(?P<qw>[-+]?\d*\.?\d+)\s*,\s*"
    r"(?P<qx>[-+]?\d*\.?\d+)\s*,\s*"
    r"(?P<qy>[-+]?\d*\.?\d+)\s*,\s*"
    r"(?P<qz>[-+]?\d*\.?\d+)\s*\)",
    re.IGNORECASE,
)


def default_config_path() -> Path:
    return Path(__file__).resolve()


def list_serial_ports() -> list[str]:
    if list_ports is None:
        return []
    try:
        return [str(port.device) for port in list_ports.comports()]
    except Exception:
        return []


def clamp_step_10(value: float) -> int:
    rounded = int(round(float(value) / 10.0) * 10)
    return max(0, min(360, rounded))


def invert_u_360(value: float) -> float:
    return round(max(0.0, min(360.0, 360.0 - float(value))), 2)


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def signed16(value: int) -> int:
    value &= 0xFFFF
    if value & 0x8000:
        return -((~value & 0xFFFF) + 1)
    return value


def signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return -((~value & 0xFFFFFFFF) + 1)
    return value


def int_to_le4(value: int) -> list[int]:
    value &= 0xFFFFFFFF
    return [value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF, (value >> 24) & 0xFF]


def deg_to_tick_0_360(deg: float) -> int:
    deg = clamp_float(deg, 0.0, 360.0)
    return clamp_int(int(round(deg * (TICK_MAX / 360.0))), 0, TICK_MAX)


def tick_to_deg_0_360(tick: int, direction: int = +1) -> float:
    tick = clamp_int(signed32(int(tick)), 0, TICK_MAX)
    if int(direction) < 0:
        tick = TICK_MAX - tick
    return float(tick) * (360.0 / float(TICK_MAX))


@dataclass
class QuaternionSample:
    qw: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    updated_at: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.qw, self.qx, self.qy, self.qz)


@dataclass
class RobotPose:
    roll: float = 180.0
    seg1: float = 180.0
    seg2: float = 180.0


@dataclass
class RobotLoads:
    load1_ma: Optional[float] = None
    load2_ma: Optional[float] = None


@dataclass
class HardwareSnapshot:
    captured_at: float
    roll: float
    seg1: float
    seg2: float
    load1_ma: Optional[float]
    load2_ma: Optional[float]
    imu1: QuaternionSample
    imu2: QuaternionSample


@dataclass(frozen=True)
class DxlConfig:
    device_name: str
    baudrate: int = DEFAULT_BAUDRATE
    protocol_version: float = 2.0
    id1_slider: int = 1
    id2_roll: int = 2
    id3_seg1: int = 3
    id4_seg2: int = 4


class ImuSerialBridge:
    def __init__(self, on_log: Callable[[str], None]) -> None:
        self.on_log = on_log
        self.port_name = ""
        self.baudrate = 115200
        self._ser = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.imu1 = QuaternionSample()
        self.imu2 = QuaternionSample()

    def connect(self, port_name: str, baudrate: int = 115200) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")
        self.disconnect()
        self.port_name = str(port_name).strip()
        self.baudrate = int(baudrate)
        if not self.port_name:
            raise RuntimeError("IMU port is empty.")
        self._ser = serial.Serial(self.port_name, self.baudrate, timeout=0.2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="error_collector-imu-reader", daemon=True)
        self._thread.start()
        self.on_log(f"IMU connected: {self.port_name} @ {self.baudrate}")

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def snapshot(self) -> tuple[QuaternionSample, QuaternionSample]:
        with self._lock:
            return (
                QuaternionSample(*self.imu1.as_tuple(), updated_at=self.imu1.updated_at),
                QuaternionSample(*self.imu2.as_tuple(), updated_at=self.imu2.updated_at),
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._ser is None:
                    break
                raw = self._ser.readline()
                if not raw:
                    continue
                self._handle_line(raw.decode("utf-8", errors="ignore").strip())
            except Exception as exc:
                self.on_log(f"IMU read error: {exc}")
                time.sleep(0.5)

    def _handle_line(self, text: str) -> None:
        matches = list(IMU_PATTERN.finditer(text))
        if not matches:
            return
        now = time.time()
        with self._lock:
            for match in matches:
                sample = QuaternionSample(
                    qw=float(match.group("qw")),
                    qx=float(match.group("qx")),
                    qy=float(match.group("qy")),
                    qz=float(match.group("qz")),
                    updated_at=now,
                )
                if int(match.group("idx")) == 1:
                    self.imu1 = sample
                else:
                    self.imu2 = sample


class Dynamixel4DofDriver:
    def __init__(self, cfg: DxlConfig, motor_directions: tuple[int, int, int, int] = DEFAULT_MOTOR_DIRECTIONS) -> None:
        if PortHandler is None or PacketHandler is None or GroupSyncRead is None or GroupSyncWrite is None:
            raise RuntimeError("dynamixel_sdk is not installed.")
        self.cfg = cfg
        self.ids = [cfg.id1_slider, cfg.id2_roll, cfg.id3_seg1, cfg.id4_seg2]
        self.pose_ids = [cfg.id2_roll, cfg.id3_seg1, cfg.id4_seg2]
        self.direction: Dict[int, int] = {
            cfg.id1_slider: -1 if int(motor_directions[0]) < 0 else 1,
            cfg.id2_roll: -1 if int(motor_directions[1]) < 0 else 1,
            cfg.id3_seg1: -1 if int(motor_directions[2]) < 0 else 1,
            cfg.id4_seg2: -1 if int(motor_directions[3]) < 0 else 1,
        }
        self.port = PortHandler(cfg.device_name)
        self.packet = PacketHandler(cfg.protocol_version)
        self.sync_write_pos = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, LEN_4)
        self.sync_read_pos = GroupSyncRead(self.port, self.packet, ADDR_PRESENT_POSITION, LEN_4)

    def open(self) -> None:
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port: {self.cfg.device_name}")
        if not self.port.setBaudRate(self.cfg.baudrate):
            raise RuntimeError(f"Failed to set baudrate: {self.cfg.baudrate}")
        self.sync_read_pos.clearParam()
        for dxl_id in self.pose_ids:
            if not self.sync_read_pos.addParam(dxl_id):
                raise RuntimeError(f"sync_read addParam failed: ID={dxl_id}")

    def close(self) -> None:
        try:
            self.port.closePort()
        except Exception:
            pass

    def _write1(self, dxl_id: int, addr: int, value: int) -> None:
        comm, err = self.packet.write1ByteTxRx(self.port, dxl_id, addr, value)
        if comm != 0:
            raise RuntimeError(f"[ID {dxl_id}] write1 comm fail: {self.packet.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID {dxl_id}] write1 dxl error: {self.packet.getRxPacketError(err)}")

    def _write4(self, dxl_id: int, addr: int, value: int) -> None:
        comm, err = self.packet.write4ByteTxRx(self.port, dxl_id, addr, value)
        if comm != 0:
            raise RuntimeError(f"[ID {dxl_id}] write4 comm fail: {self.packet.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID {dxl_id}] write4 dxl error: {self.packet.getRxPacketError(err)}")

    def torque_off_all(self) -> None:
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)

    def torque_on_all(self) -> None:
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

    def set_operating_modes(self) -> None:
        self.torque_off_all()
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_OPERATING_MODE, OP_MODE_POSITION)

    def set_profiles(self) -> None:
        for dxl_id in self.ids:
            profile_vel = 150 if dxl_id in (self.cfg.id1_slider, self.cfg.id2_roll) else 120
            self._write4(dxl_id, ADDR_PROFILE_VEL, profile_vel)
            self._write4(dxl_id, ADDR_PROFILE_ACCEL, 5)

    def deg_to_goal_tick(self, dxl_id: int, deg: float) -> int:
        tick = deg_to_tick_0_360(deg)
        if self.direction.get(dxl_id, +1) < 0:
            tick = TICK_MAX - tick
        return clamp_int(tick, 0, TICK_MAX)

    def get_present_positions(self) -> Dict[int, int]:
        comm = self.sync_read_pos.txRxPacket()
        if comm != 0:
            raise RuntimeError(f"sync_read comm fail: {self.packet.getTxRxResult(comm)}")
        positions: Dict[int, int] = {}
        for dxl_id in self.pose_ids:
            if not self.sync_read_pos.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_4):
                raise RuntimeError(f"present position unavailable: ID={dxl_id}")
            positions[dxl_id] = signed32(self.sync_read_pos.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_4))
        return positions

    def get_present_position(self, dxl_id: int) -> int:
        raw, comm, err = self.packet.read4ByteTxRx(self.port, dxl_id, ADDR_PRESENT_POSITION)
        if comm != 0:
            raise RuntimeError(f"[ID {dxl_id}] read position comm fail: {self.packet.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID {dxl_id}] read position dxl error: {self.packet.getRxPacketError(err)}")
        return signed32(int(raw))

    def get_present_currents_ma(self) -> Dict[int, float]:
        currents: Dict[int, float] = {}
        for dxl_id in (self.cfg.id3_seg1, self.cfg.id4_seg2):
            raw, comm, err = self.packet.read2ByteTxRx(self.port, dxl_id, ADDR_PRESENT_CURRENT)
            if comm != 0:
                raise RuntimeError(f"[ID {dxl_id}] read current comm fail: {self.packet.getTxRxResult(comm)}")
            if err != 0:
                raise RuntimeError(f"[ID {dxl_id}] read current dxl error: {self.packet.getRxPacketError(err)}")
            currents[dxl_id] = float(signed16(int(raw))) * DXL_PRESENT_CURRENT_UNIT_MA
        return currents

    def sync_set_goal_positions(self, goals_tick: Dict[int, int]) -> None:
        self.sync_write_pos.clearParam()
        for dxl_id, tick in goals_tick.items():
            if not self.sync_write_pos.addParam(dxl_id, int_to_le4(clamp_int(tick, 0, TICK_MAX))):
                raise RuntimeError(f"sync_write addParam failed: ID={dxl_id}")
        comm = self.sync_write_pos.txPacket()
        if comm != 0:
            raise RuntimeError(f"sync_write comm fail: {self.packet.getTxRxResult(comm)}")

    def command_4dof_deg(self, slider_deg: float, roll_deg: float, seg1_deg: float, seg2_deg: float) -> None:
        goals = {
            self.cfg.id1_slider: self.deg_to_goal_tick(self.cfg.id1_slider, slider_deg),
            self.cfg.id2_roll: self.deg_to_goal_tick(self.cfg.id2_roll, roll_deg),
            self.cfg.id3_seg1: self.deg_to_goal_tick(self.cfg.id3_seg1, seg1_deg),
            self.cfg.id4_seg2: self.deg_to_goal_tick(self.cfg.id4_seg2, seg2_deg),
        }
        self.sync_set_goal_positions(goals)


class RobotHardware:
    def __init__(self, on_log: Callable[[str], None], config_path: Optional[Path] = None) -> None:
        self.on_log = on_log
        self.config_path = Path(config_path) if config_path is not None else default_config_path()
        self.driver: Optional[Dynamixel4DofDriver] = None
        self.port_name = ""
        self.connected = False
        self.commanded_pose = RobotPose()
        self.current_linear_u = 180.0
        self.motor_directions = DEFAULT_MOTOR_DIRECTIONS

    def connect(self, port_name: str) -> None:
        self.disconnect()
        port = str(port_name).strip()
        if not port:
            raise RuntimeError("Robot port is empty.")
        driver = Dynamixel4DofDriver(DxlConfig(device_name=port), motor_directions=self.motor_directions)
        driver.open()
        driver.set_operating_modes()
        driver.set_profiles()
        driver.torque_on_all()
        self.driver = driver
        self.port_name = port
        self.connected = True
        self.current_linear_u = self._read_linear_u()
        self.on_log(f"Robot connected: {port}")

    def disconnect(self) -> None:
        if self.driver is not None:
            try:
                self.driver.torque_off_all()
            except Exception:
                pass
            try:
                self.driver.close()
            except Exception:
                pass
        self.driver = None
        self.connected = False

    def send_pose(self, roll: float, seg1: float, seg2: float) -> None:
        self.commanded_pose = RobotPose(roll=float(roll), seg1=float(seg1), seg2=float(seg2))
        if self.driver is None:
            return
        self.current_linear_u = self._read_linear_u()
        self.driver.command_4dof_deg(
            self.current_linear_u,
            invert_u_360(float(roll)),
            float(seg1),
            invert_u_360(float(seg2)),
        )

    def read_pose(self) -> RobotPose:
        if self.driver is None:
            return self.commanded_pose
        ticks = self.driver.get_present_positions()
        cfg = self.driver.cfg
        roll_raw = round(tick_to_deg_0_360(ticks.get(cfg.id2_roll, 0), self.driver.direction.get(cfg.id2_roll, +1)), 2)
        seg1_raw = round(tick_to_deg_0_360(ticks.get(cfg.id3_seg1, 0), self.driver.direction.get(cfg.id3_seg1, +1)), 2)
        seg2_raw = round(tick_to_deg_0_360(ticks.get(cfg.id4_seg2, 0), self.driver.direction.get(cfg.id4_seg2, +1)), 2)
        pose = RobotPose(
            roll=round(invert_u_360(roll_raw), 2),
            seg1=round(seg1_raw, 2),
            seg2=round(invert_u_360(seg2_raw), 2),
        )
        self.commanded_pose = pose
        return pose

    def _read_linear_u(self) -> float:
        if self.driver is None:
            return float(self.current_linear_u)
        try:
            cfg = self.driver.cfg
            tick = self.driver.get_present_position(cfg.id1_slider)
            linear = round(
                tick_to_deg_0_360(tick, self.driver.direction.get(cfg.id1_slider, +1)),
                2,
            )
            self.current_linear_u = float(linear)
        except Exception:
            pass
        return float(self.current_linear_u)

    def read_loads(self) -> RobotLoads:
        if self.driver is None:
            return RobotLoads()
        currents = self.driver.get_present_currents_ma()
        cfg = self.driver.cfg
        return RobotLoads(load1_ma=currents.get(cfg.id3_seg1), load2_ma=currents.get(cfg.id4_seg2))

    def snapshot(self, imu_bridge: "ImuHardware") -> HardwareSnapshot:
        now = time.time()
        pose = self.read_pose()
        loads = self.read_loads()
        imu1, imu2 = imu_bridge.snapshot()
        return HardwareSnapshot(
            captured_at=now,
            roll=pose.roll,
            seg1=pose.seg1,
            seg2=pose.seg2,
            load1_ma=loads.load1_ma,
            load2_ma=loads.load2_ma,
            imu1=imu1,
            imu2=imu2,
        )


class ImuHardware:
    def __init__(self, on_log: Callable[[str], None]) -> None:
        self.bridge = ImuSerialBridge(on_log)
        self.connected = False

    def connect(self, port_name: str, baudrate: int = 115200) -> None:
        self.bridge.connect(port_name, int(baudrate))
        self.connected = True

    def disconnect(self) -> None:
        self.bridge.disconnect()
        self.connected = False

    def snapshot(self) -> tuple[QuaternionSample, QuaternionSample]:
        return self.bridge.snapshot()
