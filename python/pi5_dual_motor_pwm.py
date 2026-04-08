#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

try:
    from gpiozero import PWMOutputDevice
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "gpiozero is required. Install with: pip install gpiozero lgpio"
    ) from exc

try:
    from gpiozero.pins.lgpio import LGPIOFactory
except Exception:
    LGPIOFactory = None


# Pi 5 RP1 hardware PWM-capable GPIO pins.
PI5_PWM0_CH0_GPIO = 18
PI5_PWM0_CH1_GPIO = 19
PI5_PWM1_CH0_GPIO = 12
PI5_PWM1_CH1_GPIO = 13
PI5_ALLOWED_PWM_PINS = {
    PI5_PWM0_CH0_GPIO,
    PI5_PWM0_CH1_GPIO,
    PI5_PWM1_CH0_GPIO,
    PI5_PWM1_CH1_GPIO,
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def validate_pi5_pwm_pins(left_pins: "MotorPins", right_pins: "MotorPins") -> None:
    all_pins = [
        left_pins.forward_pwm,
        left_pins.reverse_pwm,
        right_pins.forward_pwm,
        right_pins.reverse_pwm,
    ]

    if len(set(all_pins)) != 4:
        raise ValueError(
            "All four PWM pins must be unique. "
            f"Got: {all_pins}"
        )

    invalid = [p for p in all_pins if p not in PI5_ALLOWED_PWM_PINS]
    if invalid:
        raise ValueError(
            "Pi 5 hardware PWM pins must be chosen from GPIO 12, 13, 18, 19. "
            f"Invalid: {invalid}"
        )


def duty_from_voltage(
    target_voltage: float,
    supply_voltage: float = 12.0,
    max_duty: float = 0.5,
) -> float:
    """
    Convert target motor voltage to PWM duty ratio.

    Example: with 12V supply and 7.2V target, duty is 0.6. This project caps
    duty at 0.5 for hardware safety, so the effective command becomes 0.5.
    """
    if supply_voltage <= 0:
        raise ValueError("supply_voltage must be > 0")
    return clamp(target_voltage / supply_voltage, 0.0, max_duty)


@dataclass
class MotorPins:
    forward_pwm: int
    reverse_pwm: int


class PwmHBridgeMotor:
    """
    A single DC motor controlled by two PWM pins (H-bridge style).

    Direction coding:
    - 1: forward (forward_pwm gets duty, reverse_pwm = 0)
    - 0: reverse (reverse_pwm gets duty, forward_pwm = 0)
    """

    def __init__(
        self,
        pins: MotorPins,
        *,
        frequency_hz: float = 1000.0,
        max_duty: float = 0.5,
        pin_factory=None,
    ) -> None:
        if max_duty <= 0:
            raise ValueError("max_duty must be > 0")
        self.max_duty = min(max_duty, 0.5)
        self._forward = PWMOutputDevice(
            pin=pins.forward_pwm,
            active_high=True,
            initial_value=0.0,
            frequency=frequency_hz,
            pin_factory=pin_factory,
        )
        self._reverse = PWMOutputDevice(
            pin=pins.reverse_pwm,
            active_high=True,
            initial_value=0.0,
            frequency=frequency_hz,
            pin_factory=pin_factory,
        )

    def set(self, direction: int, duty: float) -> None:
        if direction not in (0, 1):
            raise ValueError("direction must be 0 or 1")

        safe_duty = clamp(float(duty), 0.0, self.max_duty)

        # Turn both sides off first to avoid overlap shoot-through.
        self._forward.value = 0.0
        self._reverse.value = 0.0

        if safe_duty <= 0:
            return
        if direction == 1:
            self._forward.value = safe_duty
        else:
            self._reverse.value = safe_duty

    def stop(self) -> None:
        self._forward.value = 0.0
        self._reverse.value = 0.0

    def close(self) -> None:
        self.stop()
        self._forward.close()
        self._reverse.close()


class DualMotorController:
    def __init__(
        self,
        left_pins: MotorPins,
        right_pins: MotorPins,
        *,
        frequency_hz: float = 1000.0,
        max_duty: float = 0.5,
        force_lgpio: bool = True,
    ) -> None:
        pin_factory = None
        if force_lgpio:
            if LGPIOFactory is None:
                raise RuntimeError(
                    "LGPIO backend unavailable. Install with: pip install lgpio"
                )
            pin_factory = LGPIOFactory()

        self.max_duty = min(max_duty, 0.5)
        self.left = PwmHBridgeMotor(
            left_pins,
            frequency_hz=frequency_hz,
            max_duty=self.max_duty,
            pin_factory=pin_factory,
        )
        self.right = PwmHBridgeMotor(
            right_pins,
            frequency_hz=frequency_hz,
            max_duty=self.max_duty,
            pin_factory=pin_factory,
        )

    def set_left(self, direction: int, duty: float) -> None:
        self.left.set(direction, duty)

    def set_right(self, direction: int, duty: float) -> None:
        self.right.set(direction, duty)

    def set_both(
        self,
        *,
        left_direction: int,
        left_duty: float,
        right_direction: int,
        right_duty: float,
    ) -> None:
        self.left.set(left_direction, left_duty)
        self.right.set(right_direction, right_duty)

    def set_both_voltage(
        self,
        *,
        left_direction: int,
        left_voltage: float,
        right_direction: int,
        right_voltage: float,
        supply_voltage: float = 12.0,
    ) -> None:
        self.left.set(
            left_direction,
            duty_from_voltage(left_voltage, supply_voltage, self.max_duty),
        )
        self.right.set(
            right_direction,
            duty_from_voltage(right_voltage, supply_voltage, self.max_duty),
        )

    def stop(self) -> None:
        self.left.stop()
        self.right.stop()

    def close(self) -> None:
        self.left.close()
        self.right.close()

    def __enter__(self) -> "DualMotorController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pi 5 dual motor PWM controller (4 PWM pins, duty capped at 50%)."
    )
    parser.add_argument(
        "--left-forward-pin",
        type=int,
        default=PI5_PWM1_CH0_GPIO,
        help="Default GPIO12 (PWM1_CH0)",
    )
    parser.add_argument(
        "--left-reverse-pin",
        type=int,
        default=PI5_PWM1_CH1_GPIO,
        help="Default GPIO13 (PWM1_CH1)",
    )
    parser.add_argument(
        "--right-forward-pin",
        type=int,
        default=PI5_PWM0_CH0_GPIO,
        help="Default GPIO18 (PWM0_CH0)",
    )
    parser.add_argument(
        "--right-reverse-pin",
        type=int,
        default=PI5_PWM0_CH1_GPIO,
        help="Default GPIO19 (PWM0_CH1)",
    )
    parser.add_argument("--frequency-hz", type=float, default=1000.0)
    parser.add_argument(
        "--max-duty",
        type=float,
        default=0.5,
        help="Safety cap. Will never exceed 0.5.",
    )
    parser.add_argument(
        "--force-lgpio",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: force lgpio backend (recommended on Pi 5), 0: auto backend",
    )

    parser.add_argument("--left-dir", type=int, choices=[0, 1], required=True)
    parser.add_argument("--right-dir", type=int, choices=[0, 1], required=True)

    duty_group_left = parser.add_mutually_exclusive_group(required=True)
    duty_group_left.add_argument("--left-duty", type=float)
    duty_group_left.add_argument("--left-voltage", type=float)

    duty_group_right = parser.add_mutually_exclusive_group(required=True)
    duty_group_right.add_argument("--right-duty", type=float)
    duty_group_right.add_argument("--right-voltage", type=float)

    parser.add_argument("--supply-voltage", type=float, default=12.0)
    parser.add_argument(
        "--hold-forever",
        type=int,
        choices=[0, 1],
        default=0,
        help="1: keep output until Ctrl+C, 0: use --hold-seconds behavior",
    )
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    return parser


def _resolve_duty(
    duty: Optional[float],
    voltage: Optional[float],
    supply_voltage: float,
    max_duty: float,
) -> float:
    if duty is not None:
        return clamp(duty, 0.0, max_duty)
    if voltage is not None:
        return duty_from_voltage(voltage, supply_voltage, max_duty)
    raise ValueError("Either duty or voltage must be provided")


def main() -> None:
    args = _build_arg_parser().parse_args()

    left_pins = MotorPins(args.left_forward_pin, args.left_reverse_pin)
    right_pins = MotorPins(args.right_forward_pin, args.right_reverse_pin)
    validate_pi5_pwm_pins(left_pins, right_pins)
    safe_max_duty = min(args.max_duty, 0.5)

    left_duty = _resolve_duty(
        duty=args.left_duty,
        voltage=args.left_voltage,
        supply_voltage=args.supply_voltage,
        max_duty=safe_max_duty,
    )
    right_duty = _resolve_duty(
        duty=args.right_duty,
        voltage=args.right_voltage,
        supply_voltage=args.supply_voltage,
        max_duty=safe_max_duty,
    )

    with DualMotorController(
        left_pins,
        right_pins,
        frequency_hz=args.frequency_hz,
        max_duty=safe_max_duty,
        force_lgpio=bool(args.force_lgpio),
    ) as controller:
        controller.set_both(
            left_direction=args.left_dir,
            left_duty=left_duty,
            right_direction=args.right_dir,
            right_duty=right_duty,
        )

        print(
            "Applied command: "
            f"left(dir={args.left_dir}, duty={left_duty:.3f}), "
            f"right(dir={args.right_dir}, duty={right_duty:.3f}), "
            f"max_duty={safe_max_duty:.3f}"
        )

        if bool(args.hold_forever):
            print("Holding output forever. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                controller.stop()
                print("Motors stopped by Ctrl+C.")
        elif args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
            controller.stop()
            print("Motors stopped.")


if __name__ == "__main__":
    main()
