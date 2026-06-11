#!/usr/bin/env python3
import numpy as np
from math import pi


from dataclasses import dataclass

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

    def __add__(self, other: "Coordinate") -> "Coordinate":
        if not isinstance(other, Coordinate):
            return NotImplemented
        return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Coordinate") -> "Coordinate":
        if not isinstance(other, Coordinate):
            return NotImplemented
        return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, factor) -> "Coordinate":
        if isinstance(factor, (int, float)):
            return Coordinate(self.x * factor, self.y * factor, self.z * factor)
        return NotImplemented

    def __truediv__(self, factor: float) -> "Coordinate":
        return Coordinate(self.x / factor, self.y / factor, self.z / factor)
    
@dataclass 
class State:
    x: float
    y: float
    theta: float

    def __add__(self, other: "State") -> "State":
        if not isinstance(other, State):
            return NotImplemented
        return State(self.x + other.x, self.y + other.y, self.theta + other.theta)

    def __sub__(self, other: "State") -> "State":
        if not isinstance(other, State):
            return NotImplemented
        return State(self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __mul__(self, factor) -> "State":
        if isinstance(factor, (int, float)):
            return State(self.x * factor, self.y * factor, self.theta * factor)
        return NotImplemented

    def __truediv__(self, factor: float) -> "State":
        return State(self.x / factor, self.y / factor, self.theta / factor)


def normalize_angle(angle: float) -> float:
    while abs(angle) > pi:
        if angle > pi:
            angle -= 2*pi
        elif angle < -pi:
            angle += 2*pi

    return angle    