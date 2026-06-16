#!/usr/bin/env python3
import numpy as np
from math import pi

from dataclasses import dataclass

@dataclass(slots=True)
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
    
@dataclass(slots=True)
class RobotOdom2D:
    x: float
    y: float
    theta: float

    def __add__(self, other: "RobotOdom2D") -> "RobotOdom2D":
        if not isinstance(other, RobotOdom2D):
            return NotImplemented
        return RobotOdom2D(self.x + other.x, self.y + other.y, self.theta + other.theta)

    def __sub__(self, other: "RobotOdom2D") -> "RobotOdom2D":
        if not isinstance(other, RobotOdom2D):
            return NotImplemented
        return RobotOdom2D(self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __mul__(self, factor) -> "RobotOdom2D":
        if isinstance(factor, (int, float)):
            return RobotOdom2D(self.x * factor, self.y * factor, self.theta * factor)
        return NotImplemented

    def __truediv__(self, factor: float) -> "RobotOdom2D":
        return RobotOdom2D(self.x / factor, self.y / factor, self.theta / factor)



