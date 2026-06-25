# ============================================================
# IMPORTS
# ============================================================

import numpy as np
from dataclasses import dataclass
from A_robots.BaseRobot import BaseRobot


# ============================================================
# TARGET RESULT
# ============================================================
@dataclass
class TargetResult:

    x_new: float
    y_new: float
    rot_new: float
    
# ============================================================
# CLASS OVERVIEW
# ============================================================

"""
Simple levy walk robot algorithm. Robot choose random direction
and path length. Walks towards this target and then generates
a new one. When odor found concentration threshold is reached
robot finds the odor and stops. 
"""
# ============================================================
# BASE STATE CLASS
# ============================================================

class RobotState:

    # ============================================
    # EVERY SEARCH STATE MUST HAVE update function
    # ============================================
    
    def update(self, robot):
        raise NotImplementedError


class ROBOT_LW(BaseRobot):
    
    """
    Initialize LW algorithm robot, which inherets all base robots logic. 
    Then initialze the base robot logic using super()
    """

    def __init__(self, s, c_found, **kwargs):

        # Initialize common robot properties defined in BaseRobot
        # ---------------------------------------------------------
        super().__init__(**kwargs)

        # LW algorithm specific
        # ---------------------------------------------------------
        self.s = s
        self.c_found = c_found

        
        # State initialization
        # ---------------------------------------------------------
        self.state = LevyState(s=self.s)
        
    def odorFound(self):
        """
        Odor is found when concentration is above threshold
        """
        if self.c > self.c_found:
            self.found_odor = True
            self.done = True
            self.found_time = self.t
            
# ============================================================
# LEVY STATE LOGIC
# ============================================================

class LevyState(RobotState):
    
    """
    Manages changing to other states when in Levy, which in this case never happens
    Defines the update function, what should the robot do when in this state.
    """

    def __init__(self, s):
        
        # State initialization
        # ---------------------------------------------------------
        self.levy = Levy(s)
        # Label for visualization
        # ---------------------------------------------------------
        self.label = "LEVY"

    def update(self, robot):

        if not robot.target_reached:
            return

        # Label for visualization
        # ---------------------------------------------------------
        result = self.levy.newTarget(robot, robot.field)

        robot.setTarget(result)
        
# ============================================================
# LEVY STRATEGY
# ============================================================

class Levy:
    
    """
    Levy target generation logic
    """

    def __init__(self, s):

        self.s = s

    def newTarget(self, robot, field):
        
        # Sample random path length and angle
        # ---------------------------------------------------------
        length = robot.rng.pareto(a=2) * self.s
        angle = 2 * np.pi * robot.rng.random()

        # Calculate new target
        # ---------------------------------------------------------
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        new_x = robot.pos[0] + dx
        new_y = robot.pos[1] + dy
        
        # Check if position is out of room bounds, if so recursivly generate new targets
        # ---------------------------------------------------------
        if not (
            field.x.min() <= new_x <= field.x.max() and
            field.y.min() <= new_y <= field.y.max()
        ):
            return self.newTarget(robot, field)

        return TargetResult(
            x_new=new_x,
            y_new=new_y,
            rot_new=np.rad2deg(angle)
        )