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

    grad_lost: bool = False


# ============================================================
# BASE STATE CLASS
# ============================================================

class RobotState:

    # ============================================
    # EVERY SEARCH STATE MUST HAVE update function
    # ============================================
    
    def update(self, robot):
        raise NotImplementedError


# ============================================================
# LEVY STATE LOGIC
# ============================================================

class LevyState(RobotState):

    """
    Manages changing to other states when in Levy
    Defines the update function, what should the robot do when in this state.
    """

    def __init__(self, s):
        
        # State initialization: When entering levy state, init Levy Algorithm
        # ---------------------------------------------------------
        self.levy = Levy(s)
        
        # Label used for visualization and debugging
        # ---------------------------------------------------------
        self.label = "LEVY"

    def update(self, robot):

        # HIT DETECTED --> Gradient search
        # ---------------------------------------------------------
        if robot.c > robot.c_hit:
            
            # set target to current robot positon
            # ---------------------------------------------------------
            robot.setTarget(TargetResult(
                x_new=robot.pos[0],
                y_new=robot.pos[1],
                rot_new=robot.pos[3]
            ))
            
            # Set target reached to true, this ensures gradient will
            # start properly
            # ---------------------------------------------------------
            robot.target_reached = True 

            # When gradient is entered from levy, reset all belief logic
            # ---------------------------------------------------------
            robot.changeState(
                GradientState(
                    reset=True,
                    robot=robot
                )
            )
            
            return

        # WAIT FOR TARGET --> Not reached, keep moving no state change needed
        # ---------------------------------------------------------
        if not robot.target_reached:
            return

        # GENERATE TARGET --> Target reached, new levy target
        # ---------------------------------------------------------
        result = self.levy.newTarget(robot, robot.field)

        robot.setTarget(result)


# ============================================================
# GRADIENT STATE
# ============================================================

class GradientState(RobotState):

    """
    Manages state changes when in Gradient and generating new targets
    """

    def __init__(self, reset, robot):
        
        # Label for visualization
        # ---------------------------------------------------------
        self.label = "GRADIENT"

        # Must memory be reset, re init
        # ---------------------------------------------------------
        if reset:
            robot.gradient.reset()
        
        self.gradient = robot.gradient

    def update(self, robot):

        # WAIT FOR MOVEMENT --> First go to gradient target
        # ---------------------------------------------------------
        if not robot.target_reached:
            return

        # GENERATE GRADIENT TARGET (Instant Evaluation)
        # ---------------------------------------------------------
        grad_result = self.gradient.newTarget(
            robot,
            robot.field
        )
        
        # LOST PLUME --> Levy
        # ---------------------------------------------------------
        if grad_result.grad_lost:

            robot.changeState(
                LevyState(s=robot.s)
            )

            return
    
        # APPLY TARGET
        # ---------------------------------------------------------
        robot.setTarget(grad_result)


# ============================================================
# ROBOT
# ============================================================

class ROBOT_GA(BaseRobot):

    # Algorithm specific
    def __init__(self,
        s,
        c_hit,
        G_step,
        c_lost,
        c_found,
        **kwargs
    ):

        # Initialize base robot with physics logic
        # ---------------------------------------------------------
        super().__init__(**kwargs)

        # Algo specific properties
        # ---------------------------------------------------------
        self.s = s
        self.G_step = G_step
        self.c_hit = c_hit
        self.c_found = c_found
        self.c_lost = c_lost
        
        # State initialization
        # ---------------------------------------------------------
        self.state = LevyState(s=self.s)
        
        # Initialize data/parameters for gradient state
        # ---------------------------------------------------------
        self.gradient = SimpleGradient(
            G_step_size=self.G_step
        )
        
    def odorFound(self):
                
        # Stop condition: concentration passes found threshold
        # ---------------------------------------------------------
        if self.c > self.c_found:
            self.done = True  
            self.found_odor = True
            self.found_time = self.t


# ============================================================
# LEVY STRATEGY
# ============================================================

class Levy:

    """
    Levy target generation logic
    """

    def __init__(self, s=1):
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


# ============================================================
# SIMPLE GRADIENT ASCENT STRATEGY
# ============================================================

class SimpleGradient:

    """
    Evaluate gradient instantly and adjust heading based on concentration differences
    """

    def __init__(self, G_step_size):
        self.step_size = G_step_size
        
        # History
        # ---------------------------------------------------------
        self.last_c = None

    def newTarget(self, robot, field):

        current_c = robot.c
        
        # PLUME LOST CHECK
        # ---------------------------------------------------------
        if current_c < robot.c_lost:
            return TargetResult(
                x_new=robot.pos[0],
                y_new=robot.pos[1],
                rot_new=robot.pos[3],
                grad_lost=True
            )

        # GRADIENT ASCENT LOGIC
        # ---------------------------------------------------------
        if self.last_c is None:
            # First step after hit: maintain current heading
            theta = np.deg2rad(robot.pos[3])
        else:
            #if decreased concentration, deviate from current heading
            if current_c < self.last_c:
                theta = np.deg2rad(robot.pos[3]) + robot.rng.normal(0, np.pi/6)
            else:
                theta = np.deg2rad(robot.pos[3])

        self.last_c = current_c

        dx = self.step_size * np.cos(theta)
        dy = self.step_size * np.sin(theta)

        # New target
        # ---------------------------------------------------------
        new_x = robot.pos[0] + dx
        new_y = robot.pos[1] + dy

        # BOUNDARY CHECK
        # If out of bound, decrease step size until robot can make step
        # ---------------------------------------------------------
        if not (
            field.x.min() <= new_x <= field.x.max() and
            field.y.min() <= new_y <= field.y.max()
        ):
            scale = 1.0

            # If scale to small, make current postion target
            # ---------------------------------------------------------
            while scale > 0.05:
                
                # Decrease step size slowly
                # ---------------------------------------------------------
                sx = robot.pos[0] + dx * scale
                sy = robot.pos[1] + dy * scale

                # If step fits, return target
                # ---------------------------------------------------------
                if (
                    field.x.min() <= sx <= field.x.max() and
                    field.y.min() <= sy <= field.y.max()
                ):
                    new_x, new_y = sx, sy
                    break

                scale *= 0.5

            # Ultimate fallback: Just turn around to stay inside bounds
            # ---------------------------------------------------------
            if scale <= 0.1:
                
                turn_dir = robot.rng.choice([3 * np.pi / 4, -3 * np.pi / 4])
                theta_new = theta + turn_dir
                
                dx = self.step_size * np.cos(theta_new)
                dy = self.step_size * np.sin(theta_new)
                
                fall_x = robot.pos[0] + dx
                fall_y = robot.pos[1] + dy
                
                
                if (
                    field.x.min() <= fall_x <= field.x.max() and
                    field.y.min() <= fall_y <= field.y.max()
                ):
                    new_x, new_y = fall_x, fall_y
                else:
                    # Ultimate fallback: do 180 degree turn
                    # ---------------------------------------------------------
                    new_theta = theta_new + np.pi
                    theta = new_theta
                    new_x = robot.pos[0] + (self.step_size * 0.5) * np.cos(new_theta)
                    new_y = robot.pos[1] + (self.step_size * 0.5) * np.sin(new_theta)

        # Valid target 
        # ---------------------------------------------------------
        return TargetResult(
            x_new=new_x,
            y_new=new_y,
            rot_new=np.rad2deg(theta),
            grad_lost=False
        )
        
    def reset(self):
        # Reset sampled averages
        # ---------------------------------------------------------
        self.last_c = None