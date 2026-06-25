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

    c_spiral: bool = False
    waiting: bool = False
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
            # start with sampling
            # ---------------------------------------------------------
            robot.target_reached = True 

            # When gradient is entered from levy, reset all belief logic
            # ---------------------------------------------------------
            robot.changeState(
                GradientState(
                    reset = True,
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
# SPIRAL STATE
# ============================================================

class SpiralState(RobotState):

    """
    Manages state changes when in Spiral and generating new targets when in Spiral state
    """

    def __init__(self, robot):
        
        # Label for visualization
        # ---------------------------------------------------------
        self.label = "SPIRAL"

        # INIT ALGO with center to rotate around
        # ---------------------------------------------------------
        self.spiral = Spiral(
            field=robot.field,
            robot = robot,
            center_x=robot.pos[0],
            center_y=robot.pos[1],
        )

    def update(self, robot):

        # ---------------------------------------------------------
        # HIT DETECTED --> Gradient search - GREEDY PARAMETER used here
        # ---------------------------------------------------------
        if robot.c > robot.c_hit*robot.S_greed:
            
            # set target to current robot positon
            # ---------------------------------------------------------
            robot.setTarget(TargetResult(
                x_new=robot.pos[0],
                y_new=robot.pos[1],
                rot_new=robot.pos[3]
            ))
            
            # Set target reached to true, this ensures gradient will
            # start with sampling
            # ---------------------------------------------------------
            robot.target_reached = True 

            # When gradient is entered from Spiral, reset all belief logic
            # ---------------------------------------------------------
            robot.changeState(
                GradientState(
                    reset = True,
                    robot=robot
                )
            )
            
            return

        # WAIT FOR TARGET
        # ---------------------------------------------------------
        if not robot.target_reached:
            return

        # SPIRAL TARGET
        # ---------------------------------------------------------
        result = self.spiral.newTarget(robot, robot.field)

        # SPIRAL COMPLETE --> Levy
        # ---------------------------------------------------------
        if result.c_spiral:

            robot.changeState(
                LevyState(
                    s=robot.s
                )
            )

            return

        robot.setTarget(result)


# ============================================================
# GRADIENT STATE
# ============================================================

class GradientState(RobotState):

    """
    Manages state changes when in Gradient and generating new targets when in Gradient state
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

        # ENTER SAMPLING IF GRADIENT TARGET REACHED
        # ---------------------------------------------------------
        robot.changeState(
            SamplingState(self.gradient)
        )


# ============================================================
# SAMPLING STATE
# ============================================================

class SamplingState(RobotState):

    """
    Manages state changes when in Sampling and generating new targets when in Spiral state
    """
    
    def __init__(self, gradient):
        
        # Label for visualization
        # ---------------------------------------------------------
        self.label = "SAMPLING"
        
        # Inherit data from gradient state
        # ---------------------------------------------------------
        self.gradient = gradient
        self.sampling = Sampling()

    def update(self, robot):

        # SAMPLE
        # ---------------------------------------------------------
        result = self.sampling.newTarget(
            robot,
            robot.field
        )

        # STILL SAMPLING
        # ---------------------------------------------------------
        if result.waiting:
            return
        
        # SAMPLING DONE --> GENERATE GRADIENT TARGET
        # ---------------------------------------------------------
        grad_result = self.gradient.newTarget(
            robot,
            robot.field
        )
        
        # LOST PLUME --> Spiral
        # ---------------------------------------------------------
        if grad_result.grad_lost:

            robot.changeState(
                SpiralState(robot)
            )

            return
    
        # APPLY TARGET
        # ---------------------------------------------------------
        robot.setTarget(grad_result)

        # SAMPLING DONE --> RETURN TO GRADIENT with memory
        # ---------------------------------------------------------
        robot.changeState(GradientState(
            reset = False, 
            robot=robot
        ))


# ============================================================
# ROBOT
# ============================================================

class ROBOT_BHGS(BaseRobot):

    
    # Algorithm specific
    def __init__(self,
        s, #levy 
        c_hit, G_step, n_angles, #gradient
        S_time, #sampling
        c_w, CF_lost, CF_found, #confidence
        kappa, k, #belief
        R_max, S_greed, #spiral
        **kwargs
    ):

        # Initialize base robot with pysics logic
        # ---------------------------------------------------------
        super().__init__(**kwargs)

        # Algo specific properties
        # ---------------------------------------------------------
        self.s = s
        self.R_max = R_max
        self.S_greed = S_greed
        self.G_step = G_step
        self.n_angles = n_angles
        self.S_time = S_time
        self.c_hit = c_hit
        self.k = k
        self.kappa = kappa
        self.c_w = c_w
        
        self.CF_found = CF_found
        self.CF_lost = CF_lost
        
        # History
        # ---------------------------------------------------------
        self.sample_history = []
        self.confidence = 0
        
        # State initialization
        # ---------------------------------------------------------
        self.state = LevyState(s=self.s)
        
        # Initialize data/parameters for gradient state
        # ---------------------------------------------------------
        self.gradient = BayGradient(
            G_step=self.G_step,
            n_angles=self.n_angles,
            k = self.k,
            kappa=self.kappa,
            c_w = self.c_w,
        )
        
    def odorFound(self):
                
        # Stop condition: very confident odor found
        # ---------------------------------------------------------
        if self.confidence > self.CF_found:
            self.done = True  
            self.found_odor = True
            self.found_time = self.t
             
        pass
            

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
# SPIRAL STRATEGY
# ============================================================

class Spiral:

    """
    Spiral until a hit occurs, or max radius is reached
    """

    def __init__(
        self,
        field,
        robot,
        center_x,
        center_y,
    ):
        
        # Spiral properties
        # ---------------------------------------------------------
        self.cx = center_x
        self.cy = center_y
        self.R_max = robot.R_max

        # Initial radius and growth rate
        # ---------------------------------------------------------
        self.r = 0.0
        self.b = 0.04
        
        # Initial angle and growth rate
        # ---------------------------------------------------------
        self.theta = 0.0
        self.theta_step = 0.01

        # Randomly choose spiral direction
        # ---------------------------------------------------------
        dir = robot.rng.choice([-1, 1])

        self.theta_step = dir*(self.theta_step) 


    def newTarget(self, robot, field):

        step_size = robot.step_size

        # Current position based on spiral values
        # ---------------------------------------------------------
        cur_x = self.cx + self.r * np.cos(self.theta)
        cur_y = self.cy + self.r * np.sin(self.theta)

        # Current angle and radius
        # ---------------------------------------------------------
        t = self.theta
        r = self.r

        # Progress spiral, until robot can reach it in one step
        # Creating small linear path in the spiral
        # ---------------------------------------------------------
        while True:
            
            # Update radius
            # ---------------------------------------------------------
            t += self.theta_step
            r = self.b * t

            # Spiral equation
            # ---------------------------------------------------------
            new_x = self.cx + r * np.cos(t)
            new_y = self.cy + r * np.sin(t)

            # Step distance
            # ---------------------------------------------------------
            dist = np.sqrt((new_x - cur_x)**2 + (new_y - cur_y)**2)

            # Robot can reach new target in one step
            # ---------------------------------------------------------
            if dist >= step_size:
                break
    

        # Robot angle 
        # ---------------------------------------------------------
        heading = np.arctan2(new_y - cur_y, new_x - cur_x)

        # Spiral complete if target out of bounds --> Switch to levy
        # ---------------------------------------------------------
        if not (
            field.x.min() <= new_x <= field.x.max() and
            field.y.min() <= new_y <= field.y.max()
        ):
            return TargetResult(
                x_new=cur_x,
                y_new=cur_y,
                rot_new=np.rad2deg(heading),
                c_spiral=True #complete spiral
            )
        
        # Update spiral parameters
        # ---------------------------------------------------------
        self.theta = t
        self.r = r
    
        # Valid spiral target, update 
        # ---------------------------------------------------------
        return TargetResult(
            x_new=new_x,
            y_new=new_y,
            rot_new=np.rad2deg(heading),
            c_spiral=self.r >= self.R_max
        )

# ============================================================
# GRADIENT STRATEGY
# ============================================================

class BayGradient:

    """
    Do gradient step with size based on confidence. Compare delta_c_avg to update belief of odor direction
    Move towards this direction with new gradient step and some random noise. Transition to sampling after
    which gradient occurs again.
    """

    def __init__(
        self,
        G_step, #gradient 
        n_angles, k, kappa, c_w #belief and confidence
    ):

        self.step_size = G_step
        self.k = k
        self.kappa = kappa
        self.c_w = c_w
        
        # History
        # ---------------------------------------------------------
        self.last_c_average = None
        self.steps = 0

        # Discretize circle
        # ---------------------------------------------------------
        self.n_angles = n_angles
        self.angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        
        # Initial belief in best direction
        # ---------------------------------------------------------
        self.belief = np.ones(n_angles) / n_angles
        self.likelihood = None
        self.likelihood_array = None
        
    def updateBelief(self, delta_c, move_theta):
        
        # Calculate alignment for ALL angles at once
        # ---------------------------------------------------------
        alignment = np.cos(self.angles - move_theta)

        # Calculate the FULL array of likelihoods 
        # ---------------------------------------------------------
        self.likelihood_array = np.exp(self.k * delta_c * alignment)
        
        # Multiply the entire belief array by the likelihood array
        # ---------------------------------------------------------
        self.belief *= self.likelihood_array
        
        # Add epsilon and normalize
        # ---------------------------------------------------------
        self.belief += 1e-12
        self.belief /= np.sum(self.belief)
        
    def chooseDir(self, robot):
        
        # choose angle with best belief + noise
        # ---------------------------------------------------------
        idx = np.argmax(self.belief)
        theta = self.angles[idx]
        theta += robot.rng.vonmises(0, self.kappa)
        
        return theta
    
    def newTarget(self, robot, field):
        
        # get last concentration avg
        # ---------------------------------------------------------
        _, avg = robot.sample_history[-1]
        
        move_theta = np.deg2rad(robot.pos[3])
        
        # if not sampled yet, move in identical direction as hit occcured
        # ---------------------------------------------------------
        if self.last_c_average is None:
            
            theta = move_theta
        
        else:
            # update belief and find best heading
            # ---------------------------------------------------------
            delta_c = avg - self.last_c_average 
            self.updateBelief(delta_c, move_theta) 
            theta = self.chooseDir(robot) 
            
        # ============================================================
        # Confidence & entropy
        # ============================================================
        entropy = -np.sum(self.belief * np.log(self.belief + 1e-12))
        max_entropy = np.log(self.n_angles)
        entropy_norm = entropy / max_entropy
        
        avg = np.clip(avg, 1e-12, None)
        c_norm = np.log(avg / robot.c_hit) / np.log(100 / robot.c_hit)
        c_norm = np.clip(c_norm, 0.0, 1.0)
        
        self.confidence = 1 - entropy_norm + c_norm*self.c_w # 0 = lost, 2 = certain
        
        # Give robot access to confidence - Plotting
        # ---------------------------------------------------------
        robot.confidence = self.confidence
        
        # Decrease step size based on confidence
        # ---------------------------------------------------------
        step = self.step_size * (2 - self.confidence)
        step = max(step, self.step_size * 0.05)  # minimum step so robot doesn't freeze

        dx = step * np.cos(theta)
        dy = step * np.sin(theta)

        # New target
        # ---------------------------------------------------------
        new_x = robot.pos[0] + dx
        new_y = robot.pos[1] + dy

        self.last_c_average = avg

        # If out of bound, decrease step size until robot can make step
        # ---------------------------------------------------------
        if not (
            field.x.min() <= new_x <= field.x.max() and
            field.y.min() <= new_y <= field.y.max()
        ):

            scale = 1.0

            # If scale to small, make current postion target --> resample
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

            # When step size becomes to small, do 135 degree turn
            # ---------------------------------------------------------
            if scale <= 0.1:
                
                best_idx = np.argmax(self.belief)
                best_theta = self.angles[best_idx]
                
                # choose random direction 90 degree offset from current best angle
                # ---------------------------------------------------------
                turn_dir = robot.rng.choice([3 * np.pi / 4, -3 * np.pi / 4])
                theta = best_theta + turn_dir
                
                # Decrease step size based on confidence
                # ---------------------------------------------------------
                step = self.step_size * (2 - self.confidence)
                step = max(step, self.step_size * 0.05)  # minimum step so robot doesn't freeze
                
                dx = step * np.cos(theta)
                dy = step * np.sin(theta)
                
                fall_x = robot.pos[0] + dx
                fall_y = robot.pos[1] + dy

                # Apply target if safe
                # ---------------------------------------------------------
                if (
                    field.x.min() <= fall_x <= field.x.max() and
                    field.y.min() <= fall_y <= field.y.max()
                ):
                    new_x, new_y = fall_x, fall_y
                else:
                    # Ultimate fallback: do 180 degree turn
                    # ---------------------------------------------------------
                    new_theta = best_theta + np.pi
                    theta = new_theta
                    new_x = robot.pos[0] + (self.step_size * 0.5) * np.cos(new_theta)
                    new_y = robot.pos[1] + (self.step_size * 0.5) * np.sin(new_theta)
        
        self.steps += 1
        
        # GRADIENT LOST CONDITION (based on confidence) - always do three grad tries
        # ---------------------------------------------------------
        if self.steps > 2 and self.confidence < robot.CF_lost:
            grad_lost = True
            robot.confidence = 0
        else: 
            grad_lost = False
        
        # Valid target 
        # ---------------------------------------------------------
        return TargetResult(
            x_new=new_x,
            y_new=new_y,
            rot_new=np.rad2deg(theta),
            grad_lost=grad_lost
        )
        
        
    def reset(self):
        # Reset belief and sampled averages
        # ---------------------------------------------------------
        self.last_c_average = None
        self.belief = np.ones(self.n_angles) / self.n_angles  
        self.steps = 0


# ============================================================
# SAMPLING STRATEGY
# ============================================================

class Sampling:

    """
    Wait in place until sampling time passed. Store sampled concentration in history buffer
    """

    def __init__(self):
        
        # History storage
        # ---------------------------------------------------------
        self.sample_buffer = []
        self.sample_t = 0

    def newTarget(self, robot, field):

        # Append concentration
        # ---------------------------------------------------------
        self.sample_buffer.append(robot.c)

        # Check sample time passed
        # ---------------------------------------------------------
        self.sample_t += robot.sim_dt
        if self.sample_t >= robot.S_time:
            
            
            # Save average in global robot variable
            # ---------------------------------------------------------
            avg = np.mean(self.sample_buffer)
            robot.sample_history.append(
                (robot.t, avg)
            )

            return TargetResult(
                x_new=robot.pos[0],
                y_new=robot.pos[1],
                rot_new=robot.pos[3],
                waiting=False
            )

                
        # Still sampling
        # ---------------------------------------------------------
        return TargetResult(
            x_new=robot.pos[0],
            y_new=robot.pos[1],
            rot_new=robot.pos[3],
            waiting=True
        )