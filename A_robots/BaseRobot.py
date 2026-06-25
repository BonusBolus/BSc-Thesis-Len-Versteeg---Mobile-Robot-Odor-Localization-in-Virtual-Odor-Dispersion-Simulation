# ============================================================
# IMPORTS
# ============================================================
import numpy as np

# ============================================================
# CLASS OVERVIEW
# ============================================================
"""
Base class for robots, here the universal robot logic is defined.
- Movement logic
- Concentration sampling (So different algorithm implementations all sample identically)
- Data is stored
"""

class BaseRobot:

    def __init__(
        self,
        
        seed, field, sim_dt, max_search_time, #Envoriment
        
        start_pos, robot_height, move_speed, turn_speed, #Robot parameters
        
        c_min = 1e-5, sample_interval = 0.5,  percentage = True, interpolate = True, noise = False #Sensing 
        
    ):

        # ============================================
        # ENVIRONMENT
        # ============================================

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.field = field
        self.sim_dt = sim_dt
        self.comsol_dt = field.comsol_dt
        
        self.max_search_time = max_search_time

        # ============================================
        # ROBOT
        # ============================================
        self.robot_height = robot_height
        self.move_speed = move_speed
        self.turn_speed = turn_speed

        self.step_size = self.move_speed * sim_dt
        self.step_angle = turn_speed * 360 * sim_dt
        
        # ============================================
        # ROBOT INITIALIZATTION
        # ============================================
                
        self.state = None
        self.pos = np.array(start_pos)
        self.pos = np.append(self.pos, 0)

        self.target_pos = self.pos.copy()
        self.target_reached = True

        # ============================================
        # SENSOR/SAMPLING
        # ============================================

        self.sample_interval = sample_interval
        self.interpolate = interpolate
        self.percentage = percentage
        self.noise = noise
        self.c_min = c_min
        self.c = 0

        # ============================================
        # TIME
        # ============================================

        self.t = 0
        self.last_sample_step = -100
        self.t_history = 0

        # ============================================
        # EXIT CONDITIONS
        # ============================================
        self.done = False
        self.timeout = False
        self.max_search_time = max_search_time
        self.found_odor = False
        self.found_time = 0

        # ============================================
        # HISTORY
        # ============================================

        self.t_history = []
        self.robot_pos_history = []
        self.c_robot_history = []
        self.sample_history = []
        self.state_history = []
        self.path_length = 0
        self.prev_pos = self.pos[:2].copy()
        
        
    def step(self, visualize=False):
        
        """
        Steps a single robot by first:
        - Checking if robot is finished or has reached time limit
        - Sample sensor data - if S_time has passed
        - Update the state 
        - Move a step
        - Store data
        - Check if odor is classfied as found
        """
        

        # Return if robot is done
        # ---------------------------------------------------------
        if self.done:
            return

        # Update Time
        # ---------------------------------------------------------
        self.t += self.sim_dt

        #Index to check if concentration data needs to be updated
        # ---------------------------------------------------------
        comsol_t_idx = int(self.t / self.comsol_dt)

        # No more COMSOL data check
        # ---------------------------------------------------------
        if comsol_t_idx >= self.field.concentration.shape[1]:
            self.done = True
            return

    
        # Time limit
        # ---------------------------------------------------------
        if self.t >= self.max_search_time:
            self.done = True
            self.timeout = True
            return
        
        # Sample sensor
        # ---------------------------------------------------------
        current_sensor_step = int(self.t // self.sample_interval)
        if current_sensor_step > self.last_sample_step:
            self.last_sample_step = current_sensor_step
            self.sampleConcentration(comsol_t_idx)
            
        # Update State
        # ---------------------------------------------------------
        self.state.update(self)

        # Execute Movement
        # ---------------------------------------------------------
        self.executeMove()
        
        # Odor found check
        # ---------------------------------------------------------
        self.odorFound()
        
        # Data storage
        # ---------------------------------------------------------
        self.updatePathLength()
        # Only need to store data when visualizing
        if visualize:
            self.storeData()
            
            
    def changeState(self, state):
        """
        Change state function utilized by specific robot classes
        """
        self.state = state
    
    ### Function used by all different robot algo classes
    def setTarget(self, result):
        
        """
        Standardized function to set robot target, used by all robot algorithms
        """
        
        self.target_pos[:] = [
            result.x_new,
            result.y_new,
            self.target_pos[2], #Height never changes
            result.rot_new
        ]

        self.target_reached = False
    
    def executeMove(self):
        
        """
        Execute movement
        1. Turn based on turn_speed
        2. If in correct heading move based on move_speed
        """
        
        # ----------------------------------
        # TURNING
        # ----------------------------------

        current_heading = self.pos[3]
        target_heading = self.target_pos[3]

        # Shortest angular difference [-180, 180]
        # ---------------------------------------------------------
        angle_error = ((target_heading - current_heading + 180) % 360) - 180

        # When turn can not be made in one step, move in the direction of the shortest turn
        # ---------------------------------------------------------
        if abs(angle_error) > self.step_angle:

            self.pos[3] += np.sign(angle_error) * self.step_angle
            self.pos[3] %= 360

            self.target_reached = False
            
            return

        # When turn can be made in one step, snap to target
        # ---------------------------------------------------------
        else:
            self.pos[3] = target_heading

        # ----------------------------------
        # Movement
        # ----------------------------------
            
        # Vector to target
        # ---------------------------------------------------------
        dx = self.target_pos[0] - self.pos[0]
        dy = self.target_pos[1] - self.pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        

        # Target reacable in one step --> Snap
        # ---------------------------------------------------------
        if dist <= self.step_size:
            
            self.pos[0] = self.target_pos[0]
            self.pos[1] = self.target_pos[1]
            self.target_reached = True

            
        else:
            
            # Normalize
            # ---------------------------------------------------------
            dx /= dist
            dy /= dist

            # Step update
            # ---------------------------------------------------------
            self.pos[0] += dx * self.step_size
            self.pos[1] += dy * self.step_size
            self.pos[3] = self.target_pos[3]

            self.target_reached = False

    def storeData(self):
        
        """
        Function to instanlty store all relevant data needed for visualization
        """

        self.t_history.append(self.t)
        self.robot_pos_history.append(self.pos.copy())
        self.c_robot_history.append(self.c)
        self.state_history.append(self.state.label)
    

    def updatePathLength(self):
        
        """
        Path lenght must be concstently calculated in order to be 
        saved to results
        """
        
        current_pos = self.pos[:2]

        step_distance = np.linalg.norm(current_pos - self.prev_pos)

        self.path_length += step_distance

        self.prev_pos = current_pos.copy()


    def sampleConcentration(self, t_idx):
        
        """
        Sensor sampling utilizes the concentration data in the comsol Field class
        """

        measured_c = self.field.get_c_sample(
            t_idx=t_idx, #which comsol index
            interpolate=self.interpolate, #interpolate grid concentration data
            x=self.pos[0],
            y=self.pos[1],
            robot_height=self.robot_height
        )
        
        if self.noise:
            
            # Noise increases at bigger concentrations
            # ---------------------------------------------------------
            relative_noise_factor = 2.5 
            proportional_std_dev = measured_c * relative_noise_factor

            # Baseline noise
            # ---------------------------------------------------------
            baseline_std_dev = 5e-4


            total_std_dev = proportional_std_dev + baseline_std_dev

            # 4.Gaussian noise
            noise = self.rng.normal(loc=0.0, scale=total_std_dev)

            # 5. Add to measurement and check your cut-off
            measured_c = measured_c + noise


        # Cut off data below concentration threshold
        # ---------------------------------------------------------
        if measured_c < self.c_min:
            self.c = np.array([0.0])
            
        else:
            
            self.c = measured_c
            
            
    def odorFound(self):
        
        """
        Standard odor is never found and robot never done, 
        robot class needs own metod to detect if odor found
        """
        
        self.done = False
        self.found_odor = False
            
    
    def getFullState(self):
        
        """
        Get full robot state in one function
        """

        return {

            # ----------------------------------------
            # POSITION
            # ----------------------------------------
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "z": float(self.pos[2]),

            "heading": float(self.pos[3]) if len(self.pos) > 3 else None,

            # ----------------------------------------
            # SENSOR
            # ----------------------------------------
            "robot_concentration": getattr(self, "c", None),

            # ----------------------------------------
            # TARGET
            # ----------------------------------------
            "target_x": (
                float(self.target_pos[0])
                if hasattr(self, "target_pos")
                else None
            ),

            "target_y": (
                float(self.target_pos[1])
                if hasattr(self, "target_pos")
                else None
            ),

            # ----------------------------------------
            # SEARCH STATE
            # ----------------------------------------
            "behavior_state": getattr(self.state, "label", None),

            # ----------------------------------------
            # PERFORMANCE
            # ----------------------------------------
            "path_length": self.path_length,
            "done": self.done,
            "found_odor": self.found_odor,
            "timeout": self.timeout,
        }