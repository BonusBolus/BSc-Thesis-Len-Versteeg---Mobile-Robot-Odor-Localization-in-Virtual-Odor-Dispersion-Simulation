# ============================================================
# IMPORTS
# ============================================================

# Classes
# ---------------------------------------------------------
from comsolProccessing import ComsolData  # COMSOL data processing class
from visualizer import Visualizer, ConcHeatMap2D, ConcPoint1D, Room3DView, BeliefPolarPlot  # visualization tools
from simulate import SimRunner, replayRun, batchRun, paramSweep #simulation running options and main class

# Tools
# ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import itertools
import copy
from tqdm import tqdm

run_name = "BHGS_NOISE_FINAL_presentation_final" + datetime.now().strftime("%Y%m%d_%H%M%S")

visualize_live = True # show live simulation
save_vid = False # save animation to file
save_run_results= True   # save reproducability data from run [seed, state]
save_full_run_log = False #full logs + concentration
max_time = 400
# ============================================================
# RUN MODE - SINGLE
# ============================================================
single_run = True
run_name = "BHGS_SINGLE" + datetime.now().strftime("%Y%m%d_%H%M%S")

replay_run = False
replay_path= r"BHGS_NOISe_presentation_final20260622_232446" #folder of the saved run
replay_robot_name = "visual_robot" #name of robot you want to rerun - robots found in robot_seeds and config


# ============================================================
# RUN MODE - BATCH/SWEEP
# ============================================================
batch_run = False
n_runs_batch = 1000
batch_name = "BHGS_BATCH_FINAL" + datetime.now().strftime("%Y%m%d_%H%M%S")


sweep_run = False
n_runs_sweep= 200
algorithm_to_sweep = "BHGS" #algorithm will be swept for multiple fields
sweep_name = "BHGS_SWEEP_FINAL" + datetime.now().strftime("%Y%m%d_%H%M%S")

# Parameters to sweep
# ---------------------------------------------------------
sweep_parameters = {
    "CF_found" : [0.9, 1, 1.1],
    "CF_lost" : [0.2, 0.3, 0.4]
}

# ============================================================
# 1. LOAD COMSOL FIELDS
# ============================================================

fields = {}
percentage = True #convert all mol to percentage (0-100%)

# Give field a name and the path to comsol txt file
# ---------------------------------------------------------
comsol_data = {
    "same_off_middle_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\off\middle\same_off_med_middle.txt",
    "same_off_corner_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\off\corner\same_off_med_corner.txt",
    "same_off_wall_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\off\wall\med\DYNAMIC_odorIntFlow_medVel_switch3_odorPosLow_VentSameOffset_allDatA.txt",
    "same_off_wall_stand": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\off\wall\low\same_off_stand_wall.txt",
    "same_off_wall_high": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\off\wall\high\same_off_high_wall.txt",
    "same_cen_middle_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\cen\middle\same_cen_med_middle.txt",
    "same_cen_corner_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\cen\corner\same_cen_med_corner.txt",
    "same_cen_wall_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\cen\wall\med\DYNAMIC_odorOutFlow_medVel_switch3_odorPosLow_VentSame_allData.txt",
    "same_cen_wall_stand": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\cen\wall\low\same_cen_stand_wall.txt",
    "same_cen_wall_high": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\same\cen\wall\high\same_cen_high_wall.txt",
    "oppo_off_middle_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\off\middle\oppo_off_med_middle.txt",
    "oppo_off_corner_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\off\corner\oppo_off_med_corner.txt",
    "oppo_off_wall_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\off\wall\med\oppo_off_med_wall.txt",
    "oppo_off_wall_stand": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\off\wall\low\oppo_off_stand_wall.txt",
    "oppo_off_wall_high": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\off\wall\high\oppo_off_high_wall.txt",
    "oppo_cen_middle_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\cen\middle\oppo_cen_med_middle.txt",
    "oppo_cen_corner_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\cen\corner\oppo_cen_med_corner.txt",
    "oppo_cen_wall_med": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\cen\wall\med\oppo_cen_med_wall.txt",
    "oppo_cen_wall_stand": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\cen\wall\low\oppo_cen_stand_wall.txt",
    "oppo_cen_wall_high": r"C:\Users\lenve\Documents\Local Personal Len\BSc\COMSOL\COMSOL_V5_FINAL\oppo\cen\wall\high\oppo_cen_high_wall.txt"
}

# Select fields to load in
# ---------------------------------------------------------
if single_run:
    selected_fields = ("oppo_cen_middle_med",)
elif (single_run or replay_run) and (batch_run or sweep_run):
    ValueError("Single run or replay run can not be ran in combination with batch/sweep runs") 
else:
    # Loop over keys in fields to load multiple
    selected_fields = tuple(
        key for key in comsol_data.keys()
        if "med" in key
    )


# Load all fields
# ---------------------------------------------------------
print("============================================================")
print("LOADING COMSOL FIELDS")
print("============================================================")
print(f"Total Fields Selected : {len(selected_fields)}")
print(f"Convert to Percentage : {percentage}")
print("Fields:")
for f_name in selected_fields:
    print(f"  - {f_name}")
print("============================================================")

for name in selected_fields:
    fields[name] = ComsolData(
        comsol_path=comsol_data[name],
        convert_to_percentage=percentage,
        load_npz=True, #load from npz if available
        save_npz=True, #save comsol txt to npz
        clip_neg=True #clip negative concentrations to near zero
    )

# ============================================================
# 2. SIMULATION SETTINGS
# ============================================================

sim_dt = 0.25  # simulation timestep [s] 

robot_physics = {
    "move_speed": 0.33, # m/s
    "turn_speed": 0.4, #rad/s
    "robot_height": 0.1, #m
    "sample_interval": 2, #s
    "percentage": True, #Concentrations converted to percentage
    "interpolate": True, #Linearly interpolate concentration field the robot samples from
    "c_min": 1e-5, #Threshold of concentration that will be viualized - and can be detected by robot. 
    "noise": False #add noise
}

starting_pos = [1.5, 1.5, robot_physics["robot_height"]] # will be overwritten if random_start = True in robot config 

# ============================================================
# 3. CREATE ROBOTS
# ============================================================

# Algorithm options:
# 1. "LW" - Levy Walk
# 2. "GA" - Gradiant Ascend
# 3. "BH" - Bayesion Heading
# 4. "BHSg" - Bayesion Heading Greedy Spiral
# 5. "SS" - Static Sensor
# ---------------------------------------------------------

if single_run or sweep_run:
    # Run single algorithm or algorithm for sweep
    robot_algorithms = ["BHGS"]
    robot_noise_settings = [False]
elif batch_run:
    robot_algorithms = ["LW", "GA", "BH", "BHGS"] #multiple algorithms to compare
    robot_noise_settings = [False, True]   #Test each algorithm with and without noise

# ============================================================
# 3. Initialize all combinations of robot algo, concentration field, and noise
# ============================================================
robot_configs = {}

# Loop over noise flag
for noise_flag in robot_noise_settings:

    # Loop over all different fields
    for field_name in selected_fields:
        
        # Loop over all different algorithms
        for algorithm in robot_algorithms:
            
            # Recognizable name
            config_name = f"{algorithm}_{field_name}_noise{noise_flag}"

            
            cfg = {
                "algorithm": algorithm,
                "field": field_name,

                "start_pos": starting_pos,
                "random_start": True,

                # Init algo parameters to empty
                "algo": {}
            }
            
            # ============================================================
            # 3. ASSING APPROPIATE PARAMETERS PER ALGORITHM
            # ============================================================
            
            if algorithm == "LW":
            
                cfg["algo"].update({
                    "s": 1.5, #levy scale 
                    "c_found": 10 #odor found treshold [%]
                })

            elif algorithm == "GA":
                
                cfg["algo"].update({
                    "s": 1.5, #levy scale
                    "c_hit": 0.001, #concentration to start GA
                    "G_step": 0.2,
                    "c_lost": 0.005, #plume lost threshold [%]
                    "c_found": 10 #odor found treshold [%]
                })
                
            elif algorithm == "BH":
                cfg["algo"].update({
                    "s": 1.5, #levy scale
                    "c_hit": 0.001, #hit concentration [%]
                    "G_step": 0.2, #gradient step size [m]
                    "n_angles": 32, #number of discretized angles
                    "S_time": 1.5, #sample wait time [s]
                    "c_w": 1.5, #contribution of concentration to confidence
                    "CF_lost": 0.3, #confidence threshold for losing plume
                    "CF_found": 1.1, #confidence threshold for finding odor source
                    "kappa": 2.5, #deviation from belief (noise)
                    "k": 4, #how quickly belief updates based on new measurements
                    "R_max": 0.75,  #maximum radius of spiral reacquisition 
                })
                
            elif algorithm == "BHGS":
                cfg["algo"].update({
                    "s": 1.5, #levy scale
                    "c_hit": 0.001, #hit concentration [%]
                    "G_step": 0.2, #gradient step size [m]
                    "n_angles": 32, #number of discretized angles
                    "S_time": 1.5, #sample wait time [s]
                    "c_w": 1.5, #contribution of concentration to confidence
                    "CF_lost": 0.3, #confidence threshold for losing plume
                    "CF_found": 1.1, #confidence threshold for finding odor source
                    "kappa": 2.5, #deviation from belief (noise)
                    "k": 4, #how quickly belief updates based on new measurements
                    "R_max": 0.75,  #maximum radius of spiral reacquisition
                    "S_greed": 50 #hit concentration multiplier in the spiral state 
                })

            # Assign standard robot physics - only overwriting noise flag
            cfg["robot_physics"] = copy.deepcopy(robot_physics)
            cfg["robot_physics"]["noise"] = noise_flag
            robot_configs[config_name] = cfg

# ============================================================
# SINGLE RUN
# ============================================================
if single_run:
    
    robot_name = config_name 
    robot_algo = robot_configs[config_name]["algorithm"]
    field_name = robot_configs[config_name]["field"]
    
    # ============================================================
    # INITIALIZE SIMULATION RUNNERS
    # ============================================================
    if replay_run:
        
        print(f"Visualizing replay from folder {run_name}, showing robot {robot_name}")
        
        runner = replayRun(run_name=replay_path, fields=fields, robot_name=robot_name, 
                           max_search_time=max_time, visualize=visualize_live, 
                           saveResults=save_run_results, saveFullData= save_full_run_log )
    else:
        
        runner = SimRunner(run_name=run_name, fields = fields, robot_physics = robot_physics, 
                           robot_configs=robot_configs, sim_dt=sim_dt,  max_search_time = max_time, seed = None, 
                           visualize= visualize_live, saveResults=save_run_results, saveFullData= save_full_run_log)
        
        print(f"Running brand new {robot_name} robot in {field_name} field with {robot_algo} algorithm")

    # ============================================================
    # INITIALIZE VIZUALIZATION PLOTS
    # ============================================================
    if visualize_live:
        
        # Pre-init all plots
        # ---------------------------------------------------------
        
        """
        The visualizer class, can handle a flexible amount of plots, in one figure.
        Currently the available plot types are: 
        
        1. ConcHeatMap2D: Concentration heat map for "xy", "xz" or "yz" plane, at a defined slice height. 
                            The concentration gradient can be smoothed, the robot position can be drawn,
                            as well as the path. varBar indicates if the concentration scale is static or 
                            variable based on the current t values. If static, the scale can be based on the
                            global concentration maxiumum [of all time steps] or the local slice concentration maxiumum.
                            
        2. ConcPoint1D:     Plots the measured concentration of the robot, based on the position in the Robot class and the sensor_height.
        
        3. Room3DView: Plots the 3D simulation room, with inlet, outlet, odor location and the robot position. Optional to show velocity quiver.
        
        4. BeliefPolarPlot: Plot belief, likelihood, direction and confidence of the robot
        
        plots = [] can be an array of these visualizations, plotting and ordering will be done automatically
        """
        plots = [
            ConcPoint1D(
                ax=None, fig=None, robot_name=robot_name, field_name=field_name,
                scale="log",
                robot_height=robot_physics["robot_height"],
                plotRealC=True, #plot real time simulated concentration
                plotRobotC=True #plot sampled concentration
            ),
            ConcHeatMap2D(
                ax=None, fig=None, robot_name=robot_name, field_name=field_name,
                plane="xy",
                slice_height=robot_physics["robot_height"],
                smooth=False, #smooth the grid data for visualzation
                varBar=False, #concentration label bar will scale with range of plot
                globalRange=False, #scale concentration bar based on global or local maximum
                scale="log",
                drawRobot=True, 
                drawPath=True, #draw path of robot
                draw3D=True, #draw 3D visualization of room 
                drawQuiver=False, #draw quiver - not recommended for small room previes
                quiver_res=250, #quiver density, higher is less arrows
                drawSample=True #draw sampling state
            ),
            BeliefPolarPlot(None, None, robot_name, field_name, 
                            draw_likelihood=True, #probabilty calculated based soley on newest measurement
                            draw_heading=True #draw heading arrow
                            ),
            
        ]

        # Draws figure, builds legend, resized everyting
        # ---------------------------------------------------------
        viz = Visualizer(runner, plots, figsize=(7.5, 6), n_col=3, show = not save_vid)

        # ============================================================
        # Live animation
        # ============================================================
        ani = viz.animate(
                interval=50,
                fig=viz.fig,
            )
        
        
        if save_vid:
            
            save_path = f"runs/{run_name}.mp4"
            print(f"Saving video to {save_path}...")
            
            # Initialize a tqdm progress bar
            # ---------------------------------------------------------
            with tqdm() as pbar:
                def update_progress(current_frame, total_frames):
                    
                    # Set the total frames only once
                    if pbar.total is None and total_frames:
                        pbar.total = total_frames
                    pbar.update(1) # Increment the progress bar by 1 frame

                ani.save(
                    save_path,
                    writer="ffmpeg",
                    fps=20,
                    dpi=200,
                    progress_callback=update_progress
                )
            print("Video saved successfully!")
        else:
            print("Run not save to video not saved")    
    else:
        # Single run with no visualization
        # ---------------------------------------------------------
        print("Running 1 robot headless")
        while not runner.finished:
            runner.step()
            
# ============================================================
# BATCH RUN
# ============================================================      
elif batch_run:
    
    print("============================================================")
    print(f" STARTING BATCH RUN: {batch_name}")
    print("============================================================")
    print(f"Total Runs per Config: {n_runs_batch}")
    print(f"Total Configurations : {len(robot_configs)}")
    print(f"Total Simulations    : {n_runs_batch * len(robot_configs)}")
    print(f"Algorithms Tested    : {', '.join(robot_algorithms)}")
    print(f"Fields Included      : {', '.join(selected_fields)}")
    print("============================================================")
    
    df = batchRun(
        fields=fields,
        robot_physics=robot_physics,
        robot_configs=robot_configs,
        sim_dt=sim_dt,
        max_search_time=max_time,
        n_runs=n_runs_batch,
        base_seed=50,
        visualize=False,
        batch_name=batch_name,
        saveResults=False,
        saveFullData=False
    )

# ============================================================
# SWEEP RUN
# ============================================================  

elif sweep_run:
    

    # 2. Build grid of parameter combinations
    keys, values = zip(*sweep_parameters.items())
    grid = [
        dict(zip(keys, v)) 
        for v in itertools.product(*values)
    ]
    # Calculate how many base configs match the sweep algorithm
    applicable_configs = len([k for k, v in robot_configs.items() if v["algorithm"] == algorithm_to_sweep])
    total_sims = len(grid) * applicable_configs * n_runs_sweep
    
    print("============================================================")
    print(f" STARTING PARAMETER SWEEP: {sweep_name}")
    print("============================================================")
    print(f"Algorithm Swept      : {algorithm_to_sweep}")
    print(f"Parameters Swept     : {', '.join(sweep_parameters.keys())}")
    print(f"Runs per Combo       : {n_runs_sweep}")
    print("------------------------------------------------------------")
    print(f"Fields Included      : {', '.join(selected_fields)}")
    print("============================================================")

    sweep_df = paramSweep(
        fields=fields,
        algorithm_to_sweep = algorithm_to_sweep,
        robot_physics=robot_physics,
        robot_configs=robot_configs,
        sweep_grid=grid,
        sim_dt=sim_dt,
        max_search_time=max_time,
        n_runs=n_runs_sweep,
        base_seed=50,
        sweep_name=f"{sweep_name}",
        saveResults=False,
        saveFullData=False,
    )