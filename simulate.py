# ============================================================
# IMPORTS
# ============================================================
from dataclasses import dataclass, field                
import copy
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime

# ============================================================
# ROBOT REGISTRY
# ============================================================

# Available robot types and algorithms
# ---------------------------------------------------------
from A_robots.SS import SS
from A_robots.LW import ROBOT_LW
from A_robots.GA import ROBOT_GA
from A_robots.BH import ROBOT_BH 
from A_robots.BHGS import ROBOT_BHGS

# Mapping string identifiers for visualization and accesing
# ---------------------------------------------------------
ROBOT_REGISTRY = {
    "SS":          SS,
    "LW":          ROBOT_LW,
    "GA":          ROBOT_GA,
    "BH":          ROBOT_BH,
    "BHGS":        ROBOT_BHGS
}

# ============================================================
# SIMULATION DATA STRUCTURES
# ============================================================

@dataclass
class SimState:
    """Sim state robot and visualizer can access - time data and active robots + fields"""
    
    # Time state
    # ---------------------------------------------------------
    sim_time: float
    t_idx_comsol: float
    frame: float

    # Robot & Fields
    # ---------------------------------------------------------
    robots: dict
    fields: dict
    
    # Finished state
    # ---------------------------------------------------------
    finished: bool
    
    
@dataclass
class RunResult:
    """Stores the results of a single robot run"""
    
    # Inique identifiers
    # ---------------------------------------------------------
    robot_name: str
    field_path: str
    field_name: str

    # Success Metrics
    # ---------------------------------------------------------
    success: bool
    found_time: int | None
    timeout: bool

    # Performance Metrics
    # ---------------------------------------------------------
    run_time: float
    path_length: float
    path_ratio: float
    
    # Positions & Error
    # ---------------------------------------------------------
    final_pos: tuple
    odor_pos: tuple
    error: float    

# ============================================================
# SIMULATOR CLASS
# ============================================================
    
class SimRunner:

    def __init__(self, fields, robot_physics, robot_configs, 
                 sim_dt, max_search_time, seed=None, 
                 visualize=False, 
                 saveResults = False, saveFullData= False, run_name = None):
        
        # Global RNG - controls reproducibility of robots
        # ---------------------------------------------------------
        self.master_seed = seed if seed is not None else np.random.randint(1e9)
        self.master_rng = np.random.default_rng(self.master_seed)


        self.fields = fields
        self.sim_dt = sim_dt
        self.visualize = visualize
        self.robot_physics = robot_physics

        # Storage variables
        # ---------------------------------------------------------
        self.saveResults = saveResults
        self.saveFullData = saveFullData
        self.run_name = run_name
        
        # Create directory if results must be saved
        if self.saveResults:
                
            self.run_dir = os.path.join("runs", run_name)
            os.makedirs(self.run_dir, exist_ok=False)
        
            # Save config files
            self.global_config = {
                "sim_dt": sim_dt,
                "master_seed": self.master_seed,
                "visualize": visualize,
                "robot_configs": robot_configs,
                "robot_physics": robot_physics
            }
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(self.global_config, f, indent=4, default=str)
            
        # Init sim variables
        # ---------------------------------------------------------
        self.sim_time = 0.0
        self.frame = 0
        self.finished = False
        self.max_search_time = max_search_time

        # COMSOL Settings - assumes all sims have the same dt, takes dt from first field
        # ---------------------------------------------------------
        first_field = next(iter(self.fields.values()))
        self.comsol_dt = first_field.comsol_dt

        # Visual settings
        # ---------------------------------------------------------
        t_min = min(f.time[-1] for f in self.fields.values())
        self.tot_frames = int(t_min / self.sim_dt) - 1

        # Robot Initialization
        # ---------------------------------------------------------
        self.robot_seeds = {}
        self.robots = {}

        
        # ============================================================
        # BUILD ALL ROBOTS
        # ============================================================
        for name, cfg in robot_configs.items():

            # Create a seed per robot based on the master seed
            # Batch and sweep runs thus have a master seed, based on which
            # each individual robots gets it on seed, which is saved to config
            # and then converted to local rng robot can use
            # ---------------------------------------------------------
            this_robot_seed = int(self.master_rng.integers(0, 1_000_000_000))
            self.robot_seeds[name] = this_robot_seed
            local_rng = np.random.default_rng(this_robot_seed)
            
            # Find the field corresponding to the field the current robot runs on
            # ---------------------------------------------------------
            field = self.fields[cfg["field"]]
            
            # Build Physics
            # ---------------------------------------------------------
            base_args = dict(
                
                seed=this_robot_seed,
                field=field,
                sim_dt=self.sim_dt,
                start_pos=cfg["start_pos"],
                robot_height=cfg.get("robot_height", 0.1),

                move_speed=robot_physics["move_speed"],
                turn_speed=robot_physics["turn_speed"],
                sample_interval=robot_physics["sample_interval"],
                percentage=robot_physics["percentage"],
                interpolate=robot_physics["interpolate"],
                c_min=robot_physics["c_min"],
                noise = robot_physics["noise"],
                max_search_time=self.max_search_time,
            )

            # Optional Physics - if robot config added something to
            # robot physics, update the corresponding value in the
            # above created base_arg
            # ---------------------------------------------------------
            for k, v in cfg.get("robot_physics", {}).items():
                base_args[k] = v
                
            # Overwrite starting pos if random start
            # ---------------------------------------------------------
            if cfg.get("random_start", False):
                base_args["start_pos"] = np.array([
                    local_rng.uniform(field.x.min(), field.x.max()),
                    local_rng.uniform(field.y.min(), field.y.max()),
                    base_args["robot_height"]
                ])
                        
            # Extract Algorithm Parameters
            # ---------------------------------------------------------
            algorithm_args = cfg.get("algo", {}) 
            
            # Get algorithm name and couple it to the available algorithms
            # ---------------------------------------------------------
            algorithm = cfg["algorithm"]
            RobotClass = ROBOT_REGISTRY.get(algorithm)
            
            if RobotClass is None:
                raise ValueError(
                    f"Unknown algorithm '{algorithm}'. "
                    f"Available: {list(ROBOT_REGISTRY)}"
                )
            
            #Initialize robot with the constructed parameters
            self.robots[name] = RobotClass(
                **base_args,
                **algorithm_args
                )
            
            #revert field reference and convert start pos to np
            self.robots[name].field_name = cfg["field"]
            self.robots[name].start_pos = np.array(base_args["start_pos"])
            
        # ---------------------------------------------------------
            
        if self.saveResults:
            # save seeds immediately
            with open(os.path.join(self.run_dir, "robot_seeds.json"), "w") as f:
                json.dump(self.robot_seeds, f, indent=4)
            
        # Initialze saving logs and time_idx for sensor sampling
        # ---------------------------------------------------------
        self.t_idx = 0
        self.robot_logs = {
            name: []
            for name in self.robots.keys()
        }
    
    # ============================================================
    # Main step for all robots
    # ============================================================
    def step(self):

        # Early Exit
        # ---------------------------------------------------------
        if self.finished:
            return self.buildState()

        # Step All Robots First
        # (they decide behavior based on current time)
        # ---------------------------------------------------------
        for robot in self.robots.items():
            # visualizae must be passed since it changes if data needs to be stored
            robot.step(visualize=self.visualize)

        # Update Global Time 
        # ---------------------------------------------------------
        self.sim_time += self.sim_dt
        self.frame += 1

        # COMSOL Index Update
        # ---------------------------------------------------------
        self.t_idx =  int(self.sim_time / self.comsol_dt)
        
        if self.saveFullData:
            self.logStep()
            
        # Check Finished
        # ---------------------------------------------------------
        if all(robot.done for robot in self.robots.values()):
            self.finished = True
            
            # Save performance metrics
            if self.saveResults:
                self._saveResults()   
            
            #Save full loggs
            if self.saveFullData:
                self._saveFullLogs()
                
            return self.buildState()

        # Update the full simstate of the robots
        return self.buildState()
    
    def buildState(self):
        """Constructs and returns the current simulation state"""
        return SimState(
            sim_time=self.sim_time,
            t_idx_comsol=self.t_idx,
            frame=self.frame,
            robots=self.robots,
            fields=self.fields,
            finished=self.finished
        )
        
    def get_result(self):
        
        """Calculate and gather run results for all robots"""
        
        results = []

        # Get results from all robots
        # ---------------------------------------------------------
        for robot_name, robot in self.robots.items():

            field = robot.field
            
            odor_pos = np.array(field.odor_pos)
            final_pos = np.array(robot.pos)

            # Error calculation
            # ---------------------------------------------------------
            start_pos_xy = robot.start_pos[:2]
            odor_pos_xy = odor_pos[:2]
            final_pos_xy = robot.pos[:2]
            error = np.linalg.norm(final_pos_xy - odor_pos_xy)
            
            optimal_path = np.linalg.norm(start_pos_xy - odor_pos_xy)
            path_ratio = (
                robot.path_length / optimal_path
                if optimal_path > 0 else np.nan
            )
            
            results.append(
                RunResult(
                    robot_name=robot_name,
                    field_path=field.comsol_path,
                    field_name=robot.field_name,

                    success=robot.found_odor,
                    found_time = robot.found_time,     
                    timeout=robot.timeout,

                    run_time=self.sim_time,
                    path_length=robot.path_length,
                    path_ratio = path_ratio,
                    
                    final_pos=tuple(float(x) for x in robot.pos),
                    odor_pos=tuple(float(x) for x in field.odor_pos),
                    error=error,
                )
            )

        return results
    
    def _saveResults(self):
        
        """Save run results to csv"""
        
        results = self.get_result()
        
        df = pd.DataFrame([r.__dict__ for r in results])
        df["master_seed"] = self.master_seed  
        
        df.to_csv(os.path.join(self.run_dir, "results.csv"), index=False)
        
        return df
    
    def logStep(self):
        
        """Log each robot step - memory intensive)"""

        for robot_name, robot in self.robots.items():

            row = {
                # Global Sim Info
                # ---------------------------------------------------------
                "sim_time": self.sim_time,
                "frame": self.frame,
                "t_idx": self.t_idx,
            }

            # Robot Full State Info
            # ---------------------------------------------------------
            row.update(robot.getFullState())

            self.robot_logs[robot_name].append(row)
    
    def _saveFullLogs(self):
        
        """Save full detailed step logs to csv"""

        # Safety Check
        # ---------------------------------------------------------
        if not hasattr(self, "robot_logs") or len(self.robot_logs) == 0:
            print("No logs to save.")
            return

        # Output Directory
        # ---------------------------------------------------------
        log_dir = os.path.join(self.run_dir, "full_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Save Each Robot Separately
        # ---------------------------------------------------------
        for robot_name, logs in self.robot_logs.items():

            if logs is None or len(logs) == 0:
                continue

            df = pd.DataFrame(logs)
            df = df.sort_values(by=["frame"])

            file_path = os.path.join(log_dir, f"{robot_name}_full_log.csv")
            df.to_csv(file_path, index=False)

        print(f"Full logs saved to: {log_dir}")
        

# ============================================================
# EXECUTION MODES
# ============================================================

def replayRun(run_name, fields, robot_name, max_search_time, 
              visualize = True, saveResults = False, saveFullData=False):
    
    """Replays a previous run from a saved configuration"""
    
    run_dir = os.path.join("runs", run_name)
    
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    # Open the config of the rerun
    # ---------------------------------------------------------
    robot_configs = config["robot_configs"]
    
    # Check if the robot name is in the file
    # ---------------------------------------------------------
    if robot_name is not None:
        if robot_name not in robot_configs:
            raise ValueError(f"Robot '{robot_name}' not found in {run_name}. Available: {list(robot_configs.keys())}")
        robot_configs = {robot_name: robot_configs[robot_name]}
    
    # Create SimRunner with all config data and original seed
    # ---------------------------------------------------------
    return SimRunner(
        fields=fields,
        robot_physics=config["robot_physics"],
        robot_configs=config["robot_configs"],
        sim_dt=config["sim_dt"],
        max_search_time=max_search_time,
        seed=config["master_seed"],
        visualize=visualize,
        run_name=run_name + "_replay",
        saveResults = saveResults,
        saveFullData= saveFullData
    )

def batchRun(fields, robot_physics, robot_configs, sim_dt, 
             max_search_time, n_runs, base_seed=None, 
             visualize=False, batch_name=None, 
             saveResults = True, saveFullData = False):
    
    """Runs multiple simulations back-to-back with generated seeds"""
    
    if batch_name is None:
        batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    
    # Master rng to generate reproducible seeds for each run
    # ---------------------------------------------------------
    batch_rng = np.random.default_rng(base_seed)
    run_seeds = batch_rng.integers(0, 1_000_000_000, size=n_runs)
    
    all_results = []
    
    for i, seed in enumerate(run_seeds):
        
        run_name = f"{batch_name}/run_{i:03d}"
        
        if saveFullData:
            print(f"[{i+1}/{n_runs}] Running & Saving {run_name} (seed={seed})")
        else:
            print(f"[{i+1}/{n_runs}] Running (RAM Only) {run_name} (seed={seed})")
        
        # Create sim based on current seed
        # ---------------------------------------------------------
        sim = SimRunner(
            fields=fields,
            robot_physics = robot_physics,
            robot_configs=robot_configs,
            sim_dt=sim_dt,
            max_search_time=max_search_time,
            seed=int(seed),
            visualize=visualize,
            saveResults=saveResults,           
            saveFullData=(saveResults and saveFullData), 
            run_name=run_name if saveFullData else None
        )    
        
        # Finish running
        # ---------------------------------------------------------
        while not sim.finished:
            sim.step()
        
        # Get data
        # ---------------------------------------------------------
        results = sim.get_result()
        all_results.extend(results)
    
    # Save combined results across all runs
    # ---------------------------------------------------------
    df = pd.DataFrame([r.__dict__ for r in all_results])

    batch_dir = os.path.join("runs", batch_name)
    os.makedirs(batch_dir, exist_ok=True)
    df.to_csv(os.path.join(batch_dir, "batch_results.csv"), index=False)
    
    batch_meta = {
        "batch_name": batch_name,
        "base_seed": base_seed,
        "n_runs": n_runs,
        "run_seeds": [int(s) for s in run_seeds]
    }
    with open(os.path.join(batch_dir, "batch_meta.json"), "w") as f:
        json.dump(batch_meta, f, indent=4)
    print(f"Batch done. Results saved to runs/{batch_name}/batch_results.csv")
    
    return df


def paramSweep(fields, algorithm_to_sweep, robot_physics, robot_configs, 
               sweep_grid, sim_dt, max_search_time, 
               n_runs, base_seed=None, 
               visualize=False, sweep_name=None, 
               saveResults=False, saveFullResults=False):
    
    """Sweeps over a grid of parameters, running a batch simulation for each combination"""

    if sweep_name is None:
        sweep_name = datetime.now().strftime("sweep_%Y%m%d_%H%M%S")

    all_results = []

    # Execute Batch for Each Parameter Combination
    # ---------------------------------------------------------
    for i, params in enumerate(sweep_grid):

        # Build label from swept params
        label = "_".join(f"{k}={v}" for k, v in params.items())
        batch_name = f"{sweep_name}/combo_{i:03d}_{label}"

        print(f"\n[Sweep {i+1}/{len(sweep_grid)}] Params: {params}")

        # Patch only the target robot with new parameters from the params sweep, leave others untouched
        swept_configs = {
            name: copy.deepcopy(cfg)
            for name, cfg in robot_configs.items()
            if cfg["algorithm"] == algorithm_to_sweep
        }
        
        #replace parameters
        for cfg in swept_configs.values():
            cfg["algo"].update(params)

        # Run Batch
        # ---------------------------------------------------------
        df = batchRun(
            fields=fields,
            robot_physics=robot_physics,
            robot_configs=swept_configs,
            sim_dt=sim_dt,
            max_search_time=max_search_time,
            n_runs=n_runs,
            base_seed=base_seed,
            visualize=visualize,
            batch_name=batch_name,
            save_idv_runs=saveResults,
            save_to_disk=saveFullResults
        )

        # Tag rows with swept param values
        # ---------------------------------------------------------
        for k, v in params.items():
            df[k] = v

        all_results.append(df)

    # Combine and Save Master CSV
    # ---------------------------------------------------------
    sweep_df = pd.concat(all_results, ignore_index=True)
    sweep_dir = os.path.join("runs", sweep_name)
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_df.to_csv(os.path.join(sweep_dir, "sweep_results.csv"), index=False)

    print(f"\nSweep done. Combined results saved to runs/{sweep_name}/sweep_results.csv")
    
    # Performance Summary
    # ---------------------------------------------------------
    swept_keys = list(sweep_grid[0].keys())
    
    print("\n─────────────────── Sweep summary ─────────────────────")
    print("Swept fields:", ", ".join(fields.keys()))
    
    field_metrics = (
    sweep_df
    .groupby(swept_keys + ["field_name"])
    .agg(
        success=("success", "mean"),
        found_time=("found_time", "mean"),
        error=("error", "mean"),
        path_ratio=("path_ratio", "mean")
    )
    .reset_index()
    )
    
    grouped = (
        field_metrics
        .groupby(swept_keys)
        .agg({
            "success": "mean",
            "found_time": "mean",
            "error": "mean",
            "path_ratio": "mean"
        })
        .reset_index()
        .rename(columns={
            "success": "success_rate",
            "found_time": "avg_found_time",
            "error": "avg_error",
            "path_ratio": "avg_path_ratio"
        })
    )
    
    g = grouped.copy()

    def norm(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-9)

    # Calculate overall score metric
    # ---------------------------------------------------------
    g["score"] = (
        1.0 * g["success_rate"] +
        0.5 * (1 - norm(g["avg_found_time"])) +
        0.5 * (1 - norm(g["avg_error"])) +
        0.25 * (1 - norm(g["avg_path_ratio"]))
    )

    grouped_sorted = g.sort_values("score", ascending=False)

    print("\n── Overall Sweep Results (field-averaged) ──\n")

    for i, row in enumerate(grouped_sorted.itertuples(index=False), 1):

        params = ", ".join(f"{k}={getattr(row, k)}" for k in swept_keys)

        print(
            f"{i:02d}. {params:<50} → "
            f"success={row.success_rate:.3f} | "
            f"time={row.avg_found_time:.2f} | "
            f"error={row.avg_error:.3f} | "
            f"path_ratio={row.avg_path_ratio:.2f} |"
            f"score={row.score:.3f}"
        )

    return sweep_df