import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import re
from pathlib import Path
from datetime import datetime

class ComsolData:
    
    def __init__(self, 
                 comsol_path,
                 convert_to_percentage,
                 load_npz = False,
                 save_npz = False,
                 clip_neg = True):
        
        self.comsol_path = Path(comsol_path)
        
        self.convert_to_percentage = True
        
        if convert_to_percentage:
            print("WATCH OUT: all concentrations will be converted to percentages\n"
                "! make sure thresholds are also percentage wise")
        
        self.load_npz = load_npz
        self.save_npz = save_npz
        self.clip_neg = clip_neg
        
        # -------------------------------------------------
        # Derived paths
        # -------------------------------------------------

        self.parameter_path = self.comsol_path.with_name(
            self.comsol_path.stem + "_parameters.txt"
        )

        self.npz_path = self.comsol_path.with_suffix(".npz")

        self.vid_path = self.comsol_path.with_name(
            self.comsol_path.stem +
            f"_simulationVid_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        )

        # -------------------------------------------------
        # Load data
        # -------------------------------------------------

        if load_npz and self.npz_path.exists():

            self._load_npz(self.npz_path)

        else:

            self._load_comsol_export()

            self._compute_grid_prop()
            self._validate_grid()

        # -------------------------------------------------
        # Load parameters
        # -------------------------------------------------

        self._load_parameters(self.parameter_path)
        
    def _load_comsol_export(self):
        
        """
        Load and read comsol file; print metadata
        """
        
        print(f"Reading COMSOL file from {self.comsol_path}")

        self.metadata = {}

        ###  read metadata of comsol file, metadata are all lines starting with % ### 
        with open(self.comsol_path, "r") as f:
            for line in f:
                if line.startswith("%"):
                    line = line.strip("%").strip()
                    if ":" in line:
                        key, value = line.split(":", 1)
                        self.metadata[key.strip()] = value.strip()
                else:
                    break

        ###  read field data ### 
        self.df = pd.read_csv(
            self.comsol_path,
            comment="%",
            sep=r"\s+"
        )

        ###  extract coordinates and data ### 
        coords = self.df.iloc[:, :3].values #first tree collumns are grid data
        data = self.df.iloc[:, 3:].values #rest is time, concentration, u, v, w, flow_vel
        
        ### each timestep consist of two collums, so the amount of columns/2 = time_steps
        n_col = data.shape[1]
        n_step = n_col // 6

        data = data.reshape(-1, n_step, 6) #reshape to (node, time_idx , [t, c, u, v, w, flow_vel])

        ### create structured data arrays
        self.coords = coords #(N,3) = (N,[x,y,z])
        self.data = data
        self.time = data[0, :, 0] #each row is full time, so only need first row (T,)
        self.concentration = data[:, :, 1] # (N,T)
        self.u = data[:, :, 2] # (N,U)
        self.v = data[:, :, 3] # (N,V)
        self.w = data[:, :, 4] # (N,W)
        self.flow_vel = data[0, :, 5] # (T,)
        

        self.print_metadata()
        
        print("Succesfully read comsol file")
        
        
    def _compute_grid_prop(self):
        """
        Compute grid/room dimensions
        """
        
        print("Extracting grid dimensions")
        
        ### Max, Min [m] room
        x_min, x_max = self.coords[:,0].min(), self.coords[:,0].max()
        y_min, y_max = self.coords[:,1].min(), self.coords[:,1].max()
        z_min, z_max = self.coords[:,2].min(), self.coords[:,2].max()

        ### length room [m]
        self.Lx = x_max - x_min
        self.Ly = y_max - y_min
        self.Lz = z_max - z_min
        
        ### nodes per dim 
        self.x = np.unique(self.coords[:,0])
        self.y = np.unique(self.coords[:,1])
        self.z = np.unique(self.coords[:,2])
        
        ### grid of room
        self.unique_coords = [self.x, self.y, self.z]

        ### N grid points per dim
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nz = len(self.z)
        
        print(self.nx)
        print(self.ny)
        print(self.nz)
        
        ### shape room, handy for reshaping flat concentration data later on - for slicing for example
        self.shape = (self.nx, self.ny, self.nz)
        
        ### comsol_dt - IMPORTANT FOR SYNCING SIMULATIONS
        self.comsol_dt = self.time[1] - self.time[0]    
        
        print(f"Grid data extracted, comsol time step = {self.comsol_dt} s")
        
    def _validate_grid(self):
        """
        Fix COMSOL export mismatch between dataframe size and expected grid size. Often 1 node is missing
        """

        expected = self.nx * self.ny * self.nz ### expected nodes
        actual = len(self.coords) 
        missing = expected - actual

        #check length missmatch
        if missing == 0:
            print("No missing data")
            return

        #Missmatch detected, often 1 value is missing, this is due to 1 weird boundary condition
        print(f"COMSOL grid mismatch")
        print(f"Expected: {expected}, Actual: {actual}, Missing: {missing}")

        if missing < 0:
            raise ValueError("Too many rows in COMSOL export")

        print(f"→ Padding {missing} missing node(s) with zeros")
        
        ### ---- pad coords ---- ###
        coord_pad = np.zeros((missing, 3))  # (N,[x,y,z]) = 0,0,0 placeholder
        self.coords = np.vstack([self.coords, coord_pad])
    
        #pad concentration
        n_steps = self.concentration.shape[1]
        conc_pad = np.zeros((missing, n_steps)) # (1, N_concentration) = (1, N_grid_nodes)
        self.concentration = np.vstack([self.concentration, conc_pad])
    
        ### clip concentration for logaritmic scale
        if self.clip_neg:
            # replace nan with 1e-12
            self.concentration = np.nan_to_num(
                self.concentration,
                nan=1e-12
            )
            self.concentration = np.clip(self.concentration, 1e-12, None)
        
        ## pad vel components
        velocity_pad = np.zeros((missing, n_steps))
        self.u = np.vstack([self.u, velocity_pad])
        self.v = np.vstack([self.v, velocity_pad])
        self.w = np.vstack([self.w, velocity_pad])
        
        ### Save to NPZ since grid is now fixed ###
        if self.save_npz:
            print(f"Saving data to NPZ file")
            self._save_npz(self.npz_path)


    def _save_npz(self, out_path):
        
        """
        NPZ is efficient numpy storing method, easier to load files again
        """

        np.savez_compressed(
            out_path,

            coords=self.coords,
            time=self.time,
            concentration=self.concentration,
            u = self.u,
            v = self.v,
            w = self.w,
            flow_vel = self.flow_vel,

            x=self.x,
            y=self.y,
            z=self.z,

            nx=self.nx,
            ny=self.ny,
            nz=self.nz,

            comsol_dt=self.comsol_dt
        )
        
        print(f"Saved processed data to {out_path}")
        
        if self.convert_to_percentage:
            print("WARNING: To convert to percentages, NPZ will be loaded")
            self._load_npz(self.npz_path)
        
    def _load_npz(self, npz_path):
        
        """
        Load from NPZ
        """

        data = np.load(npz_path)

        self.coords = data["coords"]
        self.time = data["time"]
        self.concentration = data["concentration"]
        
        if self.convert_to_percentage:
            # Find the absolute maximum concentration in the dataset
            global_max = self.concentration.max()
            
            # Prevent DivisionByZero if the entire field happens to be empty/zero
            if global_max > 0:
                self.concentration = (self.concentration / global_max) * 100
            else:
                self.concentration = self.concentration * 0.0
        
        self.u = data["u"]
        self.v = data["v"]
        self.w = data["w"]
        self.flow_vel = data["flow_vel"]

        self.x = data["x"]
        self.y = data["y"]
        self.z = data["z"]

        self.nx = int(data["nx"])
        self.ny = int(data["ny"])
        self.nz = int(data["nz"])

        self.shape = (self.nx, self.ny, self.nz)
        self.unique_coords = [self.x, self.y, self.z]

        self.comsol_dt = float(data["comsol_dt"])
        
        print(self.flow_vel)
        
        print(f"Loaded processed NPZ data from {npz_path}, comsol dt = {self.comsol_dt}")

   
    def _load_parameters(self, parameter_path):
        
        """
        Load parameter file
        """
        
        ### Function names in comsol
        safe_dict = {
            "__builtins__": {},
            "sqrt": np.sqrt,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "pi": np.pi,
            "exp": np.exp,
        }

        
        raw = {}  # name -> raw expression string

        unit_pattern = re.compile(r'\[.*?\]')   # strips [m], [m^2], [s] etc.
        quote_pattern = re.compile(r'".*?"')    # strips quoted description strings

        with open(parameter_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # split on first whitespace -> name | rest
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue

                name, rest = parts

                # remove description strings ("Room Width" etc.)
                rest = quote_pattern.sub('', rest).strip()
                # remove unit annotations like [m], [m^2/s], [mol/s]
                rest = unit_pattern.sub('', rest).strip()
                # remove any leftover asterisks from [m]*Ar artifacts
                rest = rest.strip('*').strip()

                raw[name] = rest

        # evaluate in order, building up a namespace
        params = {}
        for name, expr in raw.items():
            try:
                params[name] = eval(expr, safe_dict, params)
            except Exception as e:
                print(f"Could not evaluate '{name} = {expr}': {e}")
                
        self.Ms     = params['Ms']
        self.Us     = params['Us']
        self.odor_x = params['odor_x']
        self.odor_y = params['odor_y']
        self.odor_z = params['odor_z']
        self.As     = params['As']
        self.Ao     = params['Ao']
        
        self.odor_pos = np.array([self.odor_x, self.odor_y, self.odor_z])
        
        print(f"Ms     = {self.Ms}")
        print(f"Us     = {self.Us}")
        print(f"odor pos = {self.odor_pos}")
        print(f"As     = {self.As}")
        print(f"Ao     = {self.Ao}")
        
        # Inlet Coordinates and Dimensions
        self.inlet_x = params['inlet_x']
        self.inlet_y = params['inlet_y']
        self.inlet_z = params['inlet_z']
        self.inlet_w = params['inlet_w']
        self.inlet_d = params['inlet_d']
        self.inlet_h = params['inlet_h']
        self.inlet_pos = np.array([self.inlet_x, self.inlet_y, self.inlet_z])

        # Outlet Coordinates
        self.outlet_x = params['outlet_x']
        self.outlet_y = params['outlet_y']
        self.outlet_z = params['outlet_z']
        self.outlet_pos = np.array([self.outlet_x, self.outlet_y, self.outlet_z])

        # # Control/Physics Parameters
        # self.flow_switch = params["flow_switch"]
        # self.flow_switch_t = None
        # self.smooth_zone = None
        
        # # Only get these variables if flow switch is true 
        # if self.flow_switch == True:
        #     self.smooth_zone    = params['smooth_zone']
        #     self.flow_switch_t  = params['flow_switch_t']
        # print(f"Flow Switch T  = {self.flow_switch_t}")

        # Debug Print
        print(f"Inlet pos = {self.inlet_pos}")
        print(f"Outlet pos= {self.outlet_pos}")

##### EXTERNAL FUNCTIONS

    def get_c_slice(self, t_idx, dim1, dim2, slice_height, t_full = False):
        
        """
        Get all concentrion data from a slice [plane]. dim1 and dim2 correspond to xyz, 012.
        t_full allows concentration for all t, to find maxima
        
        COMSOL WEIRD EXPORT: the comsol data is in from z,y,x --> this is reshape method
        then transpose into x,y,z for visualization/robot
        """
        ### get concentration data from all time steps, used to find max min range for visualization ###
        if t_full:
            nt = self.concentration.shape[1] ##all concentration values
            #reshape into grid dim + time dimension
            C_grid = self.concentration.reshape(
            self.nz, self.ny, self.nx, nt
            )
            C_grid = np.transpose(C_grid, (2,1,0,3)) #transpose due to weird comsol structure(nx, ny, nz, nt)
        else:
            #get slice for 1 timestep  
            C = self.concentration[:,t_idx]
            C_grid = C.reshape(self.nz, self.ny, self.nx)
            C_grid = np.transpose(C_grid, (2,1,0)) #z is first in comsol, weird comsol structure
        
        interp_axis = ({0,1,2} - {dim1, dim2}).pop() #get diminsion in which we need to interpolate, if xy plane --> z
            
        C_slice = np.apply_along_axis(
            lambda col: np.interp(slice_height, self.unique_coords[interp_axis], col),
            axis=interp_axis,
            arr=C_grid
        )
        
        # imshow wants (ny, nx), this is (nx, ny). so transpose 
        return C_slice.T
    
    def get_c_sample(self, t_idx, interpolate, x, y, robot_height):
        
        """
        Get sample at singular point in 3D space
        """
        
        # Find the index of the grid point closest to the evaluation point. 0 is close to correct index
        i = np.argmin(np.abs(self.x - x))
        j = np.argmin(np.abs(self.y - y))

        C_grid = self.concentration[:, t_idx].reshape(self.nz, self.ny, self.nx) ## weird reshape due to comsol
        C_grid = np.transpose(C_grid, (2, 1, 0))  #reshope to x,y,z
        
        #interpolate entire grid, get value at exact position
        if interpolate:
            
            interp = RegularGridInterpolator(
                (self.x, self.y, self.z),
                C_grid,
                bounds_error=False,
                fill_value=0.0
            )
            
            c_value = interp([x, y, robot_height])
        else:
            #### no interpolation - only interpolate z-column, to get exact robot heigt
            #interpolate value based on z value above and below the (x,y)/(i,j) point
            z_column = C_grid[i, j, :] 
            
            c_value = np.array([np.interp(robot_height, self.z, z_column)])
            
        return c_value

    def print_metadata(self):
            
            """COMSOL metadata."""
            print("Model name:", self.metadata.get("Model"))
            print("Export date:", self.metadata.get("Date"))
            print("Dimensions:", self.metadata.get("Dimension"))
            print("Nodes:", self.metadata.get("Nodes"))