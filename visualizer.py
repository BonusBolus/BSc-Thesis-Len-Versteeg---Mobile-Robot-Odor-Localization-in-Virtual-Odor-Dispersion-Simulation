# ============================================================
# IMPORTS
# ============================================================
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from simulate import SimState, SimRunner
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

# ============================================================
# GLOBAL SETTINGS
# ============================================================

# Standard font settings
# ---------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    "grid.alpha": 0.25,
})

# Standard color settings
# ---------------------------------------------------------
robot_c = '#39FF14'
odor_c = '#EF9F27'
slice_c = 'cyan'

STATE_COLORS = {
    "LEVY": "#E6E6E6",
     "SPIRAL": "#003CA4",
    "GRADIENT": "#00E5FF",
    "SAMPLING": "#FFB300",
}


# ============================================================
# BASE PLOT CLASS
# ============================================================
"""
Base function constructor: Each plot should initialize on an axis 
and have an update function based on field, t_idx, sim_time, and robot_pos.
"""

class BasePlot:

    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self.legend_items = []

    def init(self, field):
        pass

    def update(self, field, t_idx, sim_time):
        return []

    def legend_items_func(self):
        """Return list of lengend tuples"""
        return self.legend_items
    
# ============================================================
# 2D CONCENTRATION HEATMAP
# ============================================================

class ConcHeatMap2D(BasePlot):
    
    # ---------------------------------------------------------
    # INITIALIZATION (STATIC SETTINGS)
    # ---------------------------------------------------------
    def __init__(self, ax, fig, robot_name, field_name, 
                 plane: str, slice_height, 
                 smooth = False, varBar = True, globalRange = True, scale = "log",  
                 drawRobot = False, drawPath = True, 
                 draw3D = False, drawQuiver = False, quiver_res = 250, 
                 drawSample = False):
        
        super().__init__(ax, fig) 

        # Sim Settings
        # ---------------------------------------------------------
        self.last_idx  = None # weird value so in beginning always override
        
        # Robot Settings
        # ---------------------------------------------------------
        self.robot_name = robot_name
        self.field_name = field_name
        
        self.drawRobot = drawRobot
        self.drawPath = drawPath
        self.drawSample = drawSample
        
        # Slice Settings
        # ---------------------------------------------------------
        self.plane = plane
        self.slice_height = slice_height
        self.smooth = smooth

        # Range Settings
        # ---------------------------------------------------------
        self.varBar = varBar
        self.globalRange = globalRange
        self.scale = scale
        
        # 3D Settings
        # ---------------------------------------------------------
        self.draw3D = draw3D
        self.drawQuiver = drawQuiver
        self.quiver_res = quiver_res
    
        # Extract Correct Dim Based on Plane
        # ---------------------------------------------------------
        planes = {
            'xy': (0, 1, 'x [m]', 'y [m]', 'xy plane'),
            'xz': (0, 2, 'x [m]', 'z [m]', 'xz plane'),
            'yz': (1, 2, 'y [m]', 'z [m]', 'yz plane'),
        }
        
        # dimensions ensures correct axis x=0, y=1, z=2
        self.dim1, self.dim2, self.xlabel, self.ylabel, self.slice_label = planes[plane]
        
        
    # ---------------------------------------------------------
    # DATA INITIALIZATION (CALLED BY VISUALIZER)
    # ---------------------------------------------------------
    def init(self, field, robot):
        
        # Plot Coordinates
        # ---------------------------------------------------------
        x = field.unique_coords[self.dim1]
        y = field.unique_coords[self.dim2]
        
        # Init Concentration Bar Label
        # ---------------------------------------------------------
        if robot.percentage:
            if self.varBar:
                c_label = "Normalized Concentration (%) \n [dynamic local range]"
            elif self.globalRange:
                c_label = "Normalized Concentration (%) \n  [static global range]"
            else:
                c_label = "Normalized Concentration (%) \n [static slice range]"
            
        else: 
            if self.varBar:
                c_label = "Concentration (mol/m^3) \n [dynamic local range]"
            elif self.globalRange:
                c_label = "Concentration (mol/m^3) \n [static global range]"
            else:
                c_label = "Concentration (mol/m^3) \n [static slice range]"
        
        # Init Concentration Bar Range
        # ---------------------------------------------------------
        vmin = robot.c_min
        
        if not self.varBar:
            if self.globalRange:
                vmax =field.concentration.max()

            elif not self.globalRange:
                C_slice_full_t = field.get_c_slice(t_idx = -100, dim1 = self.dim1, dim2 = self.dim2, slice_height = self.slice_height, t_full = True)
                vmax = C_slice_full_t.max()
                
            # SET NORMALIZATION TYPE
            if self.scale == "log":
                self.norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            elif self.scale == "lin":
                self.norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else: 
                ValueError(f"Scale must be either log or lin")
        
        else:
            self.norm = None
        
        # Interpolation Method
        # ---------------------------------------------------------
        if self.scale == "lin":
            if self.smooth:
                self.interpolation ='bilinear'
            elif not self.smooth:
                self.interpolation = "nearest"
        elif self.scale == "log":
            if self.smooth:
                self.interpolation = "bicubic"
            elif not self.smooth:
                self.interpolation = "nearest"
        
        
        self.im2D = self.ax.imshow(
            np.zeros((len(x), len(y))),
            origin='lower',
            extent=[x.min(), x.max(),
                    y.min(), y.max()],
            norm = self.norm,
            interpolation = self.interpolation,
            animated=True,
            cmap ="magma"
        )
        
        self.cbar = plt.colorbar(self.im2D, ax=self.ax, label=c_label, orientation = 'vertical', location = 'left', shrink = 0.8, pad = 0.15)
        
        # Set Axis Labels & Title
        # ---------------------------------------------------------
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        
        # Find slice dimension
        self.dim3 = ({0, 1, 2} - {self.dim1, self.dim2}).pop()
        dim3_map = {0: "x", 1: "y", 2: "z"}
        dim3_name = dim3_map[self.dim3]
        
        self.title_text = self.ax.set_title(f'Odor concentration slice of {self.slice_label} at {dim3_name} = {self.slice_height} [m]')
        
        # Draw Robot
        # ---------------------------------------------------------
        if self.drawRobot:
            
            # Triangle
            self.robot_dot, = self.ax.plot(
                [], [],
                marker=(3, 0, 0),
                markersize=10,
                linestyle='None',
                color=robot_c,
                zorder=5
            )
            self.legend_items.append((
                self.robot_dot,
                "Robot"
            ))
            
            # Init Path
            if self.drawPath:
                self.robot_trail, = self.ax.plot([], [], c="white", lw=2, 
                                    linestyle='dotted', alpha=0.75, zorder=2, label="Robot path")
                
                for state, color in STATE_COLORS.items():
                    if state == "SAMPLING":
                        representative_color = cm.magma(0.7) 
                        handle = mlines.Line2D(
                            [], [], 
                            color='None',
                            marker='o',
                            markerfacecolor=representative_color, 
                            markeredgecolor='white',
                            markersize=10
                        )
                    else:
                        handle = Patch(color=color)
                        
                    self.legend_items.append((
                        handle,
                        f"Path: {state.capitalize()}"
                    ))
                        
            if self.drawSample:
                self.last_sample_count = 0
            
        # Draw Static Odor
        # ---------------------------------------------------------
        self.odor_dot = self.ax.scatter(
            [field.odor_pos[self.dim1]], [field.odor_pos[self.dim2]],
            s=50, c=odor_c, zorder=5, edgecolors='black', label='Odor source'
        )
        
        self.legend_items.append((
            self.odor_dot,
            "Odor source"
        ))
        
        # Small inset 3D Room Setup
        # ---------------------------------------------------------
        if self.draw3D:
            self.ax_inset = inset_axes(self.ax, width="45%", height="45%", loc='right', 
                                    axes_class=Axes3D, borderpad=-5)
            self.room3D = Room3DView(ax=self.ax_inset, fig=self.fig, drawQuiver=self.drawQuiver, quiver_res=self.quiver_res, drawSlice=True, plane = self.plane, slice_height=self.slice_height, drawRobot=self.drawRobot, robot_name=self.robot_name, field_name = self.field_name)
            self.room3D.init(field, robot)
        
    # ---------------------------------------------------------
    # UPDATE FRAME
    # ---------------------------------------------------------
    def update(self, state: SimState):
        """Updates all data in the heatmap, appends to artist."""
        
        # Extract Fields and Robot
        # ---------------------------------------------------------
        robot = state.robots[self.robot_name]
        field = state.fields[self.field_name]
        t_idx = state.t_idx_comsol
        sim_time = state.sim_time
        
        if robot.found_odor:
            return []
        
        robot_pos = robot.pos
        artists = []
        
        # Update Concentration (Only if new COMSOL step passed)
        # ---------------------------------------------------------
        if t_idx != self.last_idx:
            C = field.get_c_slice(t_idx = t_idx, dim1 = self.dim1, dim2= self.dim2, slice_height= self.slice_height, t_full = False)
            self.im2D.set_data(C)
            
            if self.varBar:
                self.im2D.set_clim(vmin=C.min(), vmax=C.max())
            
            self.last_idx = t_idx
            artists.append(self.im2D)
        
        # Update Robot Pos & Path
        # ---------------------------------------------------------
        if self.drawRobot:
            
            x = robot_pos[self.dim1]
            y = robot_pos[self.dim2]

            self.robot_dot.set_data([x], [y])
                        
            # Only draw path in XY plane
            if self.plane == "xy":
                
                self.robot_dot.set_marker((3, 0, robot_pos[3]))
                
                # Draw Full Path History
                if self.drawPath and len(robot.robot_pos_history) >= 2:
                    
                    x0, y0 = robot.robot_pos_history[-2][0], robot.robot_pos_history[-2][1]
                    x1, y1 = robot.robot_pos_history[-1][0], robot.robot_pos_history[-1][1]
                    
                    color = STATE_COLORS.get(robot.state_history[-1], 'white')
                    
                    seg, = self.ax.plot([x0, x1], [y0, y1], color=color, lw=2, alpha=0.75)
                    artists.append(seg)
                    
                # Draw Sampling Markers
                if self.drawSample:
                    if len(robot.sample_history) > self.last_sample_count:
                        
                        for _, avg in robot.sample_history[self.last_sample_count:]:
                            x, y = robot.robot_pos_history[-1][0], robot.robot_pos_history[-1][1]
                            self.ax.scatter(x, y, c=avg, cmap='magma', s=40, 
                                        norm = self.norm,
                                        edgecolors='white', lw=0.8, zorder=6)
                        self.last_sample_count = len(robot.sample_history)
                
            else: 
                # If not XY plane, direction has no real meaning
                self.robot_dot.set_paths([
                    mmarkers.MarkerStyle('o').get_path()
                    .transformed(mmarkers.MarkerStyle('o').get_transform())
                ])
                
            # Update Inset 3D Plot
            if self.draw3D:
                self.artist3D = self.room3D.update(state)
                artists += self.artist3D
            
        return artists
    
    def legend_items_func(self):
        
        items = list(super().legend_items_func())
        
        # 3D inset plot legend items
        if hasattr(self, 'draw3D') and self.draw3D and hasattr(self, 'room3D'):
            items.extend(self.room3D.legend_items_func())
            
        return items

# ============================================================
# 1D CONCENTRATION POINT PLOT
# ============================================================

class ConcPoint1D(BasePlot):

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------
    def __init__(self, ax, fig, 
                 robot_name, field_name, 
                 scale, robot_height = 0, 
                 plotRealC = True, plotRobotC = True):
        
        super().__init__(ax, fig)
        
        self.robot_name = robot_name
        self.field_name = field_name
        
        self.robot_height = robot_height
        self.scale = scale
        
        self.plotRobotC = plotRobotC
        self.plotRealC = plotRealC
        
        self.time_window = 20   # s
        
    def init(self, field, robot):
        
        # Storing Arrays
        # ---------------------------------------------------------
        self.t = []
        self.cReal = []
        self.cRobot = []
        self.artists = []

        if self.plotRealC:
            self.lineReal, = self.ax.plot([], [], lw=2.5, linestyle="-", alpha=0.9)
            self.legend_items.append((
            self.lineReal,
            "Real time \n concentration"
            ))

        if self.plotRobotC:
            self.lineRobot, = self.ax.plot([], [], lw=2.5, linestyle="--", alpha=0.9, color=robot_c)
            self.legend_items.append((
            self.lineRobot,
            "Sampled \n concentration by robot"
            ))

        if self.scale == "log":
            self.ax.set_yscale('log')

        # Formatting & Labels
        # ---------------------------------------------------------
        self.ax.set_xlabel("Time [s]")
        if robot.percentage:
            self.ax.set_ylabel("Normalized Concentration (%)")
        else: 
            self.ax.set_ylabel("Concentration (mol/m^3)")
            
        self.title_text = self.ax.set_title(f'Odor concentration at robot position: []')
        self.ax.grid(True)
        self.ax.set_facecolor("#c0afaf")
        
        # Threshold Lines
        # ---------------------------------------------------------
        if hasattr(robot, 'c_hit') and robot.c_hit is not None:
            self.hit_line = self.ax.axhline(
                y=robot.c_hit,
                color="red",
                linestyle=":",
                alpha=0.7
            )
        else:
            self.hit_line = None
            
        self.artists.append(self.hit_line)
        self.legend_items.append((
            self.hit_line,
            "Hit Threshold"
            ))
        
        if robot.percentage:
            self.ax.set_ylim(robot.c_min, 100)
        else: 
            self.ax.set_ylim(robot.c_min, 10e5)
            
    # ---------------------------------------------------------
    # UPDATE FRAME
    # ---------------------------------------------------------
    def update(self, state: SimState):

        robot = state.robots[self.robot_name]
        field = state.fields[self.field_name]
        t_idx = state.t_idx_comsol
        sim_time = state.sim_time
        
        if robot.found_odor:
            return []
        
        i, j = robot.pos[0], robot.pos[1]

        # Store New Values
        # ---------------------------------------------------------
        self.t.append(sim_time)

        if len(robot.c_robot_history) > 0:
            self.cRobot.append(robot.c_robot_history[-1])
        else:
            self.cRobot.append(np.nan)

        if self.plotRealC:
            c_value = field.get_c_sample(
                t_idx=t_idx,
                interpolate = robot.interpolate,
                x=i,
                y=j,
                robot_height=self.robot_height
            )
            self.cReal.append(c_value)

        # Sliding Window (Time-Based)
        # ---------------------------------------------------------
        t_min = sim_time - self.time_window
        mask = np.array(self.t) >= t_min

        self.t = list(np.array(self.t)[mask])

        if self.plotRobotC:
            self.cRobot = list(np.array(self.cRobot)[mask])

        if self.plotRealC:
            self.cReal = list(np.array(self.cReal)[mask])

        # Update Plots
        # ---------------------------------------------------------
        if self.plotRobotC:
            self.lineRobot.set_data(self.t, self.cRobot)

        if self.plotRealC:
            self.lineReal.set_data(self.t, self.cReal)

        # Axis Handling
        # ---------------------------------------------------------
        if len(self.t) > 2:
            self.ax.set_xlim(self.t[0], self.t[-1])

            all_vals = []
            if self.plotRobotC:
                all_vals += self.cRobot
            if self.plotRealC:
                all_vals += self.cReal

        # Title Update
        # ---------------------------------------------------------
        self.title_text.set_text(
            f'Odor concentration at robot position (x,y) [m]: '
            f'({np.round(i,2)}, {np.round(j,2)})\n'
            f'Sensor z position [m] = {self.robot_height}'
        )

        return [self.lineReal, self.lineRobot, self.hit_line] if self.plotRealC and self.plotRobotC else self.artists
    
# ============================================================
# 3D ROOM VIEW
# ============================================================

class Room3DView(BasePlot):
    
    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------
    def __init__(self, ax, fig, 
                 drawQuiver, quiver_res, 
                 drawSlice, plane, slice_height, 
                 drawRobot, robot_name, field_name):
        
        super().__init__(ax, fig)
        
        self.drawQuiver = drawQuiver
        self.quiver_res = quiver_res
        self.drawSlice = drawSlice
        self.slice_height = slice_height
        self.plane = plane
        self.drawRobot = drawRobot
        
        self.robot_name = robot_name
        self.field_name = field_name
        
        planes = {
            'xy': (0, 1, 'x [m]', 'y [m]', 'xy plane'),
            'xz': (0, 2, 'x [m]', 'z [m]', 'xz plane'),
            'yz': (1, 2, 'y [m]', 'z [m]', 'yz plane'),
        }
        
        self.dim1, self.dim2, self.xlabel, self.ylabel, self.slice_label = planes[plane]
        self.dim3 = ({0, 1, 2} - {self.dim1, self.dim2}).pop()
    
    def init(self,field, robot):
    
        self.ax.set_facecolor((1,1,1,0)) # Transparent background
        max_d = [field.x.max(), field.y.max(), field.z.max()]
        min_d = [field.x.min(), field.y.min(), field.z.min()]
        
        # Draw Floor & Edged
        # ---------------------------------------------------------
        self.ax.plot([min_d[0], max_d[0]], [min_d[1], min_d[1]], [min_d[2], min_d[2]], color='gray', lw=0.5)
        self.ax.plot([min_d[0], max_d[0]], [max_d[1], max_d[1]], [min_d[2], min_d[2]], color='gray', lw=0.5)
        self.ax.plot([min_d[0], min_d[0]], [min_d[1], max_d[1]], [min_d[2], min_d[2]], color='gray', lw=0.5)

        for x in [min_d[0], max_d[0]]:
            for y in [min_d[1], max_d[1]]:
                self.ax.plot([x, x], [y, y], [min_d[2], max_d[2]], color='gray', lw=0.5, alpha=0.3)
                
        # Draw Slice Plane
        # ---------------------------------------------------------
        if self.drawSlice:
            p_d1 = np.linspace(min_d[self.dim1], max_d[self.dim1], 2)
            p_d2 = np.linspace(min_d[self.dim2], max_d[self.dim2], 2)
            D1, D2 = np.meshgrid(p_d1, p_d2)
            D3 = np.full_like(D2, self.slice_height)
        
            coords_3d = [None, None, None]
            coords_3d[self.dim1], coords_3d[self.dim2], coords_3d[self.dim3] = D1, D2, D3
            
            self.ax.plot_surface(coords_3d[0], coords_3d[1], coords_3d[2], 
                                    alpha=0.25, color=slice_c, zorder=1)
        
        # Draw Elements Odor, Robot, In/Outlets
        # ---------------------------------------------------------
        self.ax.scatter(field.odor_pos[0], field.odor_pos[1], field.odor_pos[2], 
                        color=odor_c, s=20, edgecolors='black')
        
        if self.drawRobot:
            self.robot_3d = self.ax.scatter([], [], [], color=robot_c, s=15, zorder=5)
            
        self.inlet = self.ax.scatter(
            field.inlet_pos[0], field.inlet_pos[1], field.inlet_pos[2],
            color='green', s=20, edgecolors='black', marker='s', zorder=6, label = "Flow inlet"
        )

        self.oulet = self.ax.scatter(
            field.outlet_pos[0], field.outlet_pos[1], field.outlet_pos[2],
            color='red', s=20, edgecolors='black', marker='^', zorder=6, label = "Flow outlet"
        )
        
        self.legend_items.append((self.inlet, "Flow inlet"))
        self.legend_items.append((self.oulet, "Flow outlet"))
        
        # Quiver Setup
        # ---------------------------------------------------------
        self.quiver_artist = None
        self.quiver_proxy = None
        
        if self.drawQuiver:
            self.quiver_proxy = mlines.Line2D(
                [], [], color='black', marker='>', linestyle='None', label='Flow field'
            )
            self.legend_items.append((self.quiver_proxy, "Flow field"))
            
        # 3D Axis
        # ---------------------------------------------------------
        self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_zticks([])
        self.ax.view_init(elev=20, azim=-45) 
        
        self.ax.set_xlabel('x', fontsize=7, labelpad=-15)
        self.ax.set_ylabel('y', fontsize=7, labelpad=-15)
        self.ax.set_zlabel('z', fontsize=7, labelpad=-15)
            
    # ---------------------------------------------------------
    # UPDATE FRAME
    # ---------------------------------------------------------
    def update(self, state: SimState):
        
        robot     = state.robots[self.robot_name]
        field     = state.fields[self.field_name]
        t_idx     = state.t_idx_comsol
        robot_pos = robot.pos
        
        if robot.found_odor:
            return []
        
        artists   = []

        if self.drawRobot:
            self.robot_3d._offsets3d = ([robot_pos[0]], [robot_pos[1]], [robot_pos[2]])
            artists.append(self.robot_3d)
        
        if self.drawQuiver:
            if self.quiver_artist is not None:
                self.quiver_artist.remove()
                
            skip = self.quiver_res
            
            x, y, z = field.coords[::skip, 0], field.coords[::skip, 1], field.coords[::skip, 2]
            u, v, w = field.u[::skip, t_idx], field.v[::skip, t_idx], field.w[::skip, t_idx]

            self.quiver_artist = self.ax.quiver(x, y, z, u, v, w, 
                                                length=0.3, normalize=True, color='black', alpha=0.5, label = "Flow field")
            
            artists.append(self.quiver_artist)
            
        return artists
    
# ============================================================
# POLAR BELIEF PLOT
# ============================================================

class BeliefPolarPlot(BasePlot):
    """
    Polar plot BayesianHeading robot's directional belief
    distribution +  confidence bar chart.
    """

    needs_polar: bool = False

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------
    def __init__(
        self,
        ax, fig,
        robot_name:      str,
        field_name:      str,
        draw_likelihood: bool = True,
        draw_heading:    bool = True,
    ):
        super().__init__(ax, fig)

        self.robot_name      = robot_name
        self.field_name      = field_name
        self.draw_likelihood = draw_likelihood
        self.draw_heading    = draw_heading

        self.lik_patch    = None
        self.heading_line = None
        self.title        = None

    def init(self, field, robot):

        if not hasattr(robot, 'gradient'):
            raise AttributeError(
                f"BeliefPolarPlot requires a BayesianHeading robot; "
                f"'{self.robot_name}' has no 'gradient' attribute."
            )

        gradient  = robot.gradient
        N         = gradient.n_angles
        angles    = gradient.angles
        bar_width = (2 * np.pi / N) * 0.85

        # Create two axis, for confidence and polar
        # ---------------------------------------------------------
        subplot_spec = self.ax.get_subplotspec()
        self.ax.remove()

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplot_spec, width_ratios=[4, 1], wspace=0.6
        )

        self.ax_polar = self.fig.add_subplot(inner_gs[0], projection="polar")
        self.ax_bar   = self.fig.add_subplot(inner_gs[1])

        # Belief Bars (Polar)
        # ---------------------------------------------------------
        self.bars = self.ax_polar.bar(
            angles,
            np.ones(N),
            width     = bar_width,
            bottom    = 0.0,
            color     =  odor_c,
            alpha     = 0.85,
            edgecolor = "darkgoldenrod",
            linewidth = 0.5,
            zorder    = 3,
        )

        # Uniform Reference Ring
        # ---------------------------------------------------------
        _t = np.linspace(0, 2 * np.pi, 300)
        self.ax_polar.plot(
            _t, np.ones_like(_t),
            lw=1.5, ls=':', color='gray', alpha=0.5, zorder=1
        )

        # Heading Arrow
        # ---------------------------------------------------------
        if self.draw_heading:
            self.heading_line, = self.ax_polar.plot(
                [], [],
                color= robot_c, lw=2.5, solid_capstyle='round', zorder=5
            )
            self.legend_items.append((self.heading_line, "Robot heading"))

        # Legend
        # ---------------------------------------------------------
        self.legend_items.append((self.bars.patches[0], "Directional belief"))

        if self.draw_likelihood:
            self.legend_items.append((
                Patch(color='purple', alpha=0.25),
                "Likelihood (next step)"
            ))

        # Axis
        # ---------------------------------------------------------
        self.ax_polar.set_theta_zero_location('E')   
        self.ax_polar.set_theta_direction(1)         
        self.ax_polar.set_yticklabels([])            
        self.ax_polar.spines['polar'].set_visible(False)
        self.ax_polar.set_rlabel_position(90)

        self.title = self.ax_polar.set_title(
            "Directional belief", pad=14
        )

        # Confidence Bar (Rectangular)
        # ---------------------------------------------------------
        self.conf_bar = self.ax_bar.bar([""], [0], color="gold", width=0.5)

        self.ax_bar.set_ylim(0, 2.0)
        self.ax_bar.set_xlim(-0.5, 0.5)

        self.ax_bar.set_xlabel("Confidence", labelpad=5)

        self.conf_text = self.ax_bar.text(
            0, 0.05, "0.00",
            ha="center", va="bottom", fontweight="bold"
        )

        conf_lost = getattr(robot, "confidence_low", 0.2)
        self.ax_bar.axhline(conf_lost, color="red", linestyle=":", alpha=0.5)
        self.ax_bar.axhline(1.0, color="green", linestyle=":", alpha=0.5)

        self.ax_bar.spines['top'].set_visible(False)
        self.ax_bar.spines['right'].set_visible(False)
        self.ax_bar.spines['bottom'].set_visible(False)

    # ---------------------------------------------------------
    # UPDATE FRAME
    # ---------------------------------------------------------
    def update(self, state: SimState):

        robot = state.robots[self.robot_name]

        if robot.found_odor:
            return []

        gradient = robot.gradient
        belief   = gradient.belief
        N        = gradient.n_angles
        angles   = gradient.angles

        # Exaggerate Belief Heights Visually
        # ---------------------------------------------------------
        raw_heights = belief * N
        h_min = float(raw_heights.min())
        h_max = float(raw_heights.max())
        
        if h_max > h_min:
            visual_heights = 0.2 + 1.3 * ((raw_heights - h_min) / (h_max - h_min))
        else:
            visual_heights = raw_heights
            
        viz_max_h = 1.5 

        for bar, h in zip(self.bars, visual_heights):
            bar.set_height(h)

        self.ax_polar.set_ylim(0, viz_max_h * 1.2)

        # Exaggerate Likelihood 
        # ---------------------------------------------------------
        if self.draw_likelihood:
            if gradient.likelihood_array is not None:
                
                if self.lik_patch is not None:
                    self.lik_patch.remove()
                    self.lik_patch = None
                
                lik = gradient.likelihood_array
                    
                lik_min, lik_max = lik.min(), lik.max()
                if lik_max > lik_min:
                    lik_norm = 0.2 + ((lik - lik_min) / (lik_max - lik_min)) * (viz_max_h * 0.8)
                else:
                    lik_norm = np.full_like(lik, viz_max_h * 0.8)

                thetas_c = np.append(angles, angles[0])
                lik_c    = np.append(lik_norm, lik_norm[0])

                self.lik_patch, = self.ax_polar.fill(
                    thetas_c, lik_c,
                    color='purple', alpha=0.5, zorder=2
                )

        # Update Heading Arrow
        # ---------------------------------------------------------
        if self.draw_heading:
            move_theta = np.deg2rad(robot.pos[3])
            self.heading_line.set_data(
                [move_theta, move_theta], [0.0, viz_max_h * 0.88]
            )
            
        # Title formatting
        # ---------------------------------------------------------
        behavior     = robot.state.label
        confidence   = float(getattr(robot, 'confidence', 0.0))
        entropy_norm = float(
            -np.sum(belief * np.log(belief + 1e-12)) / np.log(N)
        )

        self.title.set_text(f"State: {behavior}")

        # Confidence Bar Update
        # ---------------------------------------------------------
        conf = getattr(robot, "Confidence", getattr(gradient, "confidence", 0.0))

        self.conf_bar[0].set_height(conf)
        self.conf_text.set_text(f"{conf:.2f}")
        self.conf_text.set_position((0, conf + 0.05))

        return (
            list(self.bars)
            + ([self.lik_patch]    if self.lik_patch    is not None else [])
            + ([self.heading_line] if self.draw_heading               else [])
            + [self.conf_bar[0], self.conf_text, self.title]
        )
        
# ============================================================
# MASTER VISUALIZER CLASS
# ============================================================

class Visualizer:

    # ---------------------------------------------------------
    # INITIALIZATION & LAYOUT
    # ---------------------------------------------------------
    def __init__(self, runner, plots, figsize, n_col = 2, show = True):
                    
        self.plots = plots
        self.runner = runner
        self.show = show

        self.field = next(iter(runner.fields.values()))
        
        # Build Grid Layout
        # ---------------------------------------------------------
        n = len(plots)
        cols = min(n_col, 3)
        rows = (n + cols - 1) // cols
        
        self.fig = plt.figure(figsize=(figsize[0] * cols, figsize[1] * rows))
        axes = []

        # change pot type based on class in plot
        for i, plot in enumerate(plots):

            if isinstance(plot, Room3DView):
                ax = self.fig.add_subplot(rows, cols, i + 1, projection='3d')
            elif getattr(plot, "needs_polar", False):
                ax = self.fig.add_subplot(rows, cols, i +1, projection="polar")
            else:
                ax = self.fig.add_subplot(rows, cols, i + 1)

            axes.append(ax)

        # Bind Axes
        # ---------------------------------------------------------
        for plot, ax in zip(plots, axes):
            plot.ax = ax
            plot.fig = self.fig

        # Hide empty subplots
        for ax in axes[n:]:
            ax.set_visible(False)
            
        # Init plots
        # ---------------------------------------------------------
        for p in plots:
            field_name = p.field_name
            p.init(runner.fields[field_name],
                   runner.robots[p.robot_name])

        
        # Align Aspect Ratios perfectly for heat maps, so there same size
        # ---------------------------------------------------------
        heatmap_plots = [p for p in self.plots if isinstance(p, ConcHeatMap2D)]
        
        if len(heatmap_plots) > 1:
            max_span = max(self.field.x.max(), self.field.y.max() , self.field.z.max())
            
            for p in heatmap_plots:
                xl, xr = p.ax.get_xlim()
                yb, yt = p.ax.get_ylim()

                x_mid = (xl + xr) / 2
                y_mid = (yb + yt) / 2

                p.ax.set_xlim(x_mid - max_span / 2, x_mid + max_span / 2)
                p.ax.set_ylim(y_mid - max_span / 2, y_mid + max_span / 2)
                
        # one central legends
        self.build_legend()
                
    # ---------------------------------------------------------
    # ANIMATION STEP
    # ---------------------------------------------------------
    def update_frame(self, frame_number):
        """Updates all subplots per frame using the runner step."""
        
        state = self.runner.step()
        
        if state.finished:
            print("Finished")
            self.ani.event_source.stop()
            plt.close(self.fig)
        
        #update all plots
        artists = []
        for p in self.plots:
            artists += p.update(state)
        #general plot title
        self.fig.suptitle(f"Simulation time: {state.sim_time:.2f} s \n"
                          f"Flow velocity at inlet: {self.field.flow_vel[self.runner.t_idx]:.2f} [m/s] \n "
                          )
        
        self.fig.subplots_adjust(top=0.85)

        return artists
    
    # ---------------------------------------------------------
    # LEGEND CONSTRUCTION
    # ---------------------------------------------------------
    def build_legend(self):
        handles = []
        labels = []

        for p in self.plots:
            for h, l in p.legend_items_func():
                handles.append(h)
                labels.append(l)

        # Remove duplicates
        seen = set()
        uniq_handles = []
        uniq_labels = []

        
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)

        num_cols = (len(uniq_labels) + 1) // 2 

        self.fig.legend(
            uniq_handles, 
            uniq_labels, 
            loc="lower center", 
            bbox_to_anchor=(0.5, 0.02),
            fontsize=12, 
            facecolor='grey', 
            edgecolor='white', 
            framealpha=0.6, 
            ncol=num_cols
        )
        
        self.fig.subplots_adjust(bottom=0.25)
        
    # ---------------------------------------------------------
    # ANIMATION 
    # ---------------------------------------------------------
    def animate(self, interval=50, fig=None):

        self.ani = FuncAnimation(
            fig,
            self.update_frame,
            frames = self.runner.tot_frames,
            interval=interval,
            blit=False 
        )
        
        if self.show:
            plt.show()
            
        return self.ani