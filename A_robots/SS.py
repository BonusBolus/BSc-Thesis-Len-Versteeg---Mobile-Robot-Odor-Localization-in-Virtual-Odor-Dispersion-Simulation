from A_robots.BaseRobot import BaseRobot
import numpy as np
from dataclasses import dataclass

@dataclass
class TargetResult:
    x_new: float
    y_new: float
    rot_new: float
    

class SS(BaseRobot):

    def __init__(self, **kwargs):

        # 1. Initialize base robot with pysics logic
        super().__init__(**kwargs)
    
    
    ### Overwrite excecuteMove to never move
    def executeMove(self):
        # override base movement completely
        self.target_pos[:] = self.pos
        self.target_reached = True
        
    def step(self, visualize = True):
        
        pass
