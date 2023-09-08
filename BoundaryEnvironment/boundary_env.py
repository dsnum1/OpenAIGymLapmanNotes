from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.manual_control import ManualControl
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
import random

class SimpleEnv(MiniGridEnv):
    def __init__(self,
                agent_start_pos=(1,1),
                agent_start_dir=0,
                width=10,
                height=10,
                max_steps: int | None = None,
                **kwargs,
                ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.width = width
        self.height=height
        mission_space = MissionSpace(mission_func=self._gen_mission)
        # if max_steps is None:
        #     max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=width,
            max_steps=256,
            **kwargs,
        )
    @staticmethod
    def _gen_mission():
        return "grand mission"
    
    # MiniGridEnv._gen_grid
    def _gen_grid(self, width, height):
        self.grid = Grid(self.width, self.height)
        self.grid.wall_rect(0, 0, self.width, self.height)
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.put_obj(Goal(), self.width - 2, self.height - 2)

        self.mission = "Boundary Set Up"
        # Generate verical separation wall
        for i in range(0, self.height):
            if(i != self.height//2):
                self.grid.set(self.width//2, i, Wall())
        


        pass
def main():
    env = SimpleEnv(render_mode="human", width=9, height=9)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()
