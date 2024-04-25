import traceback
from ..perception.localization.localization_module import LocalizationModule
from ..perception.birds_view.birds_view_semantic_segmentation import BirdsViewSemanticSegmentation
from ..perception.lift_splat.lift_splat import LiftSplat


class Agent(object):
    def __init__(self, env=None):
        self.env = env

    def tick(self):
        return


class AutonomousVehicle(Agent):
    def __init__(self, env=None, vehicle=None):
        super().__init__(env)

        self.vehicle = vehicle

        # try:
        #     self.localization_model = LocalizationModule(self)
        # except Exception as e:
        #     traceback.print_exc()

        # try:
        #     self.lift_splat = LiftSplat(self.vehicle)
        # except Exception as e:
        #     traceback.print_exc()

        # self.birds_view_model = BirdsViewSemanticSegmentation(self.vehicle)

        if self.env is not None:
            self.env.add_tick_callback(self.tick)

    def tick(self):
        super().tick()

        try:
            if hasattr(self, 'localization_model'):
                self.localization_model.tick()
            if hasattr(self, 'lift_splat'):
                self.lift_splat.tick()
            if hasattr(self, 'birds_view_model'):
                self.birds_view_model.tick()
        except Exception as e:
            print(traceback.format_exc())

    def on_close (self):
        self.env.remove_tick_callback(self.tick)
        self.env.despawn_agent_vehicle(self.vehicle)

