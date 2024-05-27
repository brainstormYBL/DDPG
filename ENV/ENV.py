import numpy as np


class ENV:
    def __int__(self, par):
        self.par = par
        self.num_slot = self.par.num_slot
        self.tra_rw_uav = self.generate_rw_uav_trajectory()
        self.tra_fw_uav = self.generate_fw_uav_trajectory()
        self.dim_action = 1
        self.dim_state = 4 + self.par.num_rw_uav

    def reset(self):
        pass

    def step(self, action):
        pass

    def generate_rw_uav_trajectory(self):
        # The trajectory of RW-UAVs, consisting: [UAV ID, time slot index, x, y, z]
        rw_uav_trajectory = np.zeros((self.par.num_rw_uav, self.par.num_slot, 3))
        for index_rw_uav in range(self.par.num_rw_uav):
            for index_slot in range(self.par.num_slot):
                rw_uav_trajectory[index_rw_uav, index_slot, 0] = 
