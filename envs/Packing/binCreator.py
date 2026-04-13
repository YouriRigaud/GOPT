import numpy as np
import copy
import torch


class BoxCreator(object):
    def __init__(self):
        self.box_list = []  # generated box list

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        """
        :param length:
        :return: list
        """
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)


class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                default_box_set.append((2 + i, 2 + j, 2 + k))

    def __init__(
        self,
        box_size_set=None,
        use_weight=False,
        weight_range=(0.5, 5.0),
        use_fragility=False,
        fragility_probability=0.3,
    ):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set
        self.use_weight = use_weight
        self.weight_range = weight_range
        self.use_fragility = use_fragility
        self.fragility_probability = fragility_probability

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        l, w, h = self.box_set[idx][:3]
        # TODO: consider alternative weight distributions (ex: correlated with volume)
        weight = float(np.random.uniform(*self.weight_range)) if self.use_weight else 1.0
        fragility = int(np.random.rand() < self.fragility_probability) if self.use_fragility else 0
        self.box_list.append((l, w, h, weight, fragility))


# load data
class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):  # data url
        super().__init__()  
        self.data_name = data_name
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))  
        print("load data set successfully, data name: ", self.data_name)

    def reset(self, index=None):
        self.box_list.clear()
        box_trajs = torch.load(self.data_name)
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = box_trajs[self.index]
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1
