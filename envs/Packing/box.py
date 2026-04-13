
class Box(object):
    def __init__(self, length, width, height, x, y, z, weight=1.0, fragility=0):
        # dimension(x, y, z) + position(lx, ly, lz)
        self.size_x = length
        self.size_y = width
        self.size_z = height
        self.pos_x = x
        self.pos_y = y
        self.pos_z = z
        self.weight = weight
        self.fragility = fragility  # 0: normal, 1: fragile (nothing can be placed on top)

    def standardize(self):
        """

        Returns:
            tuple(size + position)
        """
        return tuple([self.size_x, self.size_y, self.size_z, self.pos_x, self.pos_y, self.pos_z])
