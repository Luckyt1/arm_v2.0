import numpy as np
class PDController:
    def __init__(self, kp, kd, limit=None):
        self.kp = kp
        self.kd = kd
        self.limit = limit
        if limit is not None:
            self.has_limit = True
        else:
            self.has_limit = False

    def update(self, x, xd, x_ref, xd_ref=0.0):
        output = self.kp * (x_ref - x) + self.kd * (xd_ref - xd)
        if self.has_limit:
            output = np.clip(output, -self.limit, self.limit)
        return output