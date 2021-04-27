class Iterator:
    # start: array of start position
    # stop: array of stop position
    # step: array of step position
    # All the length must be same
    def __init__(self, start, stop, step):
        self.start = [ *start ]
        self.stop = [ *stop ]
        self.step = [ *step ]

        #print('start', self.start)
        #print('stop', self.stop)
        #print('step', self.step)

        self.dim = len(self.start)

        assert self.dim == len(self.stop)
        assert self.dim == len(self.step)

        self.index = [ *self.start ]
        self.index[-1] -= self.step[-1]

    def next(self):
        for i in range(self.dim - 1, -1, -1):
            self.index[i] += self.step[i]
            if self.index[i] < self.stop[i]:
                return True
            else:
                self.index[i] = self.start[i]

        return False
