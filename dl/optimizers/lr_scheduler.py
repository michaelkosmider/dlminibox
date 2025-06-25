class LRScheduler:

    def __init__(self, optimizer, schedule):
        self.optimizer = optimizer
        self.schedule = schedule
        self.current_step = 0

    def step(self):

        if self.current_step in self.schedule:
            self.optimizer.learning_rate = self.schedule[self.current_step]

        self.current_step += 1
