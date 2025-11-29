import numpy as np
import time



class CLMetrics:

    def __init__(self, acc_matrix):
        self.R = np.array(acc_matrix)
        self.T = self.R.shape[0]

    def avg_accuracy(self):
        return np.mean(self.R[-1, :])

    def forgetting(self):
        if self.T <= 1:
            return 0.0
        forget = 0.0
        for j in range(self.T - 1):
            max_acc = np.max(self.R[j:, j])
            final_acc = self.R[-1, j]
            forget += max_acc - final_acc
        return forget / (self.T - 1)

    def bwt(self):
        if self.T <= 1:
            return 0.0
        bwt = 0.0
        for j in range(self.T - 1):
            bwt += self.R[-1, j] - self.R[j, j]
        return bwt / (self.T - 1)

    def fwt(self, rb=0.1):
        if self.T <= 1:
            return 0.0
        fwt = 0.0
        for j in range(1, self.T):
            fwt += self.R[j-1, j] - rb
        return fwt / (self.T - 1)


    def plasticity(self):
        return np.mean([self.R[j, j] for j in range(self.T)])


    def all_metrics(self):
        return {
            'accuracy': self.avg_accuracy(),
            'forgetting': self.forgetting(),
            'bwt': self.bwt(),
            'fwt': self.fwt(),
            'plasticity': self.plasticity()
        }


class CompMetrics:
    def __init__(self):
        self.task_times = []

    def start_timer(self):
        self.start = time.time()

    def end_timer(self):
        elapsed = time.time() - self.start
        self.task_times.append(elapsed)
        return elapsed


    def summary(self):
        return {
            'avg_task_time': np.mean(self.task_times) if self.task_times else 0.0,
            'total_training_time': sum(self.task_times),
        }
