import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, num_bandits=10):
        self.probs = np.random.uniform(size=num_bandits)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.num_bandits = num_bandits

    def step(self, k): # reward distribution
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.num_bandits)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def updata_regret(self, t):
        self.regret += self.bandit.best_prob -  self.bandit.probs[t] # 这是期望形式
        self.regrets.append(self.regret)
    
    # 由具体策略进行选择
    def run_one_step(self):
        return NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.updata_regret(k)


class epsilon_greedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(epsilon_greedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.num_bandits)
    
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.num_bandits) # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)
        reward_k = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (reward_k - self.estimates[k])
        return k


class declayEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(declayEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.num_bandits)
        self.total_count = 0
    
    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.num_bandits)
        else:
            k = np.argmax(self.estimates)
        
        reward_k = self.bandit.step(k)
        self.estimates[k] += 1 / (self.counts[k] + 1) * (reward_k - self.estimates[k])

        return k


def plot_results(solvers, solvers_names):
    for idx, solver in enumerate(solvers):
        time_lst = range(len(solver.regrets))
        plt.plot(time_lst, solver.regrets, label=solvers_names[idx])
    plt.xlabel("time steps")
    plt.ylabel("cumulative regrets")
    plt.title(f"{solvers[0].bandit.num_bandits}-armed bandit")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # random seed matters
    np.random.seed(1)
    epsilon_greedy_solver = epsilon_greedy(bandit=BernoulliBandit(num_bandits=10), epsilon=0.01)
    declayEpsilonGreedy_solver = epsilon_greedy(bandit=BernoulliBandit(num_bandits=10))
    solvers = [epsilon_greedy_solver, declayEpsilonGreedy_solver]
    solvers_names = ["epsilon_greedy", "declayEpsilonGreedy"]
    for solver in solvers:
        solver.run(5000)
    plot_results(solvers, solvers_names)