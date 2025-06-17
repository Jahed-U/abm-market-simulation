#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:48:52 2024

@author: jahed
"""

import numpy as np
import matplotlib.pyplot as plt
from mesa import Model, Agent
from mesa.time import BaseScheduler


class NormalAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.w1 = np.random.uniform(0, 1)
        self.w2 = np.random.uniform(0, 100)
        self.w3 = np.random.uniform(0, 1)
        self.tau = np.random.randint(1, model.tau_max + 1)
        self.epsilon = np.random.normal(0, model.sigma_epsilon)

    def step(self):
        if len(self.model.P) > self.tau:
            P_t_minus_tau_minus_1 = self.model.P[-self.tau - 1]
        else:
            P_t_minus_tau_minus_1 = self.model.P[0]
        Pf = self.model.Pf
        P_t_minus_1 = self.model.P[-1]
        numerator = (
            self.w1 * np.log(Pf / P_t_minus_1) +
            self.w2 * np.log(P_t_minus_1 / P_t_minus_tau_minus_1) +
            self.w3 * self.epsilon
        )
        denominator = self.w1 + self.w2 + self.w3
        r_e_j_t = numerator / denominator
        P_e_j_t = P_t_minus_1 * np.exp(r_e_j_t)
        rho_j_t = np.random.rand()
        P_o_j_t = P_e_j_t + self.model.Pd * (2 * rho_j_t - 1)
        if P_e_j_t > P_o_j_t:
            self.model.execute_order('buy', 1)
        elif P_e_j_t < P_o_j_t:
            self.model.execute_order('sell', 1)


class LearningAgent(NormalAgent):
    def __init__(self, unique_id, model, k=4):
        super().__init__(unique_id, model)
        self.k = k

    def adjust_weights(self, r_l_t):
        u_j_t = np.random.uniform(0, 1)
        weights = [self.w1, self.w2, self.w3]
        max_weights = [1, 100, 1]  # w1_max, w2_max, w3_max
        for i in range(3):
            if np.sign(r_l_t) == np.sign(self.epsilon):
                weights[i] += self.k * r_l_t * u_j_t * \
                    (max_weights[i] - weights[i])
            else:
                weights[i] -= self.k * r_l_t * u_j_t * weights[i]
            # Ensure weights remain within bounds
            weights[i] = max(min(weights[i], max_weights[i]), 0)
        self.w1, self.w2, self.w3 = weights

    def step(self):
        super().step()
        if len(self.model.P) > 100:
            P_t = self.model.P[-1]
            P_t_t_l = self.model.P[-101]
            r_l_t = np.log(P_t / P_t_t_l)
            self.adjust_weights(r_l_t)


class TechnicalMomentumAgent(Agent):
    def __init__(self, unique_id, model, A=100, evaluation_period=100):
        super().__init__(unique_id, model)
        self.A = A
        self.evaluation_period = evaluation_period
        self.tm_range = np.arange(5, 101)
        self.best_tm = np.random.choice(self.tm_range)
        self.performance_record = {tm: 0 for tm in self.tm_range}

    def evaluate_tm(self):
        best_performance = -np.inf
        for tm, performance in self.performance_record.items():
            if performance > best_performance:
                best_performance = performance
                self.best_tm = tm
        # Reset performance record for the next evaluation period
        self.performance_record = {tm: 0 for tm in self.tm_range}

    def step(self):
        if len(self.model.P) > self.best_tm:
            P_t = self.model.P[-1]
            P_t_tm = self.model.P[-self.best_tm - 1]
            if P_t > P_t_tm:
                self.model.execute_order('buy', self.A)
                self.performance_record[self.best_tm] += (P_t - P_t_tm)
            elif P_t < P_t_tm:
                self.model.execute_order('sell', self.A)
                self.performance_record[self.best_tm] += (P_t_tm - P_t)

            # Evaluate and adjust t_m every evaluation_period steps
            if len(self.model.P) % self.evaluation_period == 0:
                self.evaluate_tm()


class TechnicalReversalAgent(Agent):
    def __init__(self, unique_id, model, A=100, evaluation_period=100):
        super().__init__(unique_id, model)
        self.A = A
        self.evaluation_period = evaluation_period
        self.tr_range = np.arange(5, 101)
        self.best_tr = np.random.choice(self.tr_range)
        self.performance_record = {tr: 0 for tr in self.tr_range}

    def evaluate_tr(self):
        best_performance = -np.inf
        for tr, performance in self.performance_record.items():
            if performance > best_performance:
                best_performance = performance
                self.best_tr = tr
        # Reset performance record for the next evaluation period
        self.performance_record = {tr: 0 for tr in self.tr_range}

    def step(self):
        if len(self.model.P) > self.best_tr:
            P_t = self.model.P[-1]
            P_t_tr = self.model.P[-self.best_tr - 1]
            if P_t < P_t_tr:
                self.model.execute_order('buy', self.A)
                self.performance_record[self.best_tr] += (P_t_tr - P_t)
            elif P_t > P_t_tr:
                self.model.execute_order('sell', self.A)
                self.performance_record[self.best_tr] += (P_t - P_t_tr)

            # Evaluate and adjust t_r every evaluation_period steps
            if len(self.model.P) % self.evaluation_period == 0:
                self.evaluate_tr()


class StockMarketModel(Model):
    def __init__(self, N=1000, N_learning=1000, N_technical=1, N_reversal=1):
        self.num_agents = N
        self.schedule = BaseScheduler(self)
        self.Pf = 10000
        self.Pd = 1000
        self.tau_max = 10000
        self.sigma_epsilon = 0.03
        self.P = [10000]
        for i in range(self.num_agents - N_learning):
            a = NormalAgent(i, self)
            self.schedule.add(a)
        for i in range(self.num_agents - N_learning, self.num_agents):
            a = LearningAgent(i, self)
            self.schedule.add(a)
        for _ in range(N_technical):
            a = TechnicalMomentumAgent(self.num_agents, self)
            self.schedule.add(a)
        for _ in range(N_reversal):
            a = TechnicalReversalAgent(self.num_agents + N_technical, self)
            self.schedule.add(a)

    def execute_order(self, order_type, shares=1):
        if order_type == 'buy':
            self.P.append(self.P[-1] + shares * 1)  # Adjusted for shares
        elif order_type == 'sell':
            self.P.append(self.P[-1] - shares * 1)

    def step(self):
        for agent in sorted(self.schedule.agents, key=lambda a: a.unique_id):
            agent.step()
        print(f'Stock Price at step {len(self.P) - 1}: {self.P[-1]}')


def run_simulation(steps=1000):
    model = StockMarketModel(N=1000, N_learning=100, N_technical=1,
                             N_reversal=1)
    for i in range(steps):
        model.step()

    plt.figure(figsize=(10, 6))
    plt.plot(model.P, label='Stock Price')
    plt.title('Stock Price Simulation Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    # Adjust y-axis scale as needed
    plt.ylim([min(model.P) - 1000, max(model.P) + 1000])
    plt.show()
    average_stock_price = np.mean(model.P)
    print(f'Average Stock Price: {average_stock_price}')


run_simulation(steps=5000)
