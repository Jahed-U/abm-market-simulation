#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:04:21 2024

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
            self.model.execute_order('buy')
        elif P_e_j_t < P_o_j_t:
            self.model.execute_order('sell')


class StockMarketModel(Model):
    def __init__(self, N=1000):
        self.num_agents = N
        self.schedule = BaseScheduler(self)
        self.Pf = 10000
        self.Pd = 1000
        self.tau_max = 10000
        self.sigma_epsilon = 0.03
        self.P = [10000]
        for i in range(self.num_agents):
            a = NormalAgent(i, self)
            self.schedule.add(a)

    def execute_order(self, order_type):
        if order_type == 'buy':
            self.P.append(self.P[-1] + 1)
        elif order_type == 'sell':
            self.P.append(self.P[-1] - 1)

    def step(self):
        for agent in sorted(self.schedule.agents, key=lambda a: a.unique_id):
            agent.step()
        # Print the stock price at the current step
        print(f'Stock Price at step {len(self.P) - 1}: {self.P[-1]}')


def run_simulation(steps=5000):
    model = StockMarketModel()
    for i in range(steps):
        model.step()

    plt.figure(figsize=(10, 6))
    plt.plot(model.P, label='Stock Price')
    plt.title('Stock Price Simulation Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.ylim([min(model.P) - 1000, max(model.P) + 1000])
    plt.show()


run_simulation(steps=5000)
