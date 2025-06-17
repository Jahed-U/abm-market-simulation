# 🧠 Agent-Based Market Simulation in Python

This project simulates an artificial stock market using Agent-Based Modelling (ABM), based on my MSc dissertation in Mathematics. It explores how different classes of traders interact and impact market dynamics such as price behaviour, volatility, and stability.

Built in Python using the Mesa framework, this project allows us to test how learning agents and technical trading strategies affect market behaviour — and whether they create bubbles, crashes, or stable equilibria.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `Baseline 2.py` | Baseline model with normal agents only |
| `Learning Agent.py` | Adds a second class of agents with adaptive learning |
| `TA Agent Sim.py` | Includes technical momentum & reversal agents |
| `results/` | Output plots for each simulation run |
| `requirements.txt` | List of required Python packages |
| `paper_summary.pdf` | 1-page summary of the full research paper |

---

## 🔬 Models Overview

### 1️⃣ Baseline Model – `Baseline 2.py`
- 1000 Normal Agents (NAs)
- Agents use a mix of fundamental and technical rules
- Result: Price remains close to the fundamental value

### 2️⃣ Learning Agent Model – `Learning Agent.py`
- Adds 1000 Learning Agents (NA2s) that adjust strategy weights over time
- Result: Increased volatility, downward trend in prices

### 3️⃣ Technical Agents Model – `TA Agent Sim.py`
- Adds two advanced traders:
  - TA-m (momentum trader): buys on uptrends
  - TA-r (reversal trader): short sells on uptrends
- Result: 
  - TA-m alone → market bubbles  
  - TA-r alone → market crashes  
  - Both together → stabilisation & balance

---

## 📈 Key Results

Visual outputs from simulations (in `/results/` folder):

### 🔹 Baseline Simulations
- `b2 r1.png`
- `b2 r2.png`
- `b2 r3.png`

### 🔹 Learning Agent Simulations
- `LA r1.png`
- `LA r2.png`
- `LA r3.png`

### 🔹 Technical Agent Simulations
- `TAm Sim 1.png`, `TAm Sim 2.png`, `TAm Sim 3.png` – Momentum agent  
- `TAr Sim 1.png`, `TAr Sim 2.png`, `TAr Sim 3.png` – Reversal agent  
- `TArm Sim 1.png`, `TArm Sim 2.png`, `TArm Sim 3.png` – Both agents together  

---

## ⚙️ How to Run

1. Clone the repo or download the files  
2. Install the required packages:  
   `pip install -r requirements.txt`  

3. Run any of the models:  
   `python "TA Agent Sim.py"`  

> Make sure you have Python 3.7+ installed. The Mesa library is used for simulation.

---

## 📚 About the Research

This project is based on my MSc dissertation:  
> **An Overview of Agent-Based Modelling with Applications in Finance and Economics**

The simulation investigates emergent behaviours in artificial markets — such as price instability caused by aggressive technical trading — and shows how different agent behaviours interact to form market patterns.

📄 See `paper_summary.pdf` for a 1-page summary.

---

## ✍️ Author

**Jahed Ullah**  
GitHub: [Jahed-U](https://github.com/Jahed-U)

---

## 🧠 Future Work

- Add Sharpe ratio, autocorrelation, and volatility clustering metrics  
- Calibrate models to real stock price data  
- Explore execution strategies using limit order books  
- Deploy on QuantConnect for backtesting

---
