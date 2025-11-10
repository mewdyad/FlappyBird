# Flappy Bird AI

A minimalist reinforcement learning simulation of **Flappy Bird** written in Python.  
The AI learns to play through genetic evolution and mutation over generations, refining its neural network weights with each iteration.  
Includes visualizations for neural activations, performance graphs, and adaptive difficulty that scales as the AI improves.

---

## Features
- **Genetic Algorithm:** Population-based evolution of neural networks.  
- **Dynamic Difficulty:** Gradually increases as the AI scores higher.  
- **Neural Visualization:** Real-time rendering of neuron activations and connections.  
- **Fast Forward Mode:** Accelerate simulation for quicker evolution.  
- **Human Mode:** Play manually to test your instincts against the AI.  

---

## Controls
| Key | Action |
| --- | --- |
| **Space** | Pause / Resume |
| **F** | Toggle Fast Forward |
| **N** | Toggle Neural Network Visualization |
| **R** | Reset Population |
| **X** | Human Mode (Play manually) |
| **Mouse Click** | Inspect a specific bird |

---

## Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/flappy-bird-ai.git
cd flappy-bird-ai
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install pygame numpy
```

(Optional) You can export dependencies:
```bash
pip freeze > requirements.txt
```

To install from a requirements file:
```bash
pip install -r requirements.txt
```

---

## Run the Project
```bash
python flappy_bird_ai.py
```

---

## Project Structure
```
flappy_bird_ai.py         # Core simulation and AI logic
flappy_ai_champion.pkl    # Auto-saved best performing neural network
requirements.txt          # Optional dependency list
```

---

## Notes
- The AI evolves entirely through selection, crossover, and mutation — no backpropagation.  
- Elite networks are preserved each generation; weaker ones mutate toward better performance.  
- The best-performing neural network is saved automatically and reused for the next session.  
- Everything — from evolution to cognition — is visual and interactive.
