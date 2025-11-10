import pygame
import random
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict

# Initialize pygame
pygame.init()

# Game constants
WIDTH, HEIGHT = 800, 600
BASE_GRAVITY = 0.25
BASE_JUMP_STRENGTH = -5.5
BASE_PIPE_SPEED = 3.5
BASE_PIPE_GAP = 180
BASE_PIPE_FREQUENCY = 1500
PIPE_WIDTH = 70

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (40, 200, 40)
BLUE = (30, 144, 255)
RED = (255, 69, 0)
GOLD = (255, 215, 0)
BACKGROUND = (5, 5, 25)
NEURAL_COLORS = [
    (255, 100, 100),  # Input layer
    (100, 255, 100),  # Hidden layer
    (100, 100, 255)   # Output layer
]

# AI Constants
POPULATION_SIZE = 1000
ELITE_SIZE = 10
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
INPUT_NODES = 7
HIDDEN_NODES = 10
OUTPUT_NODES = 1
SAVE_FILE = "flappy_ai_champion.pkl"

class DynamicDifficulty:
    def __init__(self):
        self.base_score = 0
        self.difficulty_level = 1
        
    def update(self, current_score: int) -> None:
        self.difficulty_level = 1 + (current_score // 50) * 0.1
        self.difficulty_level = min(self.difficulty_level, 3.0)
        
    def get_pipe_gap(self) -> int:
        return max(120, int(BASE_PIPE_GAP * (1.5 - (self.difficulty_level * 0.15))))
        
    def get_pipe_speed(self) -> float:
        return min(6.0, BASE_PIPE_SPEED * (1 + (self.difficulty_level * 0.3)))
        
    def get_pipe_frequency(self) -> int:
        return max(800, int(BASE_PIPE_FREQUENCY * (1 - (self.difficulty_level * 0.1))))
        
    def get_gravity(self) -> float:
        return min(0.4, BASE_GRAVITY * (1 + (self.difficulty_level * 0.1)))

class Bird:
    def __init__(self, x: int, y: int, brain: Optional['NeuralNetwork'] = None):
        self.x = x
        self.y = y
        self.velocity = 0
        self.radius = 8
        self.alive = True
        self.score = 0
        self.fitness = 0
        self.brain = brain if brain else NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
        self.color = BLUE
        self.genome_id = random.getrandbits(64)
        self.last_flap_time = 0
        self.flap_cooldown = 100
        
    def jump(self, gravity: float) -> None:
        current_time = pygame.time.get_ticks()
        if current_time - self.last_flap_time > self.flap_cooldown:
            jump_strength = BASE_JUMP_STRENGTH * (0.8 + (gravity / BASE_GRAVITY) * 0.2)
            self.velocity = jump_strength
            self.last_flap_time = current_time
        
    def update(self, pipes: List['Pipe'], gravity: float, human_mode: bool = False) -> None:
        if not self.alive:
            return
            
        if human_mode:
            self.velocity += gravity
            self.y += self.velocity
            if self.y >= HEIGHT - self.radius or self.y <= self.radius:
                self.alive = False
            return
            
        closest_pipe = None
        for pipe in pipes:
            if pipe.x + pipe.width > self.x - self.radius:
                closest_pipe = pipe
                break
        
        inputs = np.zeros(INPUT_NODES)
        inputs[0] = self.x / WIDTH
        inputs[1] = self.y / HEIGHT
        inputs[2] = self.velocity / 10
        
        if closest_pipe:
            gap_center = closest_pipe.top_height + (closest_pipe.y - closest_pipe.top_height)/2
            inputs[3] = closest_pipe.top_height / HEIGHT
            inputs[4] = closest_pipe.y / HEIGHT
            inputs[5] = closest_pipe.x / WIDTH
            inputs[6] = abs(self.y - gap_center) / HEIGHT
        else:
            inputs[3:7] = [0.5, 0.5, 1.0, 0.1]
        
        decision = self.brain.predict(inputs)[0]
        if decision > 0.7:
            self.jump(gravity)
            
        self.velocity += gravity
        self.y += self.velocity
        
        if self.y >= HEIGHT - self.radius or self.y <= self.radius:
            self.alive = False
            
    def draw(self, screen: pygame.Surface, alpha: int = 255, human_mode: bool = False) -> None:
        if not self.alive:
            return
            
        color = self.color if not human_mode else (255, 100, 100)
        radius = self.radius if not human_mode else 10
        
        if pygame.version.vernum[0] >= 2:
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
            
            if human_mode:
                wing_pos = radius + 5 if int(pygame.time.get_ticks() / 100) % 2 == 0 else radius - 5
                pygame.draw.circle(s, (*color, alpha), (wing_pos, radius), radius//2)
            
            screen.blit(s, (int(self.x)-radius, int(self.y)-radius))
        else:
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), radius)

class Pipe:
    def __init__(self, gap: int, speed: float):
        self.width = PIPE_WIDTH
        self.x = WIDTH
        self.top_height = random.randint(100, HEIGHT - gap - 100)
        self.y = self.top_height + gap
        self.passed = False
        self.highlight = False
        self.speed = speed
        
    def update(self) -> None:
        self.x -= self.speed
        
    def draw(self, screen: pygame.Surface, human_mode: bool = False) -> None:
        if human_mode:
            for i in range(self.width):
                color_top = (
                    max(0, GREEN[0] - i//3),
                    max(0, GREEN[1] - i//3),
                    max(0, GREEN[2] - i//3)
                )
                color_bottom = (
                    max(0, GREEN[0] - (self.width-i)//3),
                    max(0, GREEN[1] - (self.width-i)//3),
                    max(0, GREEN[2] - (self.width-i)//3)
                )
                pygame.draw.rect(screen, color_top, (self.x + i, 0, 1, self.top_height))
                pygame.draw.rect(screen, color_bottom, (self.x + i, self.y, 1, HEIGHT - self.y))
            
            pygame.draw.rect(screen, (0, 100, 0), (self.x - 3, 0, 3, self.top_height))
            pygame.draw.rect(screen, (0, 100, 0), (self.x - 3, self.y, 3, HEIGHT - self.y))
            
            pygame.draw.rect(screen, (100, 255, 100), (self.x, 0, self.width, 3))
            pygame.draw.rect(screen, (100, 255, 100), (self.x, self.y - 3, self.width, 3))
        else:
            pygame.draw.rect(screen, GREEN, (self.x, 0, self.width, self.top_height))
            pygame.draw.rect(screen, GREEN, (self.x, self.y, self.width, HEIGHT - self.y))
            
    def collide(self, bird: Bird) -> bool:
        bird_rect = pygame.Rect(bird.x - bird.radius, bird.y - bird.radius,
                               bird.radius*2, bird.radius*2)
        top_pipe = pygame.Rect(self.x, 0, self.width, self.top_height)
        bottom_pipe = pygame.Rect(self.x, self.y, self.width, HEIGHT - self.y)
        return bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe)

class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_input_hidden = np.random.randn(hidden_nodes, input_nodes) * np.sqrt(2./input_nodes)
        self.weights_hidden_output = np.random.randn(output_nodes, hidden_nodes) * np.sqrt(2./hidden_nodes)
        self.last_activation = None
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.array(inputs, ndmin=2).T
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = np.maximum(0.01 * hidden_inputs, hidden_inputs)
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = 1 / (1 + np.exp(-final_inputs))
        self.last_activation = {
            'input': inputs.flatten(),
            'hidden': hidden_outputs.flatten(),
            'output': final_outputs.flatten(),
            'weights': {
                'input_hidden': self.weights_input_hidden,
                'hidden_output': self.weights_hidden_output
            }
        }
        return final_outputs
        
    def mutate(self) -> None:
        def mutate_array(arr: np.ndarray) -> np.ndarray:
            mask = np.random.rand(*arr.shape) < MUTATION_RATE
            mutation = np.random.randn(*arr.shape) * MUTATION_STRENGTH
            arr[mask] += mutation[mask]
            return np.clip(arr, -3, 3)
            
        self.weights_input_hidden = mutate_array(self.weights_input_hidden)
        self.weights_hidden_output = mutate_array(self.weights_hidden_output)
        
    def crossover(self, other: 'NeuralNetwork') -> 'NeuralNetwork':
        child = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        mask = np.random.rand(*self.weights_input_hidden.shape) < 0.5
        child.weights_input_hidden = np.where(mask, self.weights_input_hidden, other.weights_input_hidden)
        mask = np.random.rand(*self.weights_hidden_output.shape) < 0.5
        child.weights_hidden_output = np.where(mask, self.weights_hidden_output, other.weights_hidden_output)
        return child
        
    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights_input_hidden': self.weights_input_hidden,
                'weights_hidden_output': self.weights_hidden_output,
                'input_nodes': self.input_nodes,
                'hidden_nodes': self.hidden_nodes,
                'output_nodes': self.output_nodes
            }, f)
            
    @classmethod
    def load(cls, filename: str) -> Optional['NeuralNetwork']:
        if not os.path.exists(filename):
            return None
            
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        nn = cls(data['input_nodes'], data['hidden_nodes'], data['output_nodes'])
        nn.weights_input_hidden = data['weights_input_hidden']
        nn.weights_hidden_output = data['weights_hidden_output']
        return nn

class Population:
    def __init__(self):
        self.birds = []
        self.generation = 1
        self.best_score = 0
        self.avg_score = 0
        self.avg_score_history = []
        self.champion = None
        self.champion_brain = self.load_champion()
        self.initialize_population()
        
    def load_champion(self) -> Optional[NeuralNetwork]:
        return NeuralNetwork.load(SAVE_FILE)
        
    def initialize_population(self) -> None:
        self.birds = []
        
        if self.champion_brain:
            for _ in range(ELITE_SIZE):
                elite_brain = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
                elite_brain.weights_input_hidden = self.champion_brain.weights_input_hidden.copy()
                elite_brain.weights_hidden_output = self.champion_brain.weights_hidden_output.copy()
                self.birds.append(Bird(100, HEIGHT//2, elite_brain))
        
        for _ in range(POPULATION_SIZE - len(self.birds)):
            if self.champion_brain:
                new_brain = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
                new_brain.weights_input_hidden = self.champion_brain.weights_input_hidden.copy()
                new_brain.weights_hidden_output = self.champion_brain.weights_hidden_output.copy()
                new_brain.mutate()
            else:
                new_brain = None
                
            self.birds.append(Bird(100, HEIGHT//2, new_brain))
            
    def update(self, pipes: List[Pipe]) -> None:
        for bird in self.birds:
            if bird.alive:
                bird.update(pipes, BASE_GRAVITY)
                
                if any(pipe.collide(bird) for pipe in pipes):
                    bird.alive = False
                    
                if pipes and not pipes[0].passed and pipes[0].x + pipes[0].width < bird.x - bird.radius:
                    pipes[0].passed = True
                    bird.score += 1
                    
    def natural_selection(self) -> None:
        sorted_birds = sorted(self.birds, key=lambda b: b.score, reverse=True)
        scores = [b.score for b in self.birds]
        self.avg_score = sum(scores) / len(scores)
        self.avg_score_history.append(self.avg_score)
        
        current_best = max(scores)
        if current_best > self.best_score:
            self.best_score = current_best
            self.champion = sorted_birds[0]
            self.champion.brain.save(SAVE_FILE)
            
        new_birds = []
        for i in range(ELITE_SIZE):
            elite_brain = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
            elite_brain.weights_input_hidden = sorted_birds[i].brain.weights_input_hidden.copy()
            elite_brain.weights_hidden_output = sorted_birds[i].brain.weights_hidden_output.copy()
            new_birds.append(Bird(100, HEIGHT//2, elite_brain))
        
        tournament_size = 5
        for _ in range(POPULATION_SIZE - ELITE_SIZE):
            candidates = random.sample(sorted_birds[:100], tournament_size)
            parent1 = max(candidates, key=lambda b: b.score)
            candidates = random.sample(sorted_birds[:100], tournament_size)
            parent2 = max(candidates, key=lambda b: b.score)
            child_brain = parent1.brain.crossover(parent2.brain)
            child_brain.mutate()
            new_birds.append(Bird(100, HEIGHT//2, child_brain))
            
        self.birds = new_birds
        self.generation += 1

def draw_neural_network(screen: pygame.Surface, net: NeuralNetwork, pos: Tuple[int, int], size: Tuple[int, int]) -> None:
    if not net.last_activation:
        return
        
    node_radius = 6
    layer_spacing = size[0] // 3
    vertical_padding = 20
    
    for layer_idx in range(2):
        start_layer = net.last_activation['input'] if layer_idx == 0 else net.last_activation['hidden']
        end_layer = net.last_activation['hidden'] if layer_idx == 0 else net.last_activation['output']
        weights = net.last_activation['weights']['input_hidden'] if layer_idx == 0 else net.last_activation['weights']['hidden_output']
        
        start_x = pos[0] + layer_idx * layer_spacing
        end_x = pos[0] + (layer_idx + 1) * layer_spacing
        
        for i in range(len(start_layer)):
            start_y = pos[1] + vertical_padding + i * (size[1] - 2*vertical_padding) / max(1, len(start_layer)-1)
            
            for j in range(len(end_layer)):
                end_y = pos[1] + vertical_padding + j * (size[1] - 2*vertical_padding) / max(1, len(end_layer)-1)
                
                weight = weights[j,i] if layer_idx == 0 else weights[j,i]
                color = NEURAL_COLORS[layer_idx]
                alpha = min(255, int(255 * abs(weight) / 3))
                thickness = max(1, min(3, int(abs(weight))))
                
                if pygame.version.vernum[0] >= 2:
                    s = pygame.Surface((abs(end_x - start_x), abs(end_y - start_y)), pygame.SRCALPHA)
                    pygame.draw.line(s, (*color, alpha), 
                                    (0, 0 if end_y > start_y else s.get_height()),
                                    (s.get_width(), s.get_height() if end_y > start_y else 0),
                                    thickness)
                    screen.blit(s, (min(start_x, end_x), min(start_y, end_y)))
    
    for layer_idx in range(3):
        layer = None
        if layer_idx == 0:
            layer = net.last_activation['input']
        elif layer_idx == 1:
            layer = net.last_activation['hidden']
        else:
            layer = net.last_activation['output']
            
        x = pos[0] + layer_idx * layer_spacing
        
        for i, activation in enumerate(layer):
            y = pos[1] + vertical_padding + i * (size[1] - 2*vertical_padding) / max(1, len(layer)-1)
            intensity = min(255, max(0, int(255 * abs(activation))))
            if activation >= 0:
                color = (intensity, intensity//2, intensity//3)
            else:
                color = (intensity//3, intensity//2, intensity)
            pygame.draw.circle(screen, color, (int(x), int(y)), node_radius)
            pygame.draw.circle(screen, BLACK, (int(x), int(y)), node_radius, 1)

def draw_graph(surface: pygame.Surface, history: List[float], pos: Tuple[int, int], 
              size: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    if len(history) < 2:
        return
        
    max_val = max(history) if max(history) > 0 else 1
    points = []
    
    for i, val in enumerate(history):
        x = pos[0] + int((i / (len(history)-1)) * size[0])
        y = pos[1] + size[1] - int((val / max_val) * size[1])
        points.append((x, y))
        
    pygame.draw.lines(surface, color, False, points, 2)
    
    font = pygame.font.SysFont('Arial', 12)
    label = font.render(f"Max: {max_val:.1f}", True, WHITE)
    surface.blit(label, (pos[0], pos[1] - 15))

def ai_main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Flappy Bird AI - Population: {POPULATION_SIZE}")
    clock = pygame.time.Clock()
    
    small_font = pygame.font.SysFont('Arial', 14)
    medium_font = pygame.font.SysFont('Arial', 18)
    large_font = pygame.font.SysFont('Arial', 24, bold=True)
    
    population = Population()
    pipes = []
    last_pipe = pygame.time.get_ticks()
    running = True
    paused = False
    fast_forward = False
    show_network = False
    human_mode = False
    human_bird = Bird(100, HEIGHT//2)
    human_score = 0
    selected_bird = None
    difficulty = DynamicDifficulty()
    
    controls_text = [
        small_font.render("Space: Pause/Resume", True, WHITE),
        small_font.render("F: Fast Forward", True, WHITE),
        small_font.render("N: Neural Network", True, WHITE),
        small_font.render("R: Reset", True, WHITE),
        small_font.render("X: Human Mode", True, WHITE),
        small_font.render("Click: Inspect Bird", True, WHITE)
    ]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    fast_forward = not fast_forward
                elif event.key == pygame.K_n:
                    show_network = not show_network
                elif event.key == pygame.K_r:
                    population = Population()
                    pipes = []
                    last_pipe = pygame.time.get_ticks()
                    human_mode = False
                    selected_bird = None
                    difficulty = DynamicDifficulty()
                elif event.key == pygame.K_x:
                    human_mode = not human_mode
                    if human_mode:
                        human_bird = Bird(100, HEIGHT//2)
                        human_score = 0
                elif human_mode and event.key == pygame.K_SPACE:
                    human_bird.jump(difficulty.get_gravity())
            elif event.type == pygame.MOUSEBUTTONDOWN and not human_mode:
                mouse_pos = pygame.mouse.get_pos()
                for bird in population.birds:
                    if bird.alive and ((bird.x - mouse_pos[0])**2 + (bird.y - mouse_pos[1])**2) <= bird.radius**2:
                        selected_bird = bird
                        break
        
        if not paused:
            current_time = pygame.time.get_ticks()
            current_score = human_score if human_mode else max(b.score for b in population.birds) if population.birds else 0
            
            difficulty.update(current_score)
            
            if current_time - last_pipe > difficulty.get_pipe_frequency() and (not fast_forward or len(pipes) < 3):
                pipes.append(Pipe(difficulty.get_pipe_gap(), difficulty.get_pipe_speed()))
                last_pipe = current_time
                
            for pipe in pipes[:]:
                pipe.update()
                if pipe.x < -pipe.width:
                    pipes.remove(pipe)
                    
            if human_mode:
                human_bird.update(pipes, difficulty.get_gravity(), human_mode=True)
                
                if any(pipe.collide(human_bird) for pipe in pipes):
                    human_mode = False
                    
                if pipes and not pipes[0].passed and pipes[0].x + pipes[0].width < human_bird.x - human_bird.radius:
                    pipes[0].passed = True
                    human_score += 1
            else:
                population.update(pipes)
                
                if all(not bird.alive for bird in population.birds):
                    population.natural_selection()
                    pipes = []
                    last_pipe = pygame.time.get_ticks()
                    selected_bird = None
                    difficulty = DynamicDifficulty()
                
            if fast_forward and not human_mode:
                clock.tick(120)
            else:
                clock.tick(60)
        
        screen.fill(BACKGROUND)
        
        for pipe in pipes:
            pipe.draw(screen, human_mode)
            
        if human_mode:
            human_bird.draw(screen, human_mode=True)
            
            score_text = large_font.render(f"Score: {human_score}", True, WHITE)
            screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 20))
            
            diff_text = medium_font.render(f"Difficulty: {difficulty.difficulty_level:.1f}x", True, WHITE)
            screen.blit(diff_text, (WIDTH//2 - diff_text.get_width()//2, 60))
        else:
            alive_birds = [b for b in population.birds if b.alive]
            max_score = max(b.score for b in population.birds) if population.birds else 1
            
            for bird in population.birds:
                if bird.alive:
                    alpha = 30 + int(225 * (bird.score / max_score)) if max_score > 0 else 30
                    bird.draw(screen, alpha)
                    
                    if bird == selected_bird:
                        pygame.draw.circle(screen, RED, (int(bird.x), int(bird.y)), bird.radius + 4, 2)
            
            if population.champion and any(b.genome_id == population.champion.genome_id for b in population.birds):
                for bird in population.birds:
                    if bird.genome_id == population.champion.genome_id:
                        pygame.draw.circle(screen, GOLD, (int(bird.x), int(bird.y)), bird.radius + 6, 3)
        
        if not human_mode:
            stats = [
                f"Generation: {population.generation}",
                f"Alive: {len([b for b in population.birds if b.alive])}/{POPULATION_SIZE}",
                f"Best Score: {population.best_score}",
                f"Current Best: {max(b.score for b in population.birds) if population.birds else 0}",
                f"Difficulty: {difficulty.difficulty_level:.1f}x"
            ]
            
            for i, text in enumerate(stats):
                screen.blit(medium_font.render(text, True, WHITE), (10, 10 + i * 25))
            
        for i, text_surface in enumerate(controls_text):
            screen.blit(text_surface, (WIDTH - text_surface.get_width() - 10, 10 + i * 20))
        
        if not human_mode and len(population.avg_score_history) > 1:
            graph_width = 200
            graph_height = 80
            graph_pos = (WIDTH - graph_width - 20, HEIGHT - graph_height - 20)
            
            pygame.draw.rect(screen, (20, 20, 40), (*graph_pos, graph_width, graph_height))
            pygame.draw.rect(screen, (50, 50, 80), (*graph_pos, graph_width, graph_height), 1)
            
            max_val = max(population.avg_score_history)
            points = []
            for i, val in enumerate(population.avg_score_history):
                x = graph_pos[0] + int((i / (len(population.avg_score_history)-1)) * graph_width)
                y = graph_pos[1] + graph_height - int((val / max_val) * graph_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, BLUE, False, points, 2)
            
            screen.blit(small_font.render("Avg Score Trend", True, WHITE), 
                       (graph_pos[0], graph_pos[1] - 15))
            screen.blit(small_font.render(f"Max: {max_val:.1f}", True, WHITE), 
                       (graph_pos[0] + graph_width - 50, graph_pos[1] - 15))
        
        if show_network and selected_bird and selected_bird.brain.last_activation and not human_mode:
            net_pos = (WIDTH - 350, 150)
            net_size = (300, 200)
            
            pygame.draw.rect(screen, (20, 20, 40), (*net_pos, net_size[0], net_size[1]))
            pygame.draw.rect(screen, (50, 50, 80), (*net_pos, net_size[0], net_size[1]), 1)
            
            draw_neural_network(screen, selected_bird.brain, net_pos, net_size)
            
            labels = [
                "Bird X Position",
                "Bird Y Position",
                "Y Velocity",
                "Top Height",
                "Bottom Height",
                "Pipe X Position",
                "Gap Distance"
            ]
            
            for i, label in enumerate(labels):
                if i < len(selected_bird.brain.last_activation['input']):
                    y = net_pos[1] + 20 + i * (net_size[1] - 40) / max(1, len(labels)-1)
                    screen.blit(small_font.render(label, True, WHITE), 
                              (net_pos[0] - 120, y - 8))
        
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    ai_main()