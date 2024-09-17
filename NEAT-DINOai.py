import pygame
import neat
import os
import sys
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import networkx as nx
import colorsys

# New variables for easy modification
MAX_GENERATIONS = 1000
POPULATION_SIZE = 100
GAMESPEED = 600


# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 400
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NOT a dino game.")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Player properties
PLAYER_WIDTH, PLAYER_HEIGHT = 10, 50
GRAVITY = 1

# Obstacle properties
obstacle_width = 50
obstacle_height = 50
obstacle_vel_x = -5

# High obstacle position adjustment
HIGH_OBSTACLE_Y = HEIGHT - PLAYER_HEIGHT - obstacle_height - 40

# Clock
clock = pygame.time.Clock()

# Generation counter
GEN = 0

# Fitness tracking
fitness_history = []

# Create a figure for the graph
fig, ax = plt.subplots(figsize=(4, 3))
canvas = agg.FigureCanvasAgg(fig)

def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in RGB_tuples]

def update_fitness_graph():
    ax.clear()
    if fitness_history:
        ax.plot(range(1, len(fitness_history) + 1), fitness_history)
        ax.set_xlim(0, max(len(fitness_history), 10))
        ax.set_ylim(0, max(max(fitness_history) * 1.1, 1))
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Fitness')
    ax.set_title('Average Fitness per Generation')
    fig.tight_layout()
    canvas.draw()
    
    graph_str = canvas.tostring_rgb()
    graph_surf = pygame.image.fromstring(graph_str, canvas.get_width_height(), "RGB")
    return graph_surf

def get_obstacle_rect(obstacle_x, obstacle_type):
    if obstacle_type == "low":
        return pygame.Rect(obstacle_x, HEIGHT - obstacle_height, obstacle_width, obstacle_height)
    else:
        return pygame.Rect(
            obstacle_x,
            HIGH_OBSTACLE_Y - 100,
            obstacle_width,
            obstacle_height + 150
        )

def eval_genomes(genomes, config):
    global GEN
    GEN += 1

    nets = []
    ge = []
    players = []

    # Generate distinct colors for each player
    colors = generate_distinct_colors(len(genomes))

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        players.append({
            "x": 100,
            "y": HEIGHT - PLAYER_HEIGHT,
            "vel_y": 0,
            "ducking": False,
            "obstacle_index": 0,
            "color": colors[i]  # Assign a unique color to each player
        })
        ge.append(genome)

    obstacles = []
    obstacle_timer = 0
    obstacle_interval = random.randint(60, 120)

    # Create initial graph surface
    graph_surf = update_fitness_graph()
    last_graph_update = 0

    # Create font for displaying population count
    font = pygame.font.Font(None, 36)

    running = True
    frame_count = 0
    max_fitness = 0  # Track the maximum fitness
    while running and len(players) > 0:
        frame_count += 1
        clock.tick(GAMESPEED)
        SCREEN.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        obstacle_timer += 1
        if obstacle_timer >= obstacle_interval:
            obstacle_timer = 0
            obstacle_interval = random.randint(60, 120)
            obstacle_type = random.choice(["low", "high"])
            obstacles.append({
                "x": WIDTH,
                "type": obstacle_type
            })

        for obstacle in list(obstacles):
            obstacle['x'] += obstacle_vel_x

            if obstacle['x'] < -obstacle_width:
                obstacles.remove(obstacle)
                continue

            obstacle_rect = get_obstacle_rect(obstacle['x'], obstacle['type'])
            color = BLACK if obstacle['type'] == "low" else RED
            pygame.draw.rect(SCREEN, color, obstacle_rect)

        for x in reversed(range(len(players))):
            player = players[x]
            ge[x].fitness += 0.1

            if not player['ducking']:
                player['vel_y'] += GRAVITY
                player['y'] += player['vel_y']
                if player['y'] >= HEIGHT - PLAYER_HEIGHT:
                    player['y'] = HEIGHT - PLAYER_HEIGHT
                    player['vel_y'] = 0

            next_obstacle = next((obstacle for obstacle in obstacles if obstacle['x'] + obstacle_width > player['x']), None)

            if next_obstacle is None:
                continue

            distance = next_obstacle['x'] - player['x']
            obstacle_type_input = 1 if next_obstacle['type'] == "low" else 0

            output = nets[x].activate((
                player['y'],
                int(player['ducking']),
                distance,
                obstacle_type_input
            ))

            action = output.index(max(output))
            player['ducking'] = False

            if action == 1 and player['y'] == HEIGHT - PLAYER_HEIGHT and not player['ducking']:
                player['vel_y'] = -20
            elif action == 2 and player['y'] == HEIGHT - PLAYER_HEIGHT:
                player['ducking'] = True

            player_rect = pygame.Rect(
                player['x'],
                player['y'] + PLAYER_HEIGHT // 2 if player['ducking'] else player['y'],
                PLAYER_WIDTH,
                PLAYER_HEIGHT // 2 if player['ducking'] else PLAYER_HEIGHT
            )

            obstacle_rect = get_obstacle_rect(next_obstacle['x'], next_obstacle['type'])

            if player_rect.colliderect(obstacle_rect):
                ge[x].fitness -= 1
                nets.pop(x)
                ge.pop(x)
                players.pop(x)
                continue

            # Draw the player with its unique color
            pygame.draw.rect(SCREEN, player['color'], player_rect)

            # Update max fitness
            max_fitness = max(max_fitness, ge[x].fitness)

        # Draw population count
        population_text = font.render(f"Population: {len(players)}", True, BLACK)
        SCREEN.blit(population_text, (10, 10))

        # Draw max fitness
        max_fitness_text = font.render(f"Max Fitness: {max_fitness:.1f}", True, BLACK)
        SCREEN.blit(max_fitness_text, (10, 50))

        # Calculate and store average fitness for this frame
        if ge:
            avg_fitness = sum(genome.fitness for genome in ge) / len(ge)
            if len(fitness_history) < GEN:
                fitness_history.append(avg_fitness)
            else:
                fitness_history[GEN-1] = avg_fitness

        # Update graph every 60 frames (approximately once per second)
        if frame_count - last_graph_update >= 60:
            new_graph_surf = update_fitness_graph()
            # Only update if the new graph is different
            if new_graph_surf.get_buffer().raw != graph_surf.get_buffer().raw:
                graph_surf = new_graph_surf
            last_graph_update = frame_count

        # Always blit the graph surface
        SCREEN.blit(graph_surf, (WIDTH - graph_surf.get_width(), 0))

        pygame.display.update()

        # Check if max fitness threshold is reached
        if max_fitness >= 1000:
            print(f"Max fitness of {max_fitness} reached! Stopping simulation.")
            running = False

    # After each generation ends, ensure the final fitness is recorded and display the genome details
    if ge:

        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        avg_fitness = sum(genome.fitness for genome in ge) / len(ge)
        if len(fitness_history) < GEN:
            fitness_history.append(avg_fitness)
        else:
            fitness_history[GEN-1] = avg_fitness
        print(f"Generation {GEN}: Final Average Fitness = {avg_fitness}")

        # Display nodes and connections for each genome in this generation
        for i, genome in enumerate(ge):
            print(f"\nGenome {i+1}:")
            print("Nodes:")
            for node_key, node in genome.nodes.items():
                print(f"        {node_key} {node}")
            print("Connections:")
            for conn_key, conn in genome.connections.items():
                print(f"        {conn}")

    return max_fitness >= 1000  # Return True if max fitness is reached

def visualize_genome_in_window(genome, config):
    """Visualize a NEAT genome in a layered neural network structure."""
    # Create a directed graph
    G = nx.DiGraph()

    # Categorize nodes into input, output, and hidden layers
    input_nodes = set(config.genome_config.input_keys)
    output_nodes = set(config.genome_config.output_keys)
    hidden_nodes = set(genome.nodes.keys()) - input_nodes - output_nodes

    # Add all nodes to the graph
    for node_key in genome.nodes:
        G.add_node(node_key)

    # Add connections (edges) between nodes
    for conn_key, conn in genome.connections.items():
        # Ensure that both source and destination nodes exist in the genome before adding the connection
        if conn.enabled and conn_key[0] in genome.nodes and conn_key[1] in genome.nodes:
            G.add_edge(conn_key[0], conn_key[1], weight=conn.weight)

    # Define positions for nodes in layers
    pos = {}
    layer_spacing = 1.0  # Horizontal spacing between layers
    node_spacing = 0.5  # Vertical spacing between nodes within a layer

    # Set positions for input nodes (left side)
    layer_x = 0.0
    for i, node in enumerate(input_nodes):
        pos[node] = (layer_x, i * node_spacing)

    # Set positions for hidden nodes (middle layers)
    layer_x += layer_spacing
    hidden_layers = []
    for node in hidden_nodes:
        if not hidden_layers or all(any(G.has_edge(prev_node, node) for prev_node in prev_layer) for prev_layer in hidden_layers):
            hidden_layers.append([node])
        else:
            for layer in hidden_layers:
                if any(G.has_edge(prev_node, node) for prev_node in layer):
                    layer.append(node)
                    break

    for i, layer in enumerate(hidden_layers):
        for j, node in enumerate(layer):
            pos[node] = (layer_x, j * node_spacing)
        layer_x += layer_spacing

    # Set positions for output nodes (right side)
    for i, node in enumerate(output_nodes):
        pos[node] = (layer_x, i * node_spacing)

    # Draw the nodes and edges
    fig, ax = plt.subplots(figsize=(12, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=800, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

    plt.axis('off')
    plt.show()

def run(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Use the POPULATION_SIZE variable here
    config.pop_size = POPULATION_SIZE

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = None
    for generation in range(MAX_GENERATIONS):
        best_genome = p.run(eval_genomes, 1) 
        
        if best_genome.fitness >= 1000:
            winner = best_genome
            print(f"Fitness threshold reached in generation {generation + 1}!")
            break
    
    if winner is None:
        winner = p.best_genome
        print(f"Fitness threshold not reached within {MAX_GENERATIONS} generations.")

    print('\nBest genome:\n{!s}'.format(winner))

    # Visualize the winner
    visualize_genome_in_window(winner, config)

if __name__ == "__main__":
    config_data = f"""
[NEAT]
fitness_criterion      = max
fitness_threshold      = 2000
pop_size               = {POPULATION_SIZE}
reset_on_extinction    = False

[DefaultGenome]
# Node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# Node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# Node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# Node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# Genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# Connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# Connection mutation options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

# Connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

# Node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# Network parameters
feed_forward            = True
initial_connection      = full
num_hidden              = 0
num_inputs              = 4
num_outputs             = 3

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func    = max
max_stagnation          = 20
species_elitism         = 2

[DefaultReproduction]
elitism                 = 2
survival_threshold      = 0.2
"""

    config_path = "temp_config_file.txt"
    with open(config_path, "w") as f:
        f.write(config_data)

    run(config_path)
    os.remove(config_path)