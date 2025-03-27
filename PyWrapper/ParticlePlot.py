#!/bin/python3

import pygame
import matplotlib.pyplot as plt
import numpy as np
import sys
import yaml
import pathlib
import pint
import scipy.stats
import sys
import math

from fitting_util import *

sys.path.append("..")
from UnitSystem import *
from Constants import Constants


if __name__ == '__main__':
    default_dump_dir = "out/single_spec_2d_units/26_03_08_55_00"
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else default_dump_dir
    dump_dir_p = pathlib.Path(dump_dir)

    set_conversion_mode(ConversionMode.DIM)
    Constants.prepare_constants()

    cfg = {}
    with open(dump_dir_p / "config.yml", "r") as cfg_yml:
        cfg = yaml.safe_load(cfg_yml)
        species = cfg["species"]
        spec_dict = {}
        for s in species:
            spec_dict[s["name"]] = s
        cfg["species"] = spec_dict

    print(f"Loading simulation dump and determining parameters")
    m = QParse(cfg["sim"]["m"])
    sigma = QParse(cfg["sim"]["sigma"])
    tau_sim = QParse(cfg["sim"]["tau_sim"])
    characteristics(sigma, m, tau_sim)

    particle_cnt: int = 0
    for s in cfg["species"]:
        particle_cnt += cfg["species"][s]["quantity"]

    sim_size = mag(non_dim(QParse(cfg["setup"]["size"])))
    max_iterations = math.ceil(mag(QParse(cfg["setup"]["max_sim_time"]) / QParse(cfg["setup"]["dump_interval"]))) + 2

    print(f"Loading particle data")
    i = 0
    iteration = 0
    pos_scale = dim(1, "bohr").magnitude
    dims = 3#cfg["setup"]["dimensions"]
    print("Loading particle velocities")
    ppos = np.fromfile(dump_dir_p / "particle_positions.bin", dtype=np.float64)
    ppos = ppos.reshape(-1, particle_cnt, dims) * pos_scale

    # Initialize PyGame
    pygame.init()

    # Simulation parameters
    WIDTH, HEIGHT = sim_size * pos_scale, sim_size * pos_scale # Window size
    BACKGROUND_COLOR = (0, 0, 0)  # Black background
    PARTICLE_COLOR = (255, 0, 0)  # Red particles

    # Scale factor to fit the simulation domain into the screen
    sim_width, sim_height = sim_size, sim_size

    # Particle radius in pixels
    particle_radius = QParse(cfg["species"]["A"]["radius"]).magnitude

    # Create window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Particle Simulation Viewer")

    # Iteration control
    current_iteration = 0

    # Main loop
    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)  # Clear screen

        # Get particle data for the current iteration
        for pos in ppos[current_iteration]:
            x, y = int(pos[0]), int(pos[1])
            pygame.draw.circle(screen, PARTICLE_COLOR, (x, HEIGHT - y), particle_radius)

        pygame.display.flip()  # Update display

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            current_iteration = min(current_iteration + 1, max_iterations - 1)
        elif keys[pygame.K_LEFT]:
            current_iteration = max(current_iteration - 1, 0)

    pygame.quit()
    sys.exit()

