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


DRAW_VEL_LINES : bool = True


if __name__ == '__main__':
    default_dump_dir = "out/single_spec_2d_units/27_03_14_22_04"
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
    screen_scale = 1000.0 / sim_size
    pos_scale = dim(1, "bohr").magnitude
    vel_scale = dim(1, "bohr/s").magnitude
    dims = cfg["setup"]["dimensions"]

    print("Loading particle velocities")
    ppos = np.fromfile(dump_dir_p / "particle_positions.bin", dtype=np.float64)
    ppos = ppos.reshape(-1, particle_cnt, dims) * pos_scale * screen_scale

    pvels = np.fromfile(dump_dir_p / "particle_velocities.bin", dtype=np.float64)
    pvels = pvels.reshape(-1, particle_cnt, dims)
    pvels *= vel_scale * screen_scale

    pids = np.fromfile(dump_dir_p / "dbg.bin", dtype=np.uint16)
    pids = pids.reshape(-1, particle_cnt)

    # Initialize PyGame
    pygame.init()

    WIDTH, HEIGHT = sim_size * pos_scale * screen_scale, sim_size * pos_scale * screen_scale # Window size
    BACKGROUND_COLOR = (0, 0, 0)  # Black background
    PARTICLE_COLOR = (255, 0, 0)  # Red particles

    particle_radius = QParse(cfg["species"]["A"]["radius"]).magnitude * screen_scale

    # Create window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Particle Simulation Viewer")

    # Iteration control
    current_iteration = 0

    colors = [
        (255,   0,   0),
        (0  , 255,   0),
        (0  ,   0, 255),
        (255,   0, 127),
        (255, 127,   0),
        (255, 255, 127), # Here
        (100, 100, 100), # Here
        (  0, 127, 255),
        (127,   0, 255),
        (255, 255, 255)
    ]

    t = tau_sim.to('s').magnitude

    # Main loop
    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)  # Clear screen

        # Get particle data for the current iteration
        for i in range(len(ppos[current_iteration])):
            pos = ppos[current_iteration][i]
#            id = pids[current_iteration][i]
            x, y = int(pos[0]), int(pos[1])

            pygame.draw.circle(screen, PARTICLE_COLOR, (x, HEIGHT - y), particle_radius)

        # Draw velocity vectors
        if DRAW_VEL_LINES:
            for i in range(len(ppos[current_iteration])):
                pos = ppos[current_iteration][i]
                vel = pvels[current_iteration][i]
                x, y = int(pos[0]), int(pos[1])

                pygame.draw.line(screen,  PARTICLE_COLOR, (x, HEIGHT-y), (x+t*vel[0], HEIGHT-(y+t*vel[1])))

        pygame.display.flip()  # Update display

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            current_iteration = min(current_iteration + 1, max_iterations - 1)
#            for i, id in enumerate(pids[current_iteration]):
#                if id == 5:
#                    pos5 = np.array(ppos[current_iteration][id]) / pos_scale
#                    vel5 = np.array(pvels[current_iteration][id]) / vel_scale
#                elif id == 6:
#                    pos6 = np.array(ppos[current_iteration][6]) / pos_scale
#                    vel6 = np.array(pvels[current_iteration][6]) / vel_scale
#                    break
#            dr = pos5 - pos6
#            dv = vel5 - vel6
#            b = np.dot(dr, dv)
#            print(pos5, pos6, vel5, vel6, b)
        elif keys[pygame.K_LEFT]:
            current_iteration = max(current_iteration - 1, 0)

#        pygame.time.wait(100)

    pygame.quit()
    sys.exit()

