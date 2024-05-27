import numpy as np 
import random
import math 
from numba import njit

def painter_play(rules, room):
    # Returns score, xpos, ypos, env
    M, N = room.shape

    # Calculate number of steps allowed
    t = int(M * N - room.sum())

    # Add walls to the room
    env = np.ones((M + 2, N + 2))
    env[1:M+1, 1:N+1] = room

    xpos = np.zeros(t + 1, dtype=int)
    ypos = np.zeros(t + 1, dtype=int)

    # Random initial location
    while True:
        xpos[0] = random.randint(1, M)
        ypos[0] = random.randint(1, N)
        if env[xpos[0], ypos[0]] == 0:
            break

    # Random initial orientation (up=0, left=-1, right=+1, down=-2)
    direction = random.choice([-2, -1, 0, 1])

    # Initial score
    score = 0

    for i in range(t):
        # Determine dx, dy for the current direction
        if direction == 0:      # Up
            dx, dy = -1, 0
        elif direction == -1:   # Left
            dx, dy = 0, -1
        elif direction == 1:    # Right
            dx, dy = 0, 1
        elif direction == -2:   # Down
            dx, dy = 1, 0

        # Determine dxr, dyr for the right square
        if direction == 0:      # Up
            dxr, dyr = 0, 1
        elif direction == -1:   # Left
            dxr, dyr = 1, 0
        elif direction == 1:    # Right
            dxr, dyr = -1, 0
        elif direction == -2:   # Down
            dxr, dyr = 0, -1

        # Evaluate surroundings (forward, left, right)
        local = [
            env[xpos[i] + dx, ypos[i] + dy],
            env[xpos[i] - dyr, ypos[i] + dxr],
            env[xpos[i] + dyr, ypos[i] - dxr]
        ]

        localnum = int(2 * np.dot([9, 3, 1], local))
        if env[xpos[i], ypos[i]] == 2:
            localnum += 1

        # Determine direction change
        if rules[localnum] == 3:
            dirchange = random.choice([1, 2])
        else:
            dirchange = rules[localnum]

        if dirchange == 1:
            direction -= 1
            if direction == -3:
                direction = 1
        elif dirchange == 2:
            direction += 1
            if direction == 2:
                direction = -2

        # Update dx, dy for the new direction
        if direction == 0:      # Up
            dx, dy = -1, 0
        elif direction == -1:   # Left
            dx, dy = 0, -1
        elif direction == 1:    # Right
            dx, dy = 0, 1
        elif direction == -2:   # Down
            dx, dy = 1, 0

        # Paint the current square
        if env[xpos[i], ypos[i]] == 0:
            env[xpos[i], ypos[i]] = 2
            score += 1

        # Move forward if possible, else stay put
        if env[xpos[i] + dx, ypos[i] + dy] == 1:
            xpos[i + 1] = xpos[i]
            ypos[i + 1] = ypos[i]
        else:
            xpos[i + 1] = xpos[i] + dx
            ypos[i + 1] = ypos[i] + dy

    # Normalize score by time
    score /= t

    return score, xpos, ypos, env

# Test the function
# test_room = np.zeros((8, 7))
# test_rules = np.ones(54, dtype=int)
# for i in range(len(test_rules)):
#     test_rules[i] = 3

# print(painter_play(test_rules, test_room))
