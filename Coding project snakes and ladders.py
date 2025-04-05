#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
"""
Created on Fri Nov 15 23:30:11 2024
@author: abdullahrizwan
Abdullah Rizwan
ARX218
2533718
"""

# Define the board and positions of the snakes and ladders using dictionaries

ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
snakes = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
pynumber_of_simulations = 100000


# Function to run simulations
def Simulation(snake_dict, ladder_dict):
    # Define lists to count number of rolls for each iteration of the simulation
    rolls = []
    square_visit_counter = np.zeros(101, dtype=int)  # List to count most visited squares over all iterations

    # Simulate snakes and ladders

    for i in range(number_of_simulations):
        num_rolls = 0
        position = 0
        while (position < 100):
            dice = np.random.randint(1, 7)
            num_rolls += 1
            position += dice
            if position < 100:
                square_visit_counter[position] += 1
            if position > 100:
                position = 100
                square_visit_counter[position] += 1
    # Change position if landing on a snake or ladder and add counter for visited square
            if position in ladder_dict:
                position = ladder_dict[position]
                square_visit_counter[position] += 1

            if position in snake_dict:
                position = snake_dict[position]
                square_visit_counter[position] += 1

        rolls.append(num_rolls)

    return rolls, square_visit_counter


'''
something i made to check if the positions corresponding to ladder heads
have been visited more than normal positions

for i in ladders:
    x = ladders[i]
    print(square_visit_counter[x])

'''

# Using the simulation function to assign values to variables
sim_rolls, square_visit_counter = Simulation(snakes, ladders)


print(f'Total number of dice rolls in 10000 simulations:{np.sum(sim_rolls)}')


# Function to calculate the average and variance from simulations
def simulation_average(rolls):
    average = np.mean(rolls)
    variance = np.sum((rolls - average)**2)/(number_of_simulations)
    return average, variance


sim_average, sim_variance = simulation_average(sim_rolls)
sim_median = np.median(sim_rolls)
sim_sd = np.sqrt(sim_variance)
sim_min_rolls, sim_max_rolls = np.min(sim_rolls), np.max(sim_rolls)
sim_percentiles = np.percentile(sim_rolls, [10, 25, 50, 75, 90])
print(f'Simulation results for {number_of_simulations}:\n')
print(f'Average number of rolls according to simulation: {sim_average}')
print(f'Variance according to simulation results: {sim_variance}')
print(f'Median according to simulation results: {sim_median}')
print(f'Standard deviation according to simulation results: {sim_sd}')
print(f'Shortest game: {sim_min_rolls}\nLongest game:{sim_max_rolls}')
print(f'Percentiles (10th, 25th, 50th, 75th, 90th): {sim_percentiles}\n')


# Markov Chain Method
def Markov(snake_dict, ladder_dict, pdfsteps):
    P = np.zeros((101, 101))

    for matrix_row in range(100):
        for dice in range(1, 7):
            matrix_column = matrix_row+dice

            if matrix_column > 100:
                matrix_column = 100

            if matrix_column in snake_dict:
                matrix_column = snake_dict[matrix_column]
            if matrix_column in ladder_dict:
                matrix_column = ladder_dict[matrix_column]

            P[matrix_row, matrix_column] += 1/6

    P[100, 100] = 1
    Q = P[:100, :100]

    I = np.eye(100, 100)
    N = np.linalg.inv(I-Q)

    expected_steps = np.sum(N[0, :])
    nsquare = np.sum((2 * np.dot(N, N) - N)[0, :])
    variance = nsquare - expected_steps ** 2

    pdf_steps = []
    max_steps = pdfsteps

    # Initial distribution vector with starting probability at square 0
    current_dis = np.zeros(101)
    current_dis[0] = 1

    for step in range(max_steps):
        # Multiply distribution by transition matrix to find probability distribution at this step
        current_dis = np.dot(current_dis, P)

        # The probability of first reaching square 100 exactly at this step
        pdf_steps.append(current_dis[100])

        # Zero out the probability of being in the absorbing state (since we want "first" arrival)
        current_dis[100] = 0

    # Normalize the PDF to ensure consistency with the histogram density
    pdf_steps = np.array(pdf_steps)
    pdf_steps /= np.sum(pdf_steps)

    return expected_steps, variance, pdf_steps
# Output the results


markov_avg_rolls, markov_variance, markov_pdf = Markov(snakes, ladders, 150)
markov_sd = np.sqrt(markov_variance)
print('Absorbing Markov Chain Results: \n')
print(f'Expected number of steps to complete the game: {markov_avg_rolls}')
print(f'Variance in number of steps to complete the game:, {markov_variance}')
print(f'Standard deviation in number of steps: {markov_sd}\n')

# Absolute and relative error

# Absolute errors
abs_error_avg = abs(sim_average - markov_avg_rolls)
abs_error_var = abs(sim_variance - markov_variance)
abs_error_sd = abs(sim_sd - markov_sd)

# Relative errors
rel_error_avg = (abs_error_avg / markov_avg_rolls) * 100
rel_error_var = (abs_error_var / markov_variance) * 100
rel_error_sd = (abs_error_sd / markov_sd) * 100

print('Absolute errors:')
print(f'Average number of rolls: {abs_error_avg}\n Variance: {abs_error_var}\n Standard deviation: {abs_error_sd}\n')
print('Relative errors: ')
print(f'Average number of rolls: {rel_error_avg:.5f}%\n Variance: {rel_error_var:.5f}%\n Standard deviation: {rel_error_sd:.5f}%\n')

# Plotting histogram for frequency to number of rolls
plt.hist(sim_rolls, bins=50, edgecolor='black')
plt.xlabel('Number of Rolls')
plt.ylabel('Frequency')
plt.title('Histogram of number of rolls')
plt.show()

for i in range(1, 4):  # 1 to 3 standard deviations
    plt.axvline(sim_average + i * sim_sd, color='red', linestyle='--', label=f'+{i} SD')
plt.axvline(sim_average - 1 * sim_sd, color='blue', linestyle='--', label=f'-{1} SD')

# Shade ±1 standard deviation region
plt.fill_betweenx(
    y=[0, 0.03],
    x1=sim_average - sim_sd, x2=sim_average + sim_sd,
    color='green', alpha=0.2, label='±1 SD Region'
)


# Plot histogram of simulation results
plt.hist(sim_rolls, bins=50, density=True, alpha=0.6, edgecolor='black', label='Simulation Histogram')
plt.axvline(sim_average, color='black', label='Avg no. of rolls from sim data')

# Plot PDF based on Markov chain
plt.plot(range(1, len(markov_pdf) + 1), markov_pdf, color='red', linestyle='-', label='Markov Chain PDF')
plt.xlabel('Number of Rolls')
plt.ylabel('Probability of game ending after n rolls')
plt.legend()
plt.title('Probability Density Function of Rolls to Complete Game(Analytically)')
plt.figure().set_figwidth(20)
plt.show()

# Plotting a cumulative distrinution of rolls
sorted_rolls = np.sort(sim_rolls)
cdf = np.arange(len(sorted_rolls)) / len(sorted_rolls)

plt.plot(sorted_rolls, cdf, label='CDF')
plt.xlabel('Number of Rolls')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution of Rolls')
plt.axvline(sim_average, color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()

# Finding and counting outliers
q1, q3 = np.percentile(sim_rolls, [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = []
for i in sim_rolls:
    if i < lower_bound or i > upper_bound:
        outliers.append(i)
print(f'Number of Outliers: {len(outliers)}\n')


# Analysis of number of times a snake or a ladder is hit and the effect it has on average
def snake_ladder_impact(snakes, ladders):
    snake_hits, ladder_hits = 0, 0
    snake_penalty, ladder_gain = 0, 0

    for _ in range(number_of_simulations):
        position = 0
        while position < 100:
            dice = np.random.randint(1, 7)
            position += dice
            if position in ladders:
                ladder_hits += 1
                ladder_gain += ladders[position] - position
                position = ladders[position]
            elif position in snakes:
                snake_hits += 1
                snake_penalty += position - snakes[position]
                position = snakes[position]

    print('Analysis of snakes and ladders encountered: \n')
    print(f'Snakes Encountered: {snake_hits}')
    print(f'Ladders Encountered: {ladder_hits}')
    print(f'Average Snake Penalty: {snake_penalty / max(snake_hits, 1):.2f}')
    print(f'Average Ladder Gain: {ladder_gain / max(ladder_hits, 1):.2f}\n')


snake_ladder_impact(snakes, ladders)


# Now running all of this for boards with no snakes and ladders or only snakes or only ladders
def only():
    rolls = Simulation({}, {})[0]
    avg, var = simulation_average(rolls)
    sd = np.sqrt(var)
    print('Results for simulations with no snakes or ladders: \n')
    print(f'Average number of rolls according to simulation: {avg}')
    print(f'Variance according to simulation results: {var}')
    print(f'Standard deviation according to simulation results: {sd}\n')

    markov_avg, markov_var, pdf, = Markov({}, {}, 50)
    markov_sd = np.sqrt(markov_var)
    print(f'Average number of rolls according to Markov: {markov_avg}')
    print(f'Variance according to Markov {markov_var}\n')
    print(f'Standard deviation according to Markov {markov_sd}\n')

    plt.hist(rolls, bins=20, edgecolor='black')
    plt.xlabel('Number of Rolls')
    plt.ylabel('Frequency')
    plt.title('Histogram if the board had no snakes or ladders')
    plt.show()

    # Plot histogram of simulation results for only board
    plt.hist(rolls, bins=20, density=True, alpha=0.6, edgecolor='black', label='Simulation Histogram')
    plt.axvline(markov_avg, color='black', label='Avg no. of rolls from sim data')

    # Plot PDF based on Markov chain for empty board
    plt.plot(range(1, len(pdf) + 1), pdf, color='red', linestyle='-', label='Markov Chain PDF')
    plt.xlabel('Number of Rolls')
    plt.ylabel('Probability of game ending after n rolls')
    plt.legend()
    plt.title('Probability Density Function of Rolls for no snakes/ladders to Complete Game(Analytically)')
    plt.figure().set_figwidth(20)
    plt.show()


def ladders_only():
    rolls_ladders_only = Simulation({}, ladders)[0]
    ladder_avg, ladder_var = simulation_average(rolls_ladders_only)
    ladder_sd = np.sqrt(ladder_var)
    print('Results for simulations with only ladders: \n')
    print(f'Average number of rolls according to simulation: {ladder_avg}')
    print(f'Variance according to simulation results: {ladder_var}')
    print(f'Standard deviation according to simulation results: {ladder_sd}\n')

    markov_ladder_avg, markov_ladder_var, ladder_pdf, = Markov({}, ladders, 40)
    markov_ladder_sd = np.sqrt(markov_ladder_var)
    print(f'Average number of rolls according to Markov: {markov_ladder_avg}')
    print(f'Variance according to Markov {markov_ladder_var}\n')
    print(f'Standard deviation according to Markov {markov_ladder_sd}\n')

    plt.hist(rolls_ladders_only, bins=30, edgecolor='black')
    plt.xlabel('Number of Rolls')
    plt.ylabel('Frequency')
    plt.title('Histogram if the board had only ladders')
    plt.show()

    # Plot histogram of simulation results for ladders only board
    plt.hist(rolls_ladders_only, bins=30, density=True, alpha=0.6, edgecolor='black', label='Simulation Histogram')
    plt.axvline(markov_ladder_avg, color='black', label='Avg no. of rolls from sim data')

    # Plot PDF based on Markov chain for LADDER board
    plt.plot(range(1, len(ladder_pdf) + 1), ladder_pdf, color='red', linestyle='-', label='Markov Chain PDF')
    plt.xlabel('Number of Rolls')
    plt.ylabel('Probability of game ending after n rolls')
    plt.legend()
    plt.title('Probability Density Function of Rolls for only ladders to Complete Game(Analytically)')
    plt.figure().set_figwidth(20)
    plt.show()


def snakes_only():
    rolls_snakes_only = Simulation({}, snakes)[0]
    snakes_avg, snakes_var = simulation_average(rolls_snakes_only)
    snakes_sd = np.sqrt(snakes_var)
    print('Results for simulations with only snakes: \n')
    print(f'Average number of rolls according to simulation with only snakes: {snakes_avg}')
    print(f'Variance according to simulation results with only snakes: {snakes_var}')
    print(f'Standard deviation according to simulation results with only snakes: {snakes_sd}\n')

    markov_snakes_avg, markov_snakes_var, snakes_pdf, = Markov({}, snakes, 350)
    markov_snakes_sd = np.sqrt(markov_snakes_var)
    print(f'Average number of rolls according to Markov: {markov_snakes_avg}')
    print(f'Variance according to Markov {markov_snakes_var}')
    print(f'Standard deviation according to Markov {markov_snakes_sd}\n')

    plt.hist(rolls_snakes_only, bins=50, edgecolor='black')
    plt.xlabel('Number of Rolls')
    plt.ylabel('Frequency')
    plt.title('Histogram if the board had only snakes')
    plt.show()

    # Plot histogram of simulation results for snakes only board
    plt.hist(rolls_snakes_only, bins=75, density=True, alpha=0.6, edgecolor='black', label='Simulation Histogram')
    plt.axvline(markov_snakes_avg, color='black', label='Avg no. of rolls from sim data')

    # Plot PDF based on Markov chain for snake only board
    plt.plot(range(1, len(snakes_pdf) + 1), snakes_pdf, color='red', linestyle='-', label='Markov Chain PDF')
    plt.xlabel('Number of Rolls')
    plt.ylabel('Probability of game ending after n rolls')
    plt.legend()
    plt.title('Probability Density Function of Rolls for only snakes to Complete Game(Analytically)')
    plt.figure().set_figwidth(20)
    plt.show()


ladders_only()
snakes_only()
only()
# Remove square 0 and reshape array to fit 10x10 board
board_visits = np.array(square_visit_counter[1:]).reshape((10, 10))

# Reverse every other row to show the snakes and ladders board layout
for i in range(1, 10, 2):
    board_visits[i] = board_visits[i][::-1]

# Plot the heatmap
plt.figure(figsize=(8, 8))
plt.imshow(board_visits, cmap='winter_r')
plt.colorbar(label='Number of Visits')
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Heatmap of Most Visited Squares on Snakes and Ladders Board')
plt.axis('off')

# Annotate each square with its position and indicate if it has a snake or ladder
for i in range(10):
    for j in range(10):
        # Calculate the actual board position (1 to 100) based on row and column
        position = i * 10 + j + 1
        if i % 2 == 1:  # Reverse numbering for every other row
            position = i * 10 + (9 - j) + 1

        # Determine if the position has a snake or ladder
        annotation = f'{position}'
        if position in ladders:
            annotation += ' L'  # Ladder
        elif position in snakes:
            annotation += ' S'  # Snake

        # Add text annotation to the square
        plt.text(j, i, annotation, ha='center', va='center', color='black', fontsize=8)
# Invert y-axis to match board layout (1 at bottom-left to 100 at top-right)
plt.gca().invert_yaxis()

plt.show()

normalized_visits = [v / sum(square_visit_counter) for v in square_visit_counter]
top_visited = sorted(enumerate(normalized_visits), key=lambda x: x[1], reverse=True)[:5]
least_visited = sorted(enumerate(normalized_visits[1:], start=1), key=lambda x: x[1])[:5]

# Print the least visited squares
print('Top 5 Least Visited Squares (excluding Square 0):')
for square, freq in least_visited:
    print(f'Square {square}: {freq:.2%}')

print('Top 5 Most Visited Squares:')
for square, freq in top_visited:
    print(f'Square {square}: {freq:.2%}')
