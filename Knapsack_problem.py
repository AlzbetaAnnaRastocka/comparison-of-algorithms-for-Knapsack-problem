import random
import math
from itertools import combinations
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# global variables to access weight or value of an item
WEIGHT_INDEX = 1
VALUE_INDEX = 0

# generator of an instance
def generate_instances(n_items, max_value, max_weight):

    items = []
    for _ in range(n_items):
        value  = random.randint(1, max_value)
        weight = random.randint(1, max_weight)
        items.append((value, weight))

    return items

# 1. BRUTE FORCE
def brute_force_knapsack(items, weight_limit):

    best_items_combination = []
    best_score = 0
    n = len(items)

    # We go through every possible combination of items, keeping track of the weight and value of given combination
    for number_of_items_in_combination in range(1, n+1):
        # combinations(items, number+of_items_in_combination) returns all subsets of lenght number_of_items_in_combination
        for subset in combinations(items, number_of_items_in_combination):
            value = sum(item[VALUE_INDEX] for item in subset)
            weight = sum(item[WEIGHT_INDEX] for item in subset)
            # if weight is within the weight limit and value is  better than current best score, we update the best score to this value
            if weight <= weight_limit and value > best_score:
                best_score = value
                best_items_combination = subset

    return list(best_items_combination), best_score

# 2. DYNAMIC PROGRAMMING
def dynamic_programming_knapsack(items, weight_limit): 
    number_of_items = len(items) 
    table = [[0 for x in range(weight_limit + 1)] for x in range(number_of_items + 1)] # initialization of the table (number+of+items +1) x (weight_limit + 1)
 
    for item_index in range(number_of_items + 1): 
        for backpack_capacity in range(weight_limit + 1): 
            if item_index == 0 or backpack_capacity == 0: # if backpack capacity is 0 or item index is 0 we fill the corresponding row or column with 0s
                table[item_index][backpack_capacity] = 0
            elif items[item_index-1][WEIGHT_INDEX] <= backpack_capacity: #
                table[item_index][backpack_capacity] = max(items[item_index-1][VALUE_INDEX] + table[item_index-1][backpack_capacity-items[item_index-1][WEIGHT_INDEX]],  table[item_index-1][backpack_capacity]) # decision to include or not nclude current item
            else: 
                table[item_index][backpack_capacity] = table[item_index-1][backpack_capacity] 

    # backtracking
    best_items_combination = []
    current_weight_limit = weight_limit
    for item_index in range (number_of_items, 0, -1):
        # check if item was included or not

        if table[item_index][current_weight_limit] == table[item_index - 1][current_weight_limit]: # if true item was not included
          continue
        else:
            current_weight_limit -= items[item_index-1][WEIGHT_INDEX]
            best_items_combination.append(items[item_index-1])
    return best_items_combination[::-1], table[number_of_items][weight_limit] 
 
    
# 3. GREEDY KNAPSACK
def greedy_knapsack(items, weight_limit):
    sorted_items_by_value = sorted(items, key=lambda x: (x[VALUE_INDEX] / x[WEIGHT_INDEX]), reverse=True) # descending sort of items by their value/weight ratio

    current_weight = 0
    value = 0
    best_items_combination = []

    for item in sorted_items_by_value:
        if current_weight == weight_limit:
            break
        if ((current_weight + item[WEIGHT_INDEX]) <= weight_limit) and (item[VALUE_INDEX] != 0):
            value += item[VALUE_INDEX]
            best_items_combination.append(item)
            current_weight += item[WEIGHT_INDEX]
    return best_items_combination, value

# 4. FPTAS KNAPSACK
def FPTAS_knapsack(items, weight_limit, epsilon):
    number_of_items = len(items)
    v_max = max(value for value, weight in items) # maximum value in the items list
    K = epsilon*v_max/number_of_items # scaling factor

    scaled_items = [(math.floor(value/K), weight) for value, weight in items] 

    maximum_value = sum(value for value, weight in scaled_items) # maximum value that we could get by including all items
    INF = float("inf")
    table = [[INF]*(maximum_value + 1) for i in range(number_of_items+1)] # we initialize a table (max value +1)x(number of items + 1) with all infinite values
    table[0][0] = 0 # left most corner

    for item_index in range(1, number_of_items+1): # skipping the first row
        value, weight = scaled_items[item_index - 1]
        for price in range(maximum_value + 1):
            table[item_index][price] = table[item_index-1][price] # in case item can not be included we always want to set the table position to weight of i - 1 items to reach the price
            if value <= price:
                include = table[item_index - 1][price - value] + weight 
                dont_include = table[item_index][price]
                table[item_index][price] = min(include, dont_include)

    best_price = max(price for price in range(maximum_value + 1) if table[number_of_items][price] <= weight_limit)

    #backtracking
    p = best_price
    best_items_combination = []
    for item_index in range(number_of_items, 0, -1):
        # check if item was added
        if table[item_index - 1][p] != table[item_index][p]:
            p -= scaled_items[item_index - 1][VALUE_INDEX]
            best_items_combination.append(items[item_index - 1])

    return best_items_combination[::-1], sum(value for value, weight in best_items_combination)

def FPTAS_knapsack_without_backtrack(items, weight_limit, epsilon):
    number_of_items = len(items)
    if number_of_items == 0:
        return [], 0  # Return empty list and 0 value if no items

    v_max = max(value for value, weight in items)
    K = epsilon * v_max / number_of_items
    scaled_items = [(math.floor(value / K), weight) for value, weight in items]
    maximum_value = sum(value for value, weight in scaled_items)
    INF = float('inf')

    # 1D DP array: dp[price] = minimal weight needed to achieve this price
    dp = [INF] * (maximum_value + 1)
    dp[0] = 0  # Base case: 0 value requires 0 weight

    # Update the DP array for each item
    for scaled_value, weight in scaled_items:
        # Iterate backward to avoid overwriting values prematurely
        for price in range(maximum_value, scaled_value - 1, -1):
            if dp[price - scaled_value] + weight < dp[price]:
                dp[price] = dp[price - scaled_value] + weight

    # Find the best price achievable within the weight limit
    best_price = max(
        (price for price in range(maximum_value + 1) if dp[price] <= weight_limit),
        default=0
    )

    # Calculate approximate total value (original scale)
    approximate_value = best_price * K

    # Return approximate value
    return approximate_value


# function to compare run time of all methods for various input sizes
def compare_different_input_size(input_sizes, max_weight, max_value, epsilons):
    methods = [
        ("Brute-force", brute_force_knapsack),
        ("DP",           dynamic_programming_knapsack),
        ("Greedy",       greedy_knapsack),
    ]
    results = []
    for n in input_sizes:
        
        items = generate_instances(n, max_value, max_weight)
        total_weight = sum(weight for _, weight in items)

        W = int(total_weight * 0.6)
        for name, func in methods:
            if name == "Brute-force" and n > 20:
                continue
            start = time.perf_counter()
            func(items, max_weight)
            elapsed = time.perf_counter() - start
            results.append({"method": name, "n": n, "time": elapsed})

        for eps in epsilons:
             start = time.perf_counter()
             FPTAS_knapsack_without_backtrack(items, max_weight, eps)
             elapsed = time.perf_counter() - start
             results.append({"method": f"FPTAS ε={eps}", "n": n, "time": elapsed})

    df = pd.DataFrame(results)

    plt.rcParams.update({
        "font.size": 22,             #font
        "axes.titlesize": 25,        # nadpis grafu
        "axes.labelsize": 25,        # popisky osí
        "xtick.labelsize": 15,       # popisky xtickov
        "ytick.labelsize": 28        #         ytickov
    })

    fig, ax = plt.subplots(figsize=(16, 9))
    for name in df["method"].unique():
        sub = df[df["method"] == name]
        ax.plot(sub["n"], sub["time"], marker="o", label=name)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(7, 1100)
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$'])
   
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend()
    ax.set_xlabel("Počet predmetov (n)")
    ax.set_ylabel("Čas (s)")
    ax.set_title("Čas behu jednotlivých algoritmov v závislosti od počtu predmetov")

    plt.tight_layout()
    plt.show()


def main():

    # Configurable parameters (Consider setting these before running the program):
    #   knapsack_weight_limit : capacity of the knapsack
    #   max_value_for_item    : maximum possible value of each item
    #   number_of_items       : how many items to generate
    #   epsilon               : approximation factor for FPTAS (e.g. 0.1)
    #   show_items            : # # set to False to print only the total value as a result of each method; if True, it also prints the selected items!
    
    max_weight= 100
    max_value_for_item = 100
    number_of_items = 10
    epsilon = 0.2
    show_items = True   # set to False if you only want to see the total values
    
    #_______________________________________________________________________________________

    #UNCOMMENT IF YOU WANT TO SEE COMPARISIONS OF RUN TIMES WITH INPUT SIZES AND EPSILONS
    # put list of input sizes, to compare run times
    input_sizes = [2**i for i in range(3, 11)]  # [8, 16, 32, 64, ..., 1024]
   
    epsilons = [0.2, 0.5, 0.8] # put here list of epsilons

    compare_different_input_size(input_sizes, max_weight, max_value_for_item, epsilons)

    #_______________________________________________________________________________________
    
    # generate an instance (list of pairs: (value, weight),  each pair is an item)
    # knapsack_weight_limit = 80
    # instance = generate_instances(number_of_items, max_value_for_item, max_weight = knapsack_weight_limit)
    # print(f"Vygenerovali sme inštanciu s počtom predmetov: {number_of_items}, maximálna hodnota predmetu je: {max_value_for_item} a kapacita batohu je {knapsack_weight_limit}.")

    # # brute force
    
    # start = time.perf_counter()
    # brute_force_solution, brute_force_value = brute_force_knapsack(instance, knapsack_weight_limit)
    # brute_force_time  = time.perf_counter() - start
    # # print output:
    # if show_items:
        # print("Brute-force:")
        # print(f"  Selected items:       {brute_force_solution}")
        # print(f"  Total value:          {brute_force_value}")
        # print(f"  Time taken:           {brute_force_time:.4f} s")
    # else:
        # print(f"Brute-force value: {brute_force_value}; time: {brute_force_time:.4f} s")


    # # dynamic programming
    # start = time.perf_counter()
    # dynamic_programming_solution, dynamic_programming_value = dynamic_programming_knapsack(instance, knapsack_weight_limit)
    # dynamic_programming_time  = time.perf_counter() - start
    # # print output:
    # if show_items:
        # print("Dynamic Programming:")
        # print(f"  Selected items:       {dynamic_programming_solution}")
        # print(f"  Total value:          {dynamic_programming_value}")
        # print(f"  Time taken:           {dynamic_programming_time:.4f} s")
    # else:
        # print(f"DP value: {dynamic_programming_value}; time: {dynamic_programming_time:.4f} s")


    # # greedy
    # start = time.perf_counter()
    # greedy_solution, greedy_value = greedy_knapsack(instance, knapsack_weight_limit)
    # greedy_time  = time.perf_counter() - start
    # # print output:
    # if show_items:
        # print("Greedy:")
        # print(f"  Selected items:       {greedy_solution}")
        # print(f"  Total value:          {greedy_value}")
        # print(f"  Time taken:           {greedy_time:.4f} s")
    # else:
        # print(f"Greedy value: {greedy_value}; time: {greedy_time:.4f} s")

    # # FPTAS
    # start = time.perf_counter()
    # FPTAS_solution, FPTAS_value  = FPTAS_knapsack(instance, knapsack_weight_limit, epsilon)
    # FPTAS_time  = time.perf_counter() - start
    # # print output:
    # if show_items:
        # print("FPTAS:")
        # print(f"  Selected items:       {FPTAS_solution}")
        # print(f"  Total value:          {FPTAS_value}")
        # print(f"  Time taken:           {FPTAS_time:.4f} s")
    # else:
        # print(f"FPTAS value: {FPTAS_value}; time: {FPTAS_time:.4f} s")


if __name__ == "__main__":
    main()