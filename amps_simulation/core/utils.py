import matplotlib.pyplot as plt
import sympy as sp

def plot_results(t, x, state_vars):
    """
    Plots the state variable trajectories over time.

    Parameters:
    - t: Time points from the simulation.
    - x: State variable trajectories (each row corresponds to a state).
    - state_vars: List of symbolic state variables.
    """
    plt.figure(figsize=(8, 5))

    # Plot each state variable
    for i, state in enumerate(state_vars):
        if i < len(x):  # Check if we have data for this state
            plt.plot(t, x[i], label=f"${sp.latex(state)}$")
        else:
            print(f"Warning: No data available for state variable {state}")

    plt.xlabel("Time (s)")
    plt.ylabel("State Variables")
    plt.title("Circuit State Response Over Time")
    plt.legend()
    plt.grid()
    plt.show()