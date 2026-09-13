import numpy as np
import matplotlib.pyplot as plt

def plot_dancing_mutation_odds(paramMutOdds, generations=200):
    x = np.linspace(0, generations, generations)
    freq_wave = 1 + 0.5 * np.sin(x)
    phase = np.cumsum(freq_wave) * (x[1] - x[0])
    wave = np.sin(phase)
    mod_scalar = (wave + 1) / 2  # Normalize to range [0, 1]

    odds_over_time = []

    for i, base_val in enumerate(paramMutOdds):
        if i % 2 == 0:
            # Even indices ride the wave normally
            shifted = base_val * (0.5 + mod_scalar * 0.5)
        else:
            # Odd indices invert the wave
            shifted = base_val * (1 - mod_scalar * 0.5)
        odds_over_time.append(shifted)



    # Plotting
    plt.figure(figsize=(12, 6))
    for i, line in enumerate(odds_over_time):
        plt.plot(x, line, label=f'Nibble {i}')
    plt.title('Dancing Mutation Odds over Generations (Inverted Pairs)')
    plt.xlabel('Generation')
    plt.ylabel('Mutation Odds')
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
paramMutOdds = [
    0.1,  # source_type
    0.1,   # source_ID
    0.1,  # sink_type
    0.1,   # sink_ID
    0.1, 0.1, 0.1, 0.1,  # weight x4
    0.1,  # sourceNodeBias
    0.1   # initiativeGene
]

plot_dancing_mutation_odds(paramMutOdds)
