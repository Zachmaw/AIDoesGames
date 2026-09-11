# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(0, 2* 4 * np.pi, 1000)
# waves = {
#     "normal": np.sin(x),
#     "sin^3": np.sin(x) ** 3,
#     # "skewed": np.sin(x) * np.abs(np.sin(x)),
#     "mod freq": np.sin(np.cumsum(1 + 0.5 * np.sin(x)) * (x[1] - x[0]))# accumulate frequency shift
# }

# for label, y in waves.items():
#     plt.plot(x, y, label=label)

# plt.legend()
# plt.title("Shaped Sine Waves")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# ## was gonna delete everything.





import numpy as np
import matplotlib.pyplot as plt
import os
from math import exp, e




# Example usage:
params = [
    0.05,  # source_type
    0.1,   # source_ID
    0.05,  # sink_type
    0.1,   # sink_ID
    0.05, 0.2, 0.05, 0.2,  # weight x4
    0.05,  # sourceNodeBias
    0.15   # initiativeGene
]


# Example usage
paramMutOdds = [
    0.05,  # source_type
    0.1,   # source_ID
    0.05,  # sink_type
    0.1,   # sink_ID
    0.05, 0.2, 0.05, 0.2,  # weight x4
    0.05,  # sourceNodeBias
    0.15   # initiativeGene
]







def generateLookupTable(usedFunc, param1, param2):# make the txt file with the info
    # makeFile
    # perform func with params
    return usedFunc(param1, param2)



def generateInitBonusFunc(peak: int, max_len: int = 150):
    pass


def generate_fetch_bonus_table():# pregenerate the lookup table for the simulation's given generation-based initializing  genome peak.
    values = [round(((e/peak) * x * exp(-x/peak) * 6) - 3) for x in range(max_len)]## max_len is an index type of value.
    filename = make_filename(peak, max_len)
    with open(filename, 'w') as f:
        for val in values:
            f.write(str(val) + '\n')
    return values



################ FILE FUNCS
def make_filename(funcContext: list[str], param1: int, param2: int):### context = [funcName, param1Name, param2Name]
    folder = os.path.join("cached_data", funcContext)
    os.makedirs(folder, exist_ok=True)### constrain funcContext short and keep params numeric and padded to 4 deca-bits.
    return os.path.join(folder, f"{funcContext[0]}_{funcContext[1]}-{param1}_{funcContext[2]}-{param2}.txt")
# def get_filename(func_name: str, peak: int, max_len: int = 150):
#     return f"{func_name}_peak{peak}_len{max_len}.txt"

def rename_file(old_name: str, new_name: str):
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
    else:
        print(f"File '{old_name}' does not exist.")

def write2file(filename: str, text: str):# one line at a time.
    with open(filename, 'a') as file:
        file.write(text + '\n')
def readFile(filename: str, dataType):
    try:
        with open(filename, 'r') as f:
            return [dataType(line.strip()) for line in f.readlines()]
    except FileNotFoundError:
        return False
    except:
        return False

def getFromBonusTable(peak: int, max_len: int = 150):
    filename = make_filename(peak, max_len)
    return generate_fetch_bonus_table(peak, max_len)








# n = generateLookupTable(nibbleSelectionSinFunc, ???)
def nibbleSelectionSinFunc(x):
    x = np.linspace(0, 150, 1000)
    y = np.sin(np.cumsum(1 + 0.5 * np.sin(x)) * (x[1] - x[0]))# represents how much the default chances are shifted

    pass




def getDancingMutOdds(params: list[float], val: float):
    x = np.linspace(0, 150, 151)
    freq_wave = 1 + 0.5 * np.sin(x)
    phase = np.cumsum(freq_wave) * (x[1] - x[0])
    wave = np.sin(phase)
    mod_scalar = (wave + 1) / 2  # Normalize to [0, 1]




    new_odds_over_time = []
    for i in range(len(params)):
        direction = 1 if i % 2 == 0 else -1  # even indices go up, odd go down
        shifted = params[i] + direction * (mod_scalar - 0.5) * params[i]
        new_odds_over_time.append(shifted)


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

def plot_dancing_mutation_odds(paramMutOdds, generations=200):
    x = np.linspace(0, generations, generations)
    freq_wave = 1 + 0.5 * np.sin(x)
    phase = np.cumsum(freq_wave) * (x[1] - x[0])
    wave = np.sin(phase)
    mod_scalar = (wave + 1) / 2  # Normalize to [0, 1]

    odds_over_time = []
    for i in range(len(paramMutOdds)):
        direction = 1 if i % 2 == 0 else -1  # even indices go up, odd go down
        shifted = paramMutOdds[i] + direction * (mod_scalar - 0.5) * paramMutOdds[i]
        odds_over_time.append(shifted)

    # Plotting
    plt.figure(figsize=(12, 6))
    for i, line in enumerate(odds_over_time):
        plt.plot(x, line, label=f'Nibble {i}')
    plt.title('Dancing Mutation Odds over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Mutation Odds')
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()






plot_dancing_mutation_odds(paramMutOdds)

plot_dancing_mutation_odds(getDancingMutOdds(params))

