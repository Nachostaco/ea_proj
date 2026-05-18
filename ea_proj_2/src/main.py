import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import itertools
# FITNESS AND LOSS

def model(i:np.ndarray, a, b, c):
    return a * (i**2 - b * np.cos(c * np.pi * i))

def mse(params:np.ndarray, i_data, o_data):
    a, b, c = params
    return float(np.mean((o_data - model(i_data, a, b, c))**2))


# LOAD DATA
def load_data(filepath: str):
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]

# EVOLUTION STRATEGIES CORE

N_PARAMS = 3
TAU_PRIME = 1/np.sqrt(2*N_PARAMS)
TAU = 1/np.sqrt(2 * np.sqrt(N_PARAMS))

def _initialize_population(size: int, rng: np.random.Generator) -> np.ndarray:
    params = rng.uniform(-10.0, 10.0, (size, N_PARAMS))
    sigmas = rng.uniform(0.0, 10.0, (size, N_PARAMS))
    return np.hstack([params, sigmas])
    

def _mutate(individual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    params = individual[:N_PARAMS].copy()
    sigmas = individual[N_PARAMS:].copy()

    global_noise = rng.standard_normal()
    local_noise = rng.standard_normal(N_PARAMS)
    sigmas = sigmas * np.exp(TAU_PRIME * global_noise) * np.exp(TAU * local_noise)
    params = params + sigmas * rng.standard_normal()

    return np.concatenate([params, sigmas])

def _crossover_discrete(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.integers(0, 2, size=len(p1), dtype=bool)
    return np.where(mask, p1, p2)

def _crossover_intermediate(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.mean([p1, p2], axis=0)

CROSSOVER_FN = {
    'none': None,
    'discrete': _crossover_discrete,
    'intermediate': _crossover_intermediate,
}

def evolution_strategy(
        i_data:np.ndarray,
        o_data:np.ndarray,
        *,
        mu:int,
        lam: int,
        n_gen: int,
        threshold: float = 1e-5,
        scheme: str = 'comma',
        crossover: str = 'none',
        seed: int = 0,
        verbose: bool = True):
    if scheme not in ('comma', 'plus'):
        raise ValueError(f"scheme must be comma or plus, got {scheme}")
    if crossover not in CROSSOVER_FN:
        raise ValueError(f"crossover must be discrete or intermediate, got {crossover}")
    
    rng = np.random.default_rng(seed)
    label = f"({mu},{lam})" if scheme == "comma" else f"({mu}+{lam})"
    xover_fun = CROSSOVER_FN[crossover]

    population = _initialize_population(lam, rng)
    fitness = np.array([mse(individual[:N_PARAMS], i_data, o_data) for individual in population])

    best_idx = int(np.argmin(fitness))
    best_ind = population[best_idx].copy()
    best_mse = fitness[best_idx]
    history = []

    if verbose:
        _log_header(label)
        _log_row(0, best_mse, best_ind[:N_PARAMS])

    t0 = time.perf_counter()

    for gen in range(1, n_gen+1):
        parent_idx = np.argsort(fitness)[:mu]
        parents = population[parent_idx]

        offspring = np.empty((lam, 2*N_PARAMS))
        for k in range(lam):
            if xover_fun is not None:
                idx1, idx2 = rng.choice(mu, size=2, replace=False)
                child = xover_fun(parents[idx1], parents[idx2], rng)
            else:
                child = parents[k % mu]
            offspring[k] = _mutate(child, rng)
        
        offspring_fitness = np.array([mse(ind[:N_PARAMS], i_data, o_data) for ind in offspring])

        if scheme == "plus":
            pool = np.vstack([parents, offspring])
            pool_f = np.concatenate([fitness[parent_idx], offspring_fitness])
            keep = np.argsort(pool_f)[:lam]
            population = pool[keep]
            fitness = pool_f[keep]
        else:
            population = offspring
            fitness = offspring_fitness

        global_best_idx = int(np.argmin(fitness))
        if fitness[global_best_idx] < best_mse:
            best_mse = fitness[global_best_idx]
            best_ind = population[global_best_idx].copy()

        history.append(best_mse)

        if verbose and gen % 100 == 0:
            _log_row(gen, best_mse, best_ind[:N_PARAMS])

        if len(history)>2 and np.abs(best_mse - history[-2]) <= threshold: 
            break

    elapsed = time.perf_counter() - t0
    if verbose:
        _log_row(len(history), best_mse, best_ind[:N_PARAMS], final=True)
        print(f'Time elapsed: {elapsed:.2f}')

    return best_ind[:N_PARAMS], best_mse, history, elapsed

def run_benchmark(i_data, o_data, n_gen=1000):
    mu_lambda_pairs = [(10, 70), (15, 100), (30,200), (50, 300)]
    schemes = ['comma', 'plus']
    crossovers = ['none', 'discrete', 'intermediate']

    configs = list(itertools.product(mu_lambda_pairs, schemes, crossovers))
    total = len(configs)

    print(f"Benchmark for {total} configs runs")

    rows = []
    for i, ((mu, lam), scheme, xover) in enumerate(configs):
        _, best, _, elapsed = evolution_strategy(i_data, o_data,
                                                 mu=mu, lam=lam, n_gen=n_gen,
                                                 scheme=scheme, crossover=xover,
                                                 seed=42, verbose=False)
        med_mse = best
        med_time = elapsed
        label = f"({mu},{lam})" if scheme == "comma" else f"({mu}+{lam})"
        rows.append(dict(mu=mu, lam=lam, scheme=scheme, crossover=xover, label=label, mse=med_mse, time=med_time))
        done = i+1
        print(f" [{done:>3}/{total}]  {label:<10}  {scheme:<5}  "
              f"{xover:<13}  MSE={med_mse:.4f}  t={med_time:.2f}s")
        
    return rows


def _log_header(label: str):
    print(f"Evolution Strategy: {label}")
    print(f"  {'Gen':>6}  {'Best MSE':>14}  {'a':>9}  {'b':>9}  {'c':>9}")

def _log_row(gen, fit, params, final=False):
    marker = " end" if final else ""
    a,b,c = params
    print(f" {gen:>6}  {fit:>14.6f}  {a:>9.4f}  {b:>9.4f}  {c:>9.4f}{marker}")


COLORS  = {"none": "#4C72B0", "discrete": "#DD8452", "intermediate": "#55A868"}
MARKERS = {"comma": "o", "plus": "s"}
SCHEME_LABELS = {"comma": r"$(\mu,\lambda)$", "plus": r"$(\mu+\lambda)$"}

def plot_convergence(i_data, o_data, n_gen=800):
    """Side-by-side convergence curves for every crossover × scheme pair."""
    mu_lam = {"comma": (30, 300), "plus": (30, 300)}
    schemes    = ["comma", "plus"]
    crossovers = ["none", "discrete", "intermediate"]
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Convergence by scheme and crossover operator", fontsize=13)
 
    for ax, scheme in zip(axes, schemes):
        mu, lam = mu_lam[scheme]
        for xover in crossovers:
            _, _, history, _ = evolution_strategy(
                i_data, o_data,
                mu=mu, lam=lam, n_gen=n_gen,
                scheme=scheme, crossover=xover,
                seed=42, verbose=False,
            )
            ax.semilogy(history, linewidth=1.8,
                        color=COLORS[xover], label=xover)
 
        sym = f"({mu},{lam})" if scheme == "comma" else f"({mu}+{lam})"
        ax.set_title(f"{sym}-ES", fontsize=12)
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel("Best MSE  (log scale)", fontsize=11)
        ax.legend(title="crossover", fontsize=9)
        ax.grid(True, alpha=0.25, which="both")
 
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    print("Saved -> convergence.png")
    plt.show()



def plot_benchmark(rows: list):
    """
    Three-panel figure:
      A - MSE vs mu, grouped by crossover & scheme
      B - Time vs mu, grouped by crossover & scheme
      C - MSE vs Time scatter (all configs)
    """
    df = pd.DataFrame(rows)
 
    xovers  = ["none", "discrete", "intermediate"]
    schemes = ["comma", "plus"]
    ls_map  = {"comma": "--", "plus": "-"}
 
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
 
    # A: MSE vs mu
    ax = axes[0]
    for xover in xovers:
        for scheme in schemes:
            sub = df[(df.crossover == xover) & (df.scheme == scheme)]
            ax.semilogy(sub["mu"], sub["mse"],
                        color=COLORS[xover], linestyle=ls_map[scheme],
                        marker=MARKERS[scheme], markersize=6, linewidth=1.5,
                        label=f"{xover} / {SCHEME_LABELS[scheme]}")
    ax.set_xlabel(r"$\mu$  (parents)", fontsize=11)
    ax.set_ylabel("Median MSE  (log scale)", fontsize=11)
    ax.set_title("A — Quality vs Population Size")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=7, ncol=2)
 
    # B: Time vs mu
    ax = axes[1]
    for xover in xovers:
        for scheme in schemes:
            sub = df[(df.crossover == xover) & (df.scheme == scheme)]
            ax.plot(sub["mu"], sub["time"],
                    color=COLORS[xover], linestyle=ls_map[scheme],
                    marker=MARKERS[scheme], markersize=6, linewidth=1.5)
    ax.set_xlabel(r"$\mu$  (parents)", fontsize=11)
    ax.set_ylabel("Median time  (s)", fontsize=11)
    ax.set_title("B — Computation Time vs Population Size")
    ax.grid(True, alpha=0.2)
 
    # C: MSE vs Time scatter
    ax = axes[2]
    for xover in xovers:
        for scheme in schemes:
            sub = df[(df.crossover == xover) & (df.scheme == scheme)]
            ax.scatter(sub["time"], sub["mse"],
                       color=COLORS[xover], marker=MARKERS[scheme],
                       s=65, label=f"{xover} / {SCHEME_LABELS[scheme]}")
    ax.set_yscale("log")
    ax.set_xlabel("Median time  (s)", fontsize=11)
    ax.set_ylabel("Median MSE  (log scale)", fontsize=11)
    ax.set_title("C — Quality-Time Trade-off")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=7)
 
    plt.savefig("benchmark.png", dpi=150)
    print("Saved -> benchmark.png")
    plt.show()


def make_plot(i_data, o_data):
    params, fit, _, _ = evolution_strategy(
        i_data, o_data,
        mu=30, lam=300, n_gen=1000,
        scheme="plus", crossover="intermediate",
        seed=42, verbose=True,
    )
    i_dense = np.linspace(i_data.min(), i_data.max(), 600)
    a, b, c = params
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(i_data, o_data, s=18, color="#444", alpha=0.75, label="data")
    ax.plot(i_dense, model(i_dense, a, b, c), color="#E05C2E", linewidth=2,
            label=f"fit: a={a:.4f}, b={b:.4f}, c={c:.4f}\nMSE={fit:.6f}")
    ax.set_xlabel("i", fontsize=11)
    ax.set_ylabel("o", fontsize=11)
    ax.set_title(r"Best fit — $(30+300)$-ES, intermediate crossover")
    ax.legend(); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("best_fit.png", dpi=150)
    print("Saved -> best_fit.png")
    plt.show()



def main():
    i_data, o_data = load_data('./resources/model4.txt')
    GENERATIONS = 1000
    plot_convergence(i_data, o_data, n_gen=GENERATIONS)
    rows = run_benchmark(i_data, o_data, GENERATIONS)
    plot_benchmark(rows)

    make_plot(i_data, o_data)
if __name__ == "__main__":
    main()
