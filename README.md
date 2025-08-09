# Optimization algorithms

## Differential Evolutionary (DE/rand/1/bin) (`differentialEvolutionary.ipynb`)

Differential evolutionary algorithm on CPU.

- **Featuresm**  
  - **Initialization**: Population `pop` is initialized uniformly within the specified bounds.
  - **Mutation**: Three distinct vectors `a`, `b`, `c` are picked from the population and generate $v=x_a+F\cdot (x_b-x_c)$ which is the classic rand/1 mutation strategy..
  - **GPU**: On NVIDIA GPUs, JAX uses `cuSOLVER`'s `?gesvd` for smaller matrices or `?gesvdp` for larger ones, depending on performance. They use 1. bidiagonalization plus 2. divide-and-conquer or 2. QR-based methods, respectively.


:



 ✅

Crossover:
Binomial crossover is applied via a mask cross_points, with at least one dimension forced to mutate. ✅

Bounds enforcement:
np.clip ensures all trial vectors remain within the provided search bounds. ✅

Selection:
You compare the trial’s objective value f_trial with the parent’s fobj[j] and replace if better. ✅

Tracking:
The best value per generation is stored in best_value_history
