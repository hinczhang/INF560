## notes  
add `-fopenmp` in `CFLAGS`.  

## parallel notes  
### 10.02.2022  
Add parallel in `nbody_alloc.c` - 31st line - for loop: speed up  
Add parallel in `nbody_tools.c` - 222nd line - for loop: speed down  
remove in `nbody_alloc.c` - 31st line - for loop: spped up a little  
**Conclusion: parallelism's cost**  
Finally add the removed one.  
**!!!** in `nbody_alloc.c` - 31st line - for loop: not permit! Sequent storage visit.  
Failed to parallelize the node insert: due to the storage visit.  

### 11.02.2022  
Add parallel in `nbody_barnes_hut.c` - 114th line - for loop: greatly speed down due to the recursion problem.  
Add parallel in `nbody_barnes_hut.c` - 135th line - for loop: greatly speed down due to the recursion problem.  
Fail to add parallel in `nbody_barnes_hut.c` - 184th line - for loop: the same as `insert_particle`: the storage visit.  
  
Remove all parallel sections, only remain `nbody_alloc.c` - 31st line - for loop  

We try to revise the structure of `compute_force_in_node`: In the high levels, we carry out the parallel algorithm; or, use the sequent algorithm.  
**Conclusion: The parallel initialization leads to approximate 0.2s time cosumption, but the parallelism leads to the original consumption. However, by deeper level parallelism, it is possible to make the time consumption rising again.**  
It is not worthy trying to parallelize `compute_force_on_particle`, as the particles' depth is too low and is not of for-loop format, where the recursion tasks will waste much time.  

### 01.03.2022
Parallelism succeeds in the force computing.  
However, the recursion problem has not yet solved, as adding `#pragma omp task` may slightly increase time (but better than the origin). Move and compute recursion can apply this way.