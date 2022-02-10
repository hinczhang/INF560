## notes
add `-fopenmp` in `CFLAGS`.

## parallel notes
### 10.02.2022
Add parallel in `nbody_alloc.c` - 31st line - for loop: speed up \\
Add parallel in `nbody_tools.c` - 222nd line - for loop: speed down \\
remove in `nbody_alloc.c` - 31st line - for loop: spped up a little \\
**Conclusion: parallelism's cost** \\
Finally add the removed one. \\
**!!!** in `nbody_alloc.c` - 31st line - for loop: not permit! Sequent storage visit. \\
Failed to parallelize the node insert: due to the storage visit. \\

### 11.02.2022
Add parallel in `nbody_barnes_hut.c` - 114th line - for loop: greatly speed down due to the recursion problem. \\
Add parallel in `nbody_barnes_hut.c` - 135th line - for loop: greatly speed down due to the recursion problem. \\
Fail to add parallel in `nbody_barnes_hut.c` - 184th line - for loop: the same as `insert_particle`: the storage visit. \\