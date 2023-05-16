# APF_inverse_design

**APF** (Augmented Partial Factorization) is an efficienct method proposed in [Nature Computational Science 2, 815–822 (2022)](https://www.nature.com/articles/s43588-022-00370-6) that can solve multi-input electromagnetic forward problems in one shot, offering substantial speed-up and memory usage reduction compared to existing methods. Here, we generalize APF to enable efficient gradient computation for and inverse design of multi-input nanophotonic devices.

This repository includes codes used to inverse design a metasurface beam splitter using APF method, which splits the incident light equally into the $\pm$ 1 diffraction orders for any incident angle within a wide angular range. Detailed documentation is given in the comments of each file.
* [`inverse_design_beam_splitter.m`](inverse_design_codes/inverse_design_beam_splitter.m): Build the system and perform the optimization
* [`FoM_and_grad.m`](inverse_design_codes/FoM_and_grad.m): Compute the figure of merit and its gradient with respect to optimization variables.
* [`constraint_and_grad.m`](inverse_design_codes/constraint_and_grad.m): Build the inequality constraint and its gradient with respect to optimization variables.
* [`gradient_descent_BLS.m`](inverse_design_codes/gradient_descent_BLS.m): Perform the gradient-descent optimization with the learning rate determined by [backtracking line search](https://en.wikipedia.org/wiki/Backtracking_line_search).
* [`build_epsilon_pos.m`](inverse_design_codes/build_epsilon_pos.m): Build the permittivity profile.


To take fully advantage of the APF method and efficienctly perform optimizations, one needs to
* Install [MESTI.m](https://github.com/complexphoton/MESTI.m/tree/main) [download it and add the `MESTI.m/src` folder to the search path using the `addpath` command in MATLAB], the serial version of [MUMPS](http://mumps-solver.org/index.php?page=home) and its MATLAB interface.
* Install [NLopt](http://github.com/stevengj/nlopt) to use various well-developed algorithms for optimizations. If `NLopt` is not installed, one can specify other methods in [`inverse_design_beam_splitter.m`](inverse_design_codes/inverse_design_beam_splitter.m).
* Run [`inverse_design_beam_splitter.m`](inverse_design_codes/inverse_design_beam_splitter.m) [one can customise the optimization using `options`; check the documentation of this file for more details].

An animation on the evolution of the metasurface and its transmission matrix is shown below.

<img align="center" src="https://github.com/complexphoton/APF_inverse_design/blob/main/inverse_design_codes/animated_opt.gif" width=60% height=60%>

The corresponding paper will be on arXiv within the next few days.
