The main reason to keep this code is to train with the curriculum learning approaches. If curriculum learning is not needed the better code to use would be the training_multiproblem_wo_dataset_cached.py as it allows to use it with the problem manager and suite of problems already written.

## curriculum learning
  This technique comes from reinforcement learning where the complexity and size of the dataset increases as the epochs go by
  We emulate this in our approach by increasing the lenght of the simulation this achieves two things
  - When paired with a suitable learning rate scheduler and an early stopper it could mean faster convergence
  - More stable training as the model is able to navigate the loss landscape better at first, this also reduces the strain on the other hyperparameters such as learning rate

## Types of CL implemented
  In typical CL theres some measurement of difficulty within the dataset. In our approach we haven't implemented such measure, instead we just increase the lenght of the simulation by arbitrarily chosen time points, the time of the simulation can be improved in two different fashions
  1. FRONT TO BACK: Here we fix the end of the simulation time and keep reducing the initial states time
  2. BACK TO FRONT: We fix the starting time and keep increasing the end time
  So far we've seen better performance from FTB but our conclusions are not determinant and more analysis is expected.
