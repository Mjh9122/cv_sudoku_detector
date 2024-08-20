# Sudoku detector/solver
## Michael Holtz

### Goal 
Create a program that can extract and solve a sudoku puzzle from an image or video. 

### Method
#### Detect Puzzle
Read in frame from video  
<p align="center">
  <img src="https://github.com/Mjh9122/sudoku/blob/main/figures/correctly_solved_frame.png" width="300" class = "center"> 
</p>

 
Threshold image to accentuate edges and remove noise  
<p align="center">
  <img src="https://github.com/Mjh9122/sudoku/blob/main/figures/binary.png" width="300">  
</p>
Find puzzle by looking for the largest countour present in the frame   
<p align="center">
  <img src="https://github.com/Mjh9122/sudoku/blob/main/figures/puzzle.png" width="300">  
</p>
Warp puzzle to correct for perspective 
<p align="center">
  <img src="https://github.com/Mjh9122/sudoku/blob/main/figures/detected_puzzle.png" width="300">  
</p>
#### Fill in Cells
1. Classify blank/number cells
  1. I found that summing pixels in each cell clustered empty and full cells rather well.
  2. I took the largest gap in pixel sum to be the decision boundary
2. Pass full cells to digit classifier
   1. I trained a ![small CNN](https://github.com/Mjh9122/sudoku/blob/main/notebooks/TMNIST_CNN.py) based on the ![TMNIST dataset](https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist).
   2. The CNN was trained ![here](https://github.com/Mjh9122/sudoku/blob/main/notebooks/TMNIST_trainer.ipynb) and achieved 98% accuracy on a validation set.
#### Solve
7. Once the board is filled out, I apply ![my solver](https://github.com/Mjh9122/sudoku/blob/main/notebooks/solver.py)
   1. The solver works on a combination of constraint propogation and backtracking, guessing the cells with the smallest number of valid inputs at each step.
   2. The solver was !(benchmarked)[https://github.com/Mjh9122/sudoku/blob/main/solver_benchmark/benchmark_solver.ipynb] on 10,000 puzzles and averaged 42 guesses per solve.

