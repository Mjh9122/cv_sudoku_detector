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
   2. The solver was ![benchmarked](https://github.com/Mjh9122/sudoku/blob/main/solver_benchmark/benchmark_solver.ipynb) on 10,000 puzzles and averaged 42 guesses per solve.

### Results
Results are based on a 5ish second video taken on my iphone of a sudoku puzzle book. 

#### Cell accuracy
The blanks were classified correctly 99.56 percent of the time, vindicating the simple gap based classification method. Numerical cells were classified correctly 84.60 percent of the time, a fair bit off from the 98% accuracy from training. This is likely due to the noise of real images vs. typeface and the fact that this particular font my not have been in the training dataset. These combined for an overall cell accuracy of 94.39 percent. 
<p align="center">
  <img src="https://github.com/Mjh9122/sudoku/blob/main/figures/Accuracy_across_frames.png" width="500">  
</p>

This histogram shows the accuracy of cell filling across each frame of the video. Blue represents all cells, while orange is only cells with numbers in the ground truth. 

#### Board accuracy 
The correct board was detected 124 times, 33.60 percent of the total frames. While this number is disapointing on its lonesome, the noisy data insures that each type of mistake is not often repeated. This leaded to the correct board taking a commanding plurality amongst many errors that only appear once or twice. 

<p align="center">
  <img src="https://github.com/Mjh9122/sudoku/blob/main/figures/board_freq.png" width="500">  
</p>

This histogram shows the frequency of each board detected across all frames. The correct board shows up 4x as often as the nearest erronious board (124 vs. 31 times). 


#### Solution accuracy 
The lack of repeated errors helps even more when looking only at valid solutions. Each mistake has the chance of making its board unsolvable, while the correct board while always give the same solution. This means that all 124 correct detections remain, while the number of erronious boards falls from 245 to 62. 2/3 of our valid solutions are correct and the correct solution apeares 10x as much as the most popular incorrect solution. 

<p align="center">
  <img src="https://github.com/Mjh9122/sudoku/blob/main/figures/solution_freq.png" width="500">  
</p>




