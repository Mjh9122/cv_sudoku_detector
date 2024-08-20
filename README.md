# Sudoku detector/solver
## Michael Holtz

### Goal 
Create a program that can extract and solve a sudoku puzzle from an image or video. 

### Method
1. Read in frame from video
2. Threshold image to accentuate edges and remove noise
3. Find puzzle by looking for the largest countour present in the frame
4. Warp puzzle to correct for perspective
5. Classify blank/number cells
  1. I found that summing pixels in each cell clustered empty and full cells rather well.
  2. I took the largest gap in pixel sum to be the decision boundary
6. Pass full cells to digit classifier
   1. I trained a ![small CNN](https://github.com/Mjh9122/sudoku/blob/main/notebooks/TMNIST_CNN.py) based on the ![TMNIST dataset](https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist).
   2. The CNN was trained ![here](https://github.com/Mjh9122/sudoku/blob/main/notebooks/TMNIST_trainer.ipynb) and achieved 98% accuracy on a validation set.
7. Once the board is filled out, I apply my solver
   1. The solver works on a combination of constraint propogation and backtracking.
   2. The solver was benchmarked on 10,000 puzzles and averaged 42 guesses per solve.

