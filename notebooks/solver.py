import numpy as np

def check_row(row):
    legal = True
    unique, counts = np.unique(row, return_counts=True)
    count_dict = dict(zip(unique, counts))
    for num, count in count_dict.items():
        if num > 0 and count > 1:
            return False
    return True

def is_legal(board):
    rows = [r for r in board]
    cols = [c for c in board.T]
    boxes = []
    for r in (0, 3, 6):
        for c in (0, 3, 6):
            boxes.append(board[r:r+3, c:c+3].ravel())
    legal = True
    
    for i in range(9):
        if not (check_row(rows[i]) and check_row(cols[i]) and check_row(boxes[i])):
            return False
    return True     

def legal_options(board, indices):
    row, col = indices
    row_arr = board[row]
    col_arr = board.T[col]
    box = []
    start_row = row - row % 3
    start_col = col - col % 3
    for i in range(3):
        for j in range(3):
            box.append(board[i + start_row][j + start_col])
    box = np.array(box)
    all_illegal = set(row_arr).union((set(col_arr).union(set(box))))
    legal = [i for i in range(1, 10)if i not in all_illegal]
    
    return legal

def is_solved(board):
    return is_legal(board) and np.sum(board) == 405

def empty_mask(board):
    return board == 0

def get_empty_indices(board):
    mt_mask = empty_mask(board)
    mt_indices = np.argwhere(mt_mask)
    return [tuple(idx) for idx in mt_indices]

def sort_empty_indices(board, indices):
    return sorted(indices, key = lambda x: len(legal_options(board, x)))

def solve_sudoku(board):
    mt_indices = get_empty_indices(board)
    mt_indices = sort_empty_indices(board, mt_indices)
    active_index = 0
    guess_count = 0

    if not is_legal(board):
        return 'No Solution', guess_count
    
    while not is_solved(board):    
        guess_count += 1  
        num = board[mt_indices[active_index]]
        legal_moves = [l for l in legal_options(board, mt_indices[active_index]) if l > num]

        if len(legal_moves) > 0:
            board[mt_indices[active_index]] = legal_moves[0]
            mt_indices = mt_indices[:active_index + 1] + sort_empty_indices(board, mt_indices[active_index + 1:])
            active_index += 1
        else:
            board[mt_indices[active_index]] = 0
            active_index -= 1

        if active_index < 0:
            return 'No Solution', guess_count

    return board, guess_count