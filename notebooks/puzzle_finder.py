import cv2
import numpy as np
import matplotlib.pyplot as plt
from TMNIST_CNN import Net
import imutils
import torch 

model = Net()
model.load_state_dict(torch.load('./TMNIST_model.pt'))
model.eval()



def center(num):
    h, w = num.shape
    x_sums = np.sum(num, axis = 0)
    y_sums = np.sum(num, axis = 1)
    tot = np.sum(x_sums)
    x_avg, y_avg = 0, 0
    for idx, count in enumerate(x_sums):
        x_avg += idx * count
    for idx, count in enumerate(y_sums):
        y_avg += idx * count
    x_avg /= len(x_sums) * tot
    y_avg /= len(y_sums) * tot
    com = (int(x_avg * len(x_sums)), int(y_avg * len(y_sums)))
    shift = len(x_sums)//2 - com[0], len(y_sums)//2 - com[1]
    translation_matrix = np.float32([ [1,0, shift[0]], [0,1, shift[1]] ])
    cv2.warpAffine(num, translation_matrix, (h,w))
    return num

def find_puzzle(im):
    def get_corners(contour):
        top_left = contour[np.argmin([np.linalg.norm(x) for x in contour])][0]
        bottom_right = contour[np.argmax([np.linalg.norm(x) for x in contour])][0]
        points = [x[0] for x in contour]
        points = [(im_w - x, y) for x, y in points]
        top_right = contour[np.argmax([np.linalg.norm(x) for x in points])][0]
        bottom_left = contour[np.argmin([np.linalg.norm(x) for x in points])][0]
        return [top_left, bottom_left, top_right, bottom_right]
    im_h, im_w, _ = im.shape
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
     cv2.THRESH_BINARY,61,12)
    binary = cv2.bitwise_not(thresh)
    contours= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    puzzle = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(puzzle)
    puzzle_im = im[y:y+h, x:x+w]
    puzzle_gray = gray[y:y+h, x:x+w]
    corners = get_corners(puzzle)
    corners = np.array(corners)
    corners[:, 0] -= x
    corners[:, 1] -= y

    h, w = puzzle_im.shape[:2]    
    src = np.float32(corners)
    dst = np.float32([(w, 0),
                      (0, 0),
                      (w, h),
                      (0, h)])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(puzzle_gray, M, (w, h), flags=cv2.INTER_LINEAR)
    warped = cv2.flip(warped, 1)
    warped = cv2.resize(warped, (1000, 1000))
    
    return warped

def isolate_nums(puzzle):
    thresh = ~cv2.adaptiveThreshold(puzzle,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,61,12)
    empty = np.zeros_like(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if (cv2.contourArea(c) > 100 and cv2.contourArea(c) < 10000)]
    cv2.drawContours(empty, filtered_contours, -1, (255, 255, 255), -1)
    return empty

def split_squares(puzzle):
    squares = np.zeros((9, 9), dtype=object)
    h, w = puzzle.shape
    for row in range(9):
        for col in range(9):
            squares[row, col]  = puzzle[(h * row)//9:(h * (row + 1))//9, (w * col)//9:(w * (col + 1))//9]
    return squares

def center_squares(squares, number_mask):
    for row in range(9):
        for col in range(9):
            if number_mask[row, col]:
                squares[row, col] = center(squares[row, col])

def mask_numbers(squares):
    sums = np.array([np.sum(cv2.erode(square, np.ones((3, 3)))) for square in squares.ravel()])
    sums_sorted = np.array(sorted(sums))
    largest_gap = np.argmax(sums_sorted[1:] - sums_sorted[:-1])
    full_square_thresh = sums_sorted[largest_gap]
    sums = sums.reshape((9, 9))
    number_mask = sums > full_square_thresh
    return number_mask


def apply_model(squares, number_mask):
    board = np.zeros((9, 9), int)
    for row in range(9):
        for col in range(9):
            if number_mask[row, col]:
                num = squares[row, col]
                num = cv2.resize(num, (28, 28))
                #num_t = torch.Tensor(np.round(num/256))
                num_t = torch.Tensor(num/256)
                num_t = num_t.unsqueeze(0)
                num_t = num_t.unsqueeze(0)

                outputs = model(num_t)
                _, predicted = torch.max(outputs.data, 1)
                board[row, col] = int(predicted)
    return board