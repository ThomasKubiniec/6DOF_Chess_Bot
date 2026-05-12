'''
This is mostly a class to make testing data for the robot ik solvers (NN and Optimizer).
The real system will grab the x, y, coordinates from the camera (converting them to real world coords)
and the piece dicitonary will tell the robot the grab height of the piece.

Assign each square an x, y coordinate, and each piece a z coordinate
'''

import torch
import random

class chess_coords:
    def __init__(self, 
                 board_width, board_length, board_thickness, 
                 a1_xy_tup):
        
        self.board_y = board_width
        self.board_x = board_length # the robot's x direction is pointing down the board to the opponent
        self.board_z = board_thickness # the robot's y direction is pointing to the left, to the right of the opponent
        
        self.a1_x = a1_xy_tup[0] # distance in x from the robot worldframe to the a1 center
        self.a1_y = a1_xy_tup[1] # distance in y from the robot worldframe to the a1 center

        self.piece_list_w = ['p', 'h', 'b', 'r', 'q', 'k']
        self.piece_list_b = ['P', 'H', 'B', 'R', 'Q', 'K']

        self.assign_board_coords()


        # these are the grasping heights, local minima in width
        # lower = white, UPPER = black
        self.piece_z_grasp_dict = {
            'p' : 5/8, # white pawn
            'h' : 7/16, # white knight
            'b' : 7/8, # white bishop
            'r' : 15/16, # white rook
            'q' : 1 + 5/16, # white queen
            'k' : 1 + 3/4, # white king

            'P' : 5/8, # black pawn
            'H' : 7/16, # black knight
            'B' : 7/8, # black bishop
            'R' : 15/16, # black rook
            'Q' : 1 + 5/16, # black queen
            'K' : 1 + 3/4 # black king
        }


        # these are the grasping heights, local minima in width
        # lower = white, UPPER = black
        self.piece_z_clearance_dict = {
            'p' : 2, # white pawn
            'h' : 3, # white knight
            'b' : 2, # white bishop
            'r' : 2, # white rook
            'q' : 3, # white queen
            'k' : 3, # white king
            
            'P' : 2, # black pawn
            'H' : 3, # black knight
            'B' : 2, # black bishop
            'R' : 2, # black rook
            'Q' : 3, # black queen
            'K' : 3 # black king
        }




    def assign_board_coords(self):
        self.col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.row_names = ['1', '2', '3', '4', '5', '6', '7', '8']

        square_x = self.board_x/8
        square_y = self.board_y/8

        center_offset_x = square_x/2
        center_offset_y = square_y/2

        x_start = self.a1_x + center_offset_x
        y_start = self.a1_y + center_offset_y
        
        self.x_coords = torch.linspace(start= x_start, end= square_x * 7, steps= 8)
        self.y_coords = torch.linspace(start= y_start, end= square_y * 7, steps= 8)

        print(f'x center of squares = {self.x_coords}')
        print(f'y center of squares = {self.y_coords}')

        self.row_dict = {}
        for i, x in enumerate(self.x_coords):
            self.row_dict[self.row_names[i]] = x

        self.col_dict = {}
        for j, y in enumerate(self.y_coords):
            self.col_dict[self.col_names[j]] = y

        



    def make_random_waypoints(self):
        rand_piece = random.choice(self.piece_list_w)
        
        rand_row1 = random.choice(self.row_names)
        rand_col1 = random.choice(self.col_names)
        
        rand_row2 = random.choice(self.row_names)
        rand_col2 = random.choice(self.col_names)
        
        xyz_0 = self.row_dict[rand_row1], self.col_dict[rand_col1], self.piece_z_grasp_dict[rand_piece] + self.board_z

        xyz_1 = self.row_dict[rand_row1], self.col_dict[rand_col1], self.piece_z_clearance_dict[rand_piece] + self.board_z

        xyz_2 = self.row_dict[rand_row2], self.col_dict[rand_col2], self.piece_z_clearance_dict[rand_piece] + self.board_z

        xyz_3 = self.row_dict[rand_row2], self.col_dict[rand_col2], self.piece_z_grasp_dict[rand_piece] + self.board_z
        

        return [xyz_0, xyz_1, xyz_2, xyz_3]