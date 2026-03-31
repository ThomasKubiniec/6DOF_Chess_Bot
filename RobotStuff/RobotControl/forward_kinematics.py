"""
DH based Forward Kinematics and Numerical Jacobians

"""

import numpy as np

class Robot_math:
    def __init__(self, 
                 a, 
                 alpha, 
                 d, 
                 theta, 
                 WT= None, 
                 joint_type=None, 
                 bounds=None,
                 fail_dist=None,
                 pad_dist=None):
        self.a = a # link length
        self.alpha = alpha # link twist
        self.d = d # link offset
        self.theta = theta # joint angle

        if WT is None: # no Worldframe Transformation
            self.WT = np.eye(4)
        else: # Worldframe Transformation is a 4x4 homogeneous matrix
            self.WT = WT

        self.joint_type = joint_type # 'r' for revolute, 'p' for prismatic
        # tuple of how far the joint can rotate or extend in either direction
        self.joint_bounds = bounds
        self.fail_dist_vect = fail_dist # radius of the link
        self.pad_dist_vect = pad_dist # distance between the joints and the links

        self.q_vect = np.zeros(len(self.a)) # values of each joint (should be kept in range of joint bounds)

        self.curr_joint_H = [] # keeps track of the current homogeneous transformation for each joint
        self.curr_jacobian = [] # keeps track of Jacobian matrix for joints


    def build_DH_A(self, a_i, alpha_i, d_i, theta_i):
        sa = np.sin(alpha_i)
        ca = np.cos(alpha_i)
        
        st = np.sin(theta_i)
        ct = np.cos(theta_i)

        A = np.array([[ct, (-st * ca), (st * sa), (a_i * ct)],
                      [st, (ct * ca), (-ct * sa), (a_i * st)],
                      [0, sa, ca, d_i],
                      [0, 0, 0, 1]])
        
        return A
    
 
    # def FK(self, input_qs):
    def FK(self):
        '''
        returns Forward Kinematics for 
        each joint in the sequence defined by the DH parameters
        '''
        # for i, q in enumerate(input_qs): # make sure joints are in bounds
        #     low, high = self.joint_bounds[i]
        #     if low <= q <= high:
        #         continue
        #     else:
        #         print(f"joint_{i} is out of bounds")
        #         return self.curr_joint_H

        # H = self.WT @ np.eye(4) # apply Worldframe Transformation by premultiplying
        H = self.WT # since WT is I(4) if not otherwise stated, premult is unnecessary. 
        joint_FKs = []
        for i in range(len(self.a)):
            if self.joint_type[i] == 'r': # if revolute joint, add q to theta
                # print('joint type is rev')
                t = self.theta[i] + self.q_vect[i]
                d = self.d[i]
                
            elif self.joint_type[i] == 'p': # if prismatic joint, add q to d
                # print('joint type is pris')
                t = self.theta[i]
                d = self.d[i] + self.q_vect[i]


            A = self.build_DH_A(a_i= self.a[i],
                                alpha_i= self.alpha[i],
                                d_i= d,
                                theta_i= t)
            # print(f'A = \n{A}')
            H = H @ A
            # print(f'H = \n{H}')
            joint_FKs.append(H)

        self.curr_joint_H = joint_FKs
        return self.curr_joint_H


    def give_ds(self):
        '''
        return the d_vector of H for every frame of the robot: Worldframe to n
        '''
        self.FK()
        ds = []

        WT_d = self.WT[0:3, 3]
        ds.append(WT_d)

        for j in self.curr_joint_H:
            my_d = j[0:3, 3]
            ds.append(my_d)
            
        return ds
    


    def give_Rs(self):
        '''
        return the R matricies of H every frame of the robot: Worldframe to n
        '''
        self.FK()
        Rs = []

        WT_R = self.WT[0:3, 0:3]
        Rs.append(WT_R)

        for j in self.curr_joint_H:
            my_R = j[0:3, 0:3]
            Rs.append(my_R)
            
        return Rs




    def J(self):
        '''
        J_omega = 0 (for prismatic)
        J_omega = z_i_min_1 (for revolute)

        J_vel = z_i_min_1 (for prismatic) 
        J_vel = z_i_min_1 cross (o_n - o_i_min_1) (for revolute) 

        Ji = [[Jv_1, ..., Jv_i],
              [Jw_1, ..., Jw_i]]

        for two revolute joints:
        
        J2 = [[z0 x (o2 - o1), z1 x (o2 - o1)],
              [z0, z1]]
        '''
        Jacobians = [] # jacobians for each joint
        # get positions and orientations of the joints
        os = [] # positions of joints in worldframe
        zs = [] # orientations of z axis in worldframe


        my_Hs = []
        my_Hs.append(self.WT)
        my_Hs.extend(self.curr_joint_H)
        # for H in my_Hs:
            # print(f'H = \n{H}')

        for H in my_Hs:
            os.append(H[0:3, 3]) # tx, ty, tz column of Homogeneous matrix
            zs.append(H[0:3, 2]) # r31, r32, r33
    

        for i in range(1, len(my_Hs)): # from z0 to zn-1
            # print('at top of i loop')
            my_cols = [] # list of the jacobian column vectors
            for j in range(i): #J1 to Jn
                # print('at top of j loop')
                
                if self.joint_type[j] == 'r': # if revolute joint
                    Jv = np.cross( np.array(zs[j]), np.array((os[i] - os[j])) ) # z_i_min_1 x (o_n - o_i_min_1)
                    Jw = np.array(zs[j]) #zs[i]
                    # print(f'rev Jv = {Jv}, rev Jw = {Jw}')
                elif self.joint_type[j] == 'p': # if prismatic joint
                    Jv = np.array(zs[j])
                    Jw = np.zeros(3)
                    # print(f'pris Jv = {Jv}, pris Jw = {Jw}')
                
                J_col = np.concatenate((Jv, Jw)) # [Jv[0], Jv[1], Jv[2], Jw[0], Jw[1], Jw[2]]
                my_cols.append(J_col)

            my_J_matrix = np.column_stack(my_cols) # jacobian for joint i
            Jacobians.append(my_J_matrix)

        self.curr_jacobian = Jacobians
        return Jacobians 
    




    def closest_point_segment_segment(self, p1, p2, p3, p4):
        """
        Returns the minimum distance between segments p1-p2 and p3-p4.
        """
        d1 = p2 - p1  # direction of segment 1
        d2 = p4 - p3  # direction of segment 2
        r = p1 - p3

        a = np.dot(d1, d1)  # |d1|^2
        e = np.dot(d2, d2)  # |d2|^2
        f = np.dot(d2, r)

        if a < 1e-10 and e < 1e-10:  # both degenerate
            return np.linalg.norm(r)

        if a < 1e-10:  # segment 1 is a point
            s = 0.0
            t = np.clip(f / e, 0, 1)
        else:
            c = np.dot(d1, r)
            if e < 1e-10:  # segment 2 is a point
                t = 0.0
                s = np.clip(-c / a, 0, 1)
            else:
                b = np.dot(d1, d2)
                denom = a * e - b * b
                if denom != 0:
                    s = np.clip((b * f - c * e) / denom, 0, 1)
                else:
                    s = 0.0  # parallel segments, arbitrary choice
                t = (b * s + f) / e
                # re-clamp t and re-derive s
                if t < 0:
                    t = 0.0
                    s = np.clip(-c / a, 0, 1)
                elif t > 1:
                    t = 1.0
                    s = np.clip((b - c) / a, 0, 1)

        closest_on_1 = p1 + s * d1
        closest_on_2 = p3 + t * d2
        return np.linalg.norm(closest_on_1 - closest_on_2)


    def check_self_collision(self, joint_positions, link_radii, skip_adjacent=True):
        """
        joint_positions: list of 3D points [j0, j1, j2, j3, j4, j5, j6] j_i = (p_i, p_i+1)
        link_radii: radius of each link capsule
        """
        n_links = len(joint_positions) - 1
        for i in range(n_links):
            for j in range(i + 2, n_links):  # skip i+1 (adjacent)
                dist = self.closest_point_segment_segment(
                    joint_positions[i], joint_positions[i+1],
                    joint_positions[j], joint_positions[j+1]
                )
                if dist < link_radii[i] + link_radii[j]:
                    return True  # collision detected
        return False
    

    def do_fk_and_check_crash(self):
        '''
        move the joints to their new positions
        update the list of joint positions wrt world frame
        check if the robot crashes into itself
        '''
        joint_pos = self.give_ds()
        crash = self.check_self_collision(joint_positions= joint_pos,
                                          link_radii= self.fail_dist_vect)
        
        return crash













def test_making_jacobian_logic():
    H0 = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])
    
    o = H0[0:3, 3] # x, y, z
    z = H0[0:3, 2] # r31, r32, r33

    print(f'o = {o}')
    print(f'z = {z}')

    print('try crossing two perpenidcular vectors')
    a = [1, 0, 0]
    b = [0, 1, 0]
    print(f'a {a} x b {b} = {np.cross(a, b)}')

    print('try stacking two vectors as columns of a matrix')
    c = [1, 2, 3, 4, 5, 6]
    d = [7, 8, 9, 10, 11, 12]
    print(f'c = {c}')
    print(f'd = {d}')
    print(f'c and d column stacked = {np.column_stack(c, d)}')

            


def test_Robot_Math():
    a = [500, 500, 0]
    alpha = [0, 0, 0]
    d = [400, 0, 0]
    theta = [0, 0, 0]
    joint_types = ['r', 'r', 'p']
    # joint_bounds = [(np.deg2rad(-90), np)]


    R = Robot_math(a= a,
                   alpha= alpha,
                   d= d,
                   theta= theta)
    
    R.joint_type = joint_types

    R.q_vect = [np.deg2rad(0), np.deg2rad(0), 0]
    R.FK()
    # print(R.curr_joint_H)
    for p in R.curr_joint_H:
        print(p)

    R.J()
    # print(R.curr_jacobian)
    for j in R.curr_jacobian:
        print(f'jacobian: \n{j}')



                    




if __name__ == "__main__":
    # test_making_jacobian_logic()
    test_Robot_Math()
