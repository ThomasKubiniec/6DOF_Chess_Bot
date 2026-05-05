'''
The goal is to make approximately 100k total samples of good and okay frames (trajectory points), 
discarding bad or crashing frames.

Good points have a target weight of 1.0. Okay has a target weight of 0.3. 
Loss is multiplied by the target_weight. 



Since this is a medium sized dataset, it will take a long
time to compute, so we will need to save it. 

The format of the dataset:
(q_curr, delta_q_prev, Goal_XYZ, Goal_Ori_SO3, delta_q_pred, target_weight)
'''