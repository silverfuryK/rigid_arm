from model_arm_data import *
import time
import multiprocessing
from tqdm import tqdm

models = []

num_models = 16

# for _ in range(num_models):
arm = ArmEnv(datafile="dataset/data1.json", render_mode=None)
    # models.append(arm)

total_timesteps = 7e6

# joint limits
a1 = [-3.14159, 3.14159]
a2 = [-2.40855, 2.40855]
a3 = [-2.65988, 2.65988]
a4 = [-3.14159, 3.14159]
a5 = [-2.23402, 2.23402]
a6 = [-3.14159, 3.14159]

# sample random joint angles

# def generate_random_actuator_positions(_):

q = [
        np.random.uniform(a1[0], a1[1]), 
        np.random.uniform(a2[0], a2[1]), 
        np.random.uniform(a3[0], a3[1]), 
        np.random.uniform(a4[0], a4[1]), 
        np.random.uniform(a5[0], a5[1]), 
        np.random.uniform(a6[0], a6[1]),
        0,0,0,0,0,0
        ]
i = 0
done = False
# while i<total_timesteps:
for i in tqdm(range(int(total_timesteps))):
    d = False
    
    if done:
        done = False
        # print("done")
        arm.reset_model()
        q = [
        np.random.uniform(a1[0], a1[1]), 
        np.random.uniform(a2[0], a2[1]), 
        np.random.uniform(a3[0], a3[1]), 
        np.random.uniform(a4[0], a4[1]), 
        np.random.uniform(a5[0], a5[1]), 
        np.random.uniform(a6[0], a6[1]),
        0,0,0,0,0,0
        ]
        # print('iteration:',i, '\r')
        # time.sleep(0.1)
    else:
        obs, _,_,done,_ = arm.step(q)
    i+=1
