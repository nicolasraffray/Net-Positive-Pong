import csv
import pickle
import random
from datetime import datetime
from pathlib import Path

import cv2
import gym
import numpy as np

render = False
benchmark = False

batch_size = 5
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99
dimension = 80 * 70

# resume from previous checkpoint?

my_file = Path("./episode_file.csv")
""" for the first game played there is no history 
Can we rename resume, isnt clear to what it is doing - seperation of concerns"""
if my_file.is_file():
  print('Resuming run')
  resume = True
else:
  print('First run')
  resume = False

def import_csv(csvfilename):
  data = []
  row_index = 0
  with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
    reader = csv.reader(scraped, delimiter=',')
    for row in reader:
      data.append(row[0])
  return data

if resume:
  data = import_csv('episode_file.csv')
  episode_number = int(data[0])
else:
  episode_number = 0


if resume:
  model = pickle.load(open('save.p', 'rb'))
  #takes 10-15 ms on macbook pro
else:
  model = {}
  if benchmark:
    np.random.seed(5) ; model['W1'] = np.random.randn(200,dimension)/np.sqrt(dimension)
    np.random.seed(5) ; model['W2'] = np.random.randn(150,200)/np.sqrt(200)
    np.random.seed(5) ; model['W3'] = np.random.randn(100,150)/np.sqrt(150)
    np.random.seed(5) ; model['W4'] = np.random.randn(100)/np.sqrt(100)
  else:
    model['W1'] = np.random.randn(200,dimension)/np.sqrt(dimension)
    model['W2'] = np.random.randn(150,200)/np.sqrt(200)
    model['W3'] = np.random.randn(100,150)/np.sqrt(150)
    model['W4'] = np.random.randn(100)/np.sqrt(100)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
#updated .iteritems to .items
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } 


# below function used for balck and white game
def pre_process_image(frame): # function for when we use the main pong on server
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  frame = cv2.cvtColor(cv2.resize(frame,(80,80)), cv2.COLOR_BGR2GRAY)

  ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY) # et is useless
  frame[frame == 255] = 1
  frame = frame.ravel()
  return frame

def prepro(I): # function for openai gym image conversion taken from a Deeplearning site
  """ prepro 210x160x3 uint8 frame into 6000 (80x80) 1D float vector """
  I = I[34:194] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
  I = I[:,20:]
  cv2.imwrite("image.png", I)
  I = I[::2,::2,0] # downsample by factor of 2.
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
  return I.astype(np.float).ravel()

def relu(Z): 
  return np.maximum(0.0,Z)

def forward_prop(input_array, weights_dict): 
  Z1 = np.dot(weights_dict['W1'], input_array)
  A1 = relu(Z1)
  Z2 = np.dot(weights_dict['W2'], A1) 
  A2 = relu(Z2)
  Z3 = np.dot(weights_dict['W3'], A2)
  A3 = relu(Z3)
  Z4 = np.dot(weights_dict['W4'], A3)
  A4 = 1.0 / (1.0 + np.exp(-Z4))
  forward_output = {"X": input_array, "Z1": Z1, "A1": A1,
   "Z2": Z2, "A2": A2,"Z3":Z3,"A3": A3, "Z4": Z4, "A4": A4}
  return forward_output

def make_move(A3):
  if A3 > 0.975:
    action = 2 
  elif A3 < 0.025:
    action = 3
  elif A3 > np.random.uniform():
    action = 2
  else:
    action = 3 
  return action 

def true_y(action): 
  ''' function for randomly generating true y ''' 
  if action == 2:
    y = 1 
  else:
    y = 0 
  return y 

# states from forward prop need to be saved for backprop
def compute_grad(y,A3): 
  loss_grad.append(y - A3) # this value needs to be appended into an array

def Relu_derivative(Z):
    Z[Z > 0 ] = 1
    Z[Z <= 0] = 0 
    return Z

def back_prop(ep_input, ep_Z1, ep_A1, ep_Z2, ep_A2, ep_Z3, ep_A3,ep_end_grad):
  dW4 = np.dot(ep_A3.T, ep_end_grad).ravel()

  dC_dA3 = np.outer(ep_end_grad, model['W4'])
  dA3_dZ3 = Relu_derivative(ep_Z3)
  dC_dZ3 = dC_dA3 * dA3_dZ3

  dW3 =  np.dot(dC_dZ3.T, ep_A2)
 
  dC_dZ3 = np.sum(dC_dZ3, axis = 0, keepdims = True)
  dC_dA2 = np.dot(dC_dZ3, model['W3'])
  dA2_dZ2 = Relu_derivative(ep_Z2)
  dC_dZ2 = dC_dA2 * dA2_dZ2

  dW2 = np.dot(dC_dZ2.T, ep_A1)

  dC_dZ2 = np.sum(dC_dZ2, axis = 0, keepdims = True)
  dC_dA1 = np.dot(dC_dZ2, model['W2'])
  dA1_dZ1 = Relu_derivative(ep_Z1)
  dC_dZ1 = dC_dA1 * dA1_dZ1

  dW1 = np.dot(dC_dZ1.T, ep_input)

  derivatives = {}
  derivatives['W1'] = dW1
  derivatives['W2'] = dW2
  derivatives['W3'] = dW3
  derivatives['W4'] = dW4

  return derivatives


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  discounted_r
  # standardizing
  discounted_r -= np.mean(discounted_r)
  discounted_r /= np.std(discounted_r)
  return discounted_r


env = gym.make("Pong-v0")
observation = env.reset()
prev_frame = None
state, z1, h1, z2, h2, z3, h3, z4, loss_grad, r = [], [], [], [], [], [], [], [], [], []
running_reward = None
reward_sum = 0 
cumulative_batch_rewards = 0

while True:
  if render: env.render()

  # cv2.imwrite('color_img.jpg', observation1)
  frame = prepro(observation)
  d_frame = frame - prev_frame if prev_frame is not None else np.zeros(dimension)
  prev_frame = frame

  net_vals = forward_prop(d_frame, model)

  state.append(d_frame)
  z1.append(net_vals["Z1"])
  h1.append(net_vals["A1"])
  z2.append(net_vals["Z2"])
  h2.append(net_vals["A2"])
  z3.append(net_vals["Z3"])
  h3.append(net_vals["A3"])

  action = make_move(net_vals["A4"])
  y = true_y(action)
  loss_grad.append(y - net_vals["A4"])

  observation,reward,done,info = env.step(action)
  r.append(reward)
  reward_sum += reward


  if done: 
    episode_number += 1 

    episode_input = np.vstack(state)
    episode_z1 = np.vstack(z1)
    episode_h1 = np.vstack(h1)
    episode_z2 = np.vstack(z2)
    episode_h2 = np.vstack(h2)
    episode_z3 = np.vstack(z3)
    episode_h3 = np.vstack(h3)
    

 


    episode_loss_grad = np.vstack(loss_grad)
    episode_reward = np.vstack(r)

    state, z1, h1, z2, h2, z3, h3, z4, loss_grad, r = [], [], [], [], [], [], [], [], [], []

    discounted_ep_rewards = discount_rewards(episode_reward)

    episode_loss_grad *= discounted_ep_rewards
    grad = back_prop(episode_input, episode_z1 ,episode_h1, episode_z2, episode_h2, episode_z3, episode_h3, episode_loss_grad)
    
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over the batch
    if episode_number % batch_size == 0:
      for k,v in model.items():
        #updated .iteritems to .items to work with python3
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v)

    # === Documenting Performance and resetting below
    if episode_number % batch_size == 1:
      cumulative_batch_rewards = reward_sum
      batch_average = reward_sum
    elif episode_number % batch_size == 0:
      cumulative_batch_rewards += reward_sum
      batch_average = cumulative_batch_rewards/batch_size
    else:
      cumulative_batch_rewards += reward_sum
      batch_average = cumulative_batch_rewards/(episode_number % batch_size)

    #print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    #removed print for performance purposes
    
    if episode_number % 100 == 0: 
      pickle.dump(model, open('save.p', 'wb'))
      #takes 15-20ms on macbook pro
    if episode_number % batch_size == 0: 
      with open('episode_file.csv', mode='w') as episode_file: #store the last episode
        episode_writer = csv.writer(episode_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        episode_writer.writerow([episode_number])
      with open('performance_file.csv', mode='a') as performance_file: #track performance over time
        performance_writer = csv.writer(performance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        performance_writer.writerow([datetime.now(), episode_number, batch_average])

    reward_sum = 0 
    observation = env.reset() # reset env
    prev_x = None
            
env.close()
