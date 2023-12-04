
#def __func() denotes private function anything else is public. Same for variables.
# e.g. __x is private

#Asuzu's Imports
import numpy as np
import math as m

#Dan's Imports
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Reshape

def WIDTH():
    carWidth = 30
    return carWidth

def LENGTH():
    carLength = 50
    return carLength

def RADIANS(deg):
    deg = deg % 360  # convert car value in degrees.
    theta = deg * m.pi/ 180.0
    return theta

def THETA_INCREMENT(): # the value by the car turn if it is to turn. 20 is preferred.
    return 20.0

def MAX_SPEED():
    return 200 # 50 is standard max car speed.

def TIME(): # the time it takes to reach the full car speed.
    return 20

# Ideally a precise set of driving instruction per frame.
actions_1 =np.array([
                            ["Straight", "Accelerate"], 
                            ["Straight", "Accelerate"],
                            ["Straight", "Accelerate"], 
                            ["Straight", "Accelerate"],
                            ["Right", "Brake"], 
                            ["Right", "Brake"],
                            ["Straight", "Accelerate"],
                            ["Straight", "Coast"],
                            ["Straight", "Brake"],
                            ["Right", "Brake"],
                            ["Right", "Brake"], 
                            ["Straight", "Accelerate"]
                            ])
class CarObj:

    # tire positions on car
    #     FRONT?
    # t[0]----t[1]
    # |         |
    # |  Car    |
    # |  Shape  |
    # |         |
    # t[2]----t[3]
    #     BACK?

    __center = np.zeros(2).reshape(1, 2) # center of the car [x, y] position size: [1x2].
    __tires = np.zeros(8).reshape(4, 2) # list of tires [x, y] positions size: [4x1x2].
    front_bumper_pos = [] # front bumper [x, y] size: [1x2].
    __car_start_pos = None #initial [x, y] starting position of the car.
    tires_history = []
    front_bumper_history = []

    def __init__(self, racer_name, front_bumper_pos, theta):  # Constructor
        self.original_pos = front_bumper_pos.copy
        self.original_theta = theta
        self.racer_name = racer_name # driver's name
        self.front_bumper_pos = front_bumper_pos.copy()    
        self.car_theta = theta  # car theta is the angle of turn.
        self.__speed = 0   # speed of the car, with a starting value of Acceleration
        self.__accelerate = 30 # car's acceleratino rate.
        self.__jerk = 10 # car's jerk rate. jerk is the rate of change of the acceleration.
        self.__decelerate = 0 # car's deceleration rate.
        self.__djerk = 0.01 # car's deceleration rate
        self.update_Carposition() # update the car's center and tires.

        # returns the start position of the front tires at the start of the race.
        self.__car_start_pos = self.front_bumper_pos.copy()  


        #Dan's Additions for Learning.
        self.action_space = [
                            [[1,0,0],[1,0,0]], [[1,0,0],[0,1,0]], [[1,0,0],[0,0,1]],
                            [[0,1,0],[1,0,0]], [[0,1,0],[0,1,0]], [[0,1,0],[0,0,1]],
                            [[0,0,1],[1,0,0]], [[0,0,1],[0,1,0]], [[0,0,1],[0,0,1]]
                            ]
        #Hyper-parameters for the "Deep Q Learning"
        self.gamma= 0.99            #Scale of "Future Rewards" in the Q-equation in learn()
        self.epsilon = 1.0          #Exploration % chance.  Starts high and lowers.
        self.epsilon_dec = 0.99     #Decay factor of epsilon
        self.epsilon_min = 0.10     #Floor of epsilon
        self.batch_size = 64        #Sample size of Memory during learn()
        self.memory = ReplayBuffer(1_000_000, 8, 12, discrete=True)
        #Randomly Seeded Neural Network
        self.create_new_model() 

    def reset(self, verbose=True):
        self.front_bumper_pos = self.original_pos
        self.car_theta = self.original_theta
        self.__speed = 1   # speed of the car, with a starting value of Acceleration
        self.__accelerate = 30 # car's acceleratino rate.
        self.__jerk = 10 # car's jerk rate. jerk is the rate of change of the acceleration.
        self.__decelerate = 0 # car's deceleration rate.
        self.__djerk = 0.01 # car's deceleration rate
        self.epsilon = 1.0 
        print("Front Bumper History before reset: ", self.front_bumper_history)
        self.tires_history = []
        self.front_bumper_history = []
        self.update_Carposition() # update the car's center and tires.
        if verbose:
            print(F"The current status of the car is: ")
            print(F"\t front_bumper_pos: {self.front_bumper_pos}")
            print(F"\t original_pos: {self.original_pos}")
            print(F"\t car_theta: {self.car_theta}")
            print(F"\t __speed: {self.__speed}")
            print(F"\t __accelerate: {self.__accelerate}")
            print(F"\t tires_history: {len(self.tires_history)}")
            print(F"\t front_bumper_history: {len(self.front_bumper_history)} : {self.front_bumper_history}")

    def update_car_history(self):
        self.front_bumper_history.append(self.front_bumper_pos.copy())
        self.tires_history.append([self.__tires[0].copy(), self.__tires[1].copy(),
                                       self.__tires[2].copy(), self.__tires[3].copy()])
        
        #self.front_bumper_history.append([self.front_bumper_pos[0],self.front_bumper_pos[1]] )
        #self.tires_history.append(self.__tires)

    def create_new_model(self):
        """
        Added by Dan.  
        Create a new randomly seeded model.
        """

        model = Sequential()

        #Input Layer.  
        model.add(Input(shape=(8, )))

        #First Hidden Layer
        model.add(Dense(64, activation='sigmoid'))

        #Second Hidden Layer
        model.add(Dense(64, activation='sigmoid'))

        #Output Layer
        model.add(Dense(2*3))
        model.add(Reshape((-1, 3)))
        model.add(Activation('softmax'))

        #Package the model
        model.compile(
                loss='mse', 
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        model.summary()

        self.model = model

    def save_model(self, name):
        """
        Written by Dan
        Save the model using a name. Will save in the current working directory.
        """
        print(F"Saving the model: {name}.model ...")
        try:
            self.model.save(F"{name}.model")
            print(F"The model saved successfully.")
        except:
            print(F"ERROR: The model did not save.")
            assert False

    def load_model(self, name):
        """
        written by Dan.
        Load the name of the model.  THe model should be in the current working directory.
        {name}.model
        """
        print(F"Loading the model: {name}.model ...")
        try:
            self.model = tf.keras.models.load_model(F"{name}.model")
            print(F"The model loaded successfully.")
        except:
            print(F"ERROR: The model did not load.")
            assert False

    def make_decision(self, state):
        """
        Written by Dan.  
        state is a tuple of 8 values:
        Distances in the following directions:
            N, NE, E, SE, S, SW, W, NW
        Car will pick the best action based on state.
        100% explotation with ZERO exploration.  ie, not Explore()
        
        Output is tuple of strings, like: ['Right', 'Accelerate']
        """
        print(F"The model is making a decision based on {state}")
        outputs = self.model.predict([state])
        outputs = outputs[0]
        decision = self._interprete_output(outputs)
        return decision

    def _interprete_output(self, outputs):
        decision = []
        first_max = max(outputs[0])
        if outputs[0][0] == first_max:
            decision.append("Left")
        elif outputs[0][1] == first_max:
            decision.append("Straight")
        elif outputs[0][2] == first_max:
            decision.append("Right")
        else:
            decision.append("ERROR")

        second_max = max(outputs[1])
        if outputs[1][0] == second_max:
            decision.append("Accelerate")
        elif outputs[1][1] == second_max:
            decision.append("Coast")
        elif outputs[1][2] == second_max:
            decision.append("Brake")
        else:
            decision.append("ERROR")

        print(F"The decision is: {decision}.")
        return decision
        
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def explore(self, state):
        """
        Written by Dan.
        Car picks the next action with threshold of random choices to factiltate exploration.
        make_decision is 0% exploration and 100% exploitation while
        Explore() is Epislon% Exploration and (100-Epsilon)% exploitation.

        Output is a tuple of two elements:
            Element 1: 1 hot matrix, like [[1,0,0],[0,1,0]]
            Element 2: decisions tupel of strings like ['Right', 'Accelerate']
        """
        #state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            rand_num = random.randint(0,len(self.action_space)-1)
            print(F"The random number was: {rand_num} when the length of the list is: {len(self.action_space)}")
            action = self.action_space[rand_num]
            #action = np.random.choice(self.action_space)
        else: 
            outcome = self.model.predict(state)
            action = [[0,0,0],[0,0,0]]
            action[0][np.argmax(outcome[0])] = 1
            action[1][np.argmax(outcome[1])] = 1
        
        decision = self._interprete_output(action)

        return action, decision

    def learn(self):
        """
        Written by Dan.
        Car will take a sample of the Memory and then fit the model to be better.
        Ie, MAGIC!
        """
        if self.memory.mem_cntr > self.batch_size: #Not enough data to sample.
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            
            
            #print(f"action: {action}")
            #print(f"action_values: {action_values}")
            reshaped_action_values= []
            for matrix in action_values:
                reshaped_action_values.append(matrix.transpose())
            #print(np.array(action).shape)
            action_indices = np.dot(action, reshaped_action_values).reshape((1, 36, 64)) #Dan thinks this line will break.
            
            #action_indices = np.dot(action, action_values) #Dan thinks this line will break.

            #print(state.shape)
            q_eval = self.model.predict([state])

            q_next = self.model.predict(new_state)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            print(F"Shapes of: reward:{reward.shape},action_indices:{action_indices.shape}, q_eval:{q_eval.shape}, q_next:{q_next.shape}, q_target:{q_target.shape}, batch_index:{batch_index.shape}")

            q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done
            #This is where Dan thought it would break.  [[0,1,0],[0, 0,1]] is not a one-hot vector 
            #for the dotproduct to give the indices of the actions for this 2D matrix.

            #Train the model.
            _self.model.fit(state, q_target, verbose=0)

            #Decrease the Exploration % chance until it reaches floor.
            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
            self.epsilon_min else self.epsilon_min


    def update_Carposition(self): # calculate and update the car's center and tires.
        #[0][0][Ft1]----------[0][1][Ft2]
        #|                              |
        #|                              |
        #|                              |
        #|                              |
        #[1][0][Bt3]----------[1][1][Bt4]

        fx, fy = self.front_bumper_pos[0], self.front_bumper_pos[1] # position of the front bumper.

        #obtaining the XY coordinates of the tires using the front bumper XY position.
        tire1_x = fx - 0.5 * WIDTH() * m.cos(RADIANS(self.car_theta)) # tire1x
        tire1_y = fy + 0.5 * WIDTH() * m.sin(RADIANS(self.car_theta)) # tire1y

        tire2_x = fx + 0.5 * WIDTH() * m.cos(RADIANS(self.car_theta))  # tire2x
        tire2_y = fy - 0.5 * WIDTH() * m.sin(RADIANS(self.car_theta))  # tire2y

        tire4_x = tire2_x - LENGTH() * m.cos(RADIANS(self.car_theta + 90))  # tire4x
        tire4_y = tire2_y - LENGTH() * m.sin(RADIANS(-(self.car_theta + 90)))  # tire4y

        tire3_x = tire4_x + WIDTH() * m.cos(RADIANS(self.car_theta + 180))  # tire3x
        tire3_y = tire4_y - WIDTH() * m.sin(RADIANS(self.car_theta + 180))  # tire3y

        centerX = fx + 0.5 * LENGTH() * m.cos(RADIANS(self.car_theta - 90)) # car_center xpos
        centerY = fy + 0.5 * LENGTH() * m.sin(RADIANS(self.car_theta + 90)) # car_center ypos


        self.__center = [centerX, centerY]

        # Updating the tire list
        self.__tires[0] = [tire1_x, tire1_y] #tire1
        self.__tires[1] = [tire2_x, tire2_y] #tire2
        self.__tires[2] = [tire3_x, tire3_y] #tire3
        self.__tires[3] = [tire4_x, tire4_y] #tire4

        #print("Updaing Car History.")
        self.update_car_history()
        #print(F"The front bumper history: {self.front_bumper_history}")
        #print(F"The tire history: {self.tires_history}")

    def get_tires(self):
        return self.__tires

    def __calcCar_motion(self, turn): # calculate and update the car's turn and orientation
        if self.__speed <= 0: # If car has been decelerated below zero to a negative value.
            self.__speed = 0

        # car_theta + 90 to face the car upright
        # Calculate the car's new position after turn car_theta + 90

        self.front_bumper_pos[0] = self.__car_start_pos[0] + (self.__speed * m.cos(RADIANS(self.car_theta + 90)))
        self.front_bumper_pos[1] = self.__car_start_pos[1] - (self.__speed * m.sin(RADIANS(self.car_theta + 90)))

        # if no car turn was choosen, don't increase car turn angle

        if turn == 0:
            theta_increment = 0.0
        else:
            theta_increment = THETA_INCREMENT()


        if turn == -1: # turn left
            self.car_theta = self.car_theta + theta_increment

        elif turn == 1: # turn right
            self.car_theta = self.car_theta - theta_increment

        #Set the new position that the car moved to as the car starting position for the new draw.
        self.__car_start_pos[0] = self.front_bumper_pos[0]
        self.__car_start_pos[1] = self.front_bumper_pos[1]


    def __calc_speed(self, accel_or_brake): # Calculates by how much the car should accelerate or move.
        if accel_or_brake == 1: #if accelerate is picked
            if self.__speed < MAX_SPEED(): # if max speed is not yet reached
                #increase car's accleration and speed and reset decelerate to its default value.
                self.__decelerate = self.__djerk
                self.__accelerate += self.__jerk
                self.__speed = self.__speed + self.__accelerate * (TIME()/ 20)



        elif accel_or_brake == -1: # if brake is picked.
            # if car speed is still above the value 0
            if self.__speed > 0:
                self.__speed = self.__speed - self.__decelerate * (TIME()/ 20)
                self.__decelerate += self.__djerk # if brake is pressed multiple times.


        elif accel_or_brake == 0: # if maintain the same velocity or speed is chosen -> coast.
            self.__decelerate = self.__djerk
            self.__speed = self.__speed + self.__accelerate * (TIME()/ 20) # coast

        # if the current car speed goes over the max car speed
        if self.__speed > MAX_SPEED():
            self.__speed = MAX_SPEED()




    def __interprete_NN_decision(self, do_these): # Interpret Neural network decisions.
                                        # Actions array elements are set to mimic possible Neural
                                        # network decisions. NN stands for Neural Network.

        left, straight, right = "Left", "Straight", "Right"
        accelerate, coast, brake = "Accelerate", "Coast", "Brake" # coast is equal to keeping the same
                                                                  # velocity. or not accelerating.

        turn = 0 # -1, 0, 1 means car is to turn left, stay straight or turn right respectively.

        if do_these[0] == left:
            turn = -1

        elif do_these[0] == straight:
            turn = 0

        elif do_these[0] == right:
            turn = 1


        if do_these[1] == accelerate:
            self.__calc_speed(1)

        elif do_these[1] == coast:
            self.__calc_speed(0)

        elif do_these[1] == brake:
            self.__calc_speed(-1)
            if self.__speed == 0: # if car speed is 0. Stop turning and set the car straight.
                turn = 0

        # Move the car.
        self.__calcCar_motion(turn)
        self.update_Carposition()

    def displayCar_Info(self): #Displays all the car's public attributes.
        #print(F'Car name = {self.racer}')

        print(F'Car_theta = {self.car_theta}')
        print(F'Car FBumper ={self.front_bumper_pos}')

        #print(F'Car center = {self.__center}')

        print(F'Car Accelerate = {self.__accelerate}')
        print(F'Car Speed = {self.__speed}')



    def drive(self, element):

        #for i, element in enumerate(actions_1):
        print("element = ", element)
        self.__interprete_NN_decision(element)
        self.displayCar_Info()

    def get_speed(self):
        return self.__speed

    #def NN_interim(self, radar_info):
     #   if (m.absradar_info[1])== :


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        #Reference: https://www.youtube.com/watch?v=5fHngyN8Qhw
        #Referemce: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_keras.py
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        #self.action_memory = np.zeros((self.mem_size, n_actions, ), dtype=dtype)
        self.action_memory = np.zeros((self.mem_size, 2, 3), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        self.mem_cntr += 1
        index = self.mem_cntr % self.mem_size
        print(F"The shape of the state buffer is {self.state_memory[index].shape} and the shape of the state is {state}.")
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        #Store one hot encoding of actions, if appropiate.
        #Dan Change this latter to [[0,0,1],[1,0,0]] (ie, 2x3 one of zeroes)
        #if self.discrete:  
        #    actions = np.zeros(self.action_memory.shape[1])   
        #    actions[action] = 1.0
        #    self.action_memory[index] = actions
        #else:
            #self.action_memory[index] = action
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory = 1 - done
        

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        print(batch)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


if __name__ == "__main__":
    
    #Example of the code to work with Neural network.
    print(F"The program started successfully.")
    Dan_Car = CarObj("Dan", [200, 30], 90)
    print(F"The car object has been made.")
    Dan_Car.create_new_model()
    #Dan_Car.load_model("example")
    print(F"A new model has been made.")
    print(F"Making a decision ...")
    Dan_Car.make_decision((10, 10, 10, 10, 10, 10, 10, 10)) #distance of 10 in all 8 directions.
    print(F"Saving the model ...")
    Dan_Car.save_model("example")
    print(F"The model saved.")
