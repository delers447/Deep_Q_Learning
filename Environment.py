import numpy as np
#import math as m
from matplotlib import pyplot as plt
from Car import CarObj
from PIL import Image

def TRACK_ROW():
    return 720

def TRACK_COL():
    return 1200

class EnvironmentClass:
    def __init__(self, track_file, start_position, orientation, racers_names):
        # Matrix for the track
        self.racers_names = racers_names# arrays of racers' names
        self.TRACK_MAT = RACE_TRACK(track_file)
        self.__carlist = []
        self.__cars_adjust(orientation)  # orient the cars'facing directions
        #self.__all_cars_start_pos = start_position.copy()
        self.__all_cars_start_pos = start_position
        self.__load_cars()  # loads the cars into their starting positions and stores in carlist

    def track_displayer(self):
        plt.imshow(np.transpose(self.TRACK_MAT), interpolation='nearest')
        plt.show()

    def NUM_OF_DRIVERS(self):
        return len(self.racers_names)

    def __load_cars(self):
        for i in range(0, self.NUM_OF_DRIVERS()):  # initialize the drivers car starting positions
            __car_obj = CarObj(self.racers_names[i], self.__all_cars_start_pos, self.orient_angle)
            self.__carlist.append(__car_obj)

    def reset_cars(self):
        for index in range(len(self.__carlist)):
            self.__carlist[index].reset()
        """
        for i in range(0, self.NUM_OF_DRIVERS()):
            self.__carlist[i].front_bumper_pos = [620, 150]
            self.__carlist[i].__speed = 1
            self.__carlist[i].__accelerate = 10
            #self.__carlist[i].front_bumper_pos = self.__all_cars_start_pos
            #print(F"The new start position is {self.__carlist[i].front_bumper_pos}.")
            self.__carlist[i].car_theta = self.orient_angle
            #self.__carlist[i].tires_history = []
            #self.__carlist[i].front_bumper_history = []
            self.__carlist[i].update_Carposition()
            self.__carlist[i].epsilon = 1.0
        """
    def __cars_adjust(self, orientation):
        orientation = orientation.upper()

        if orientation == "NORTH":
            self.orient_angle = 0

        elif orientation == "WEST":
           self.orient_angle = 90

        elif orientation == "SOUTH":
           self.orient_angle = 180

        elif orientation == "EAST":
           self.orient_angle = 270

    def train(self, car_index=1):
        score = 0
        reward_hist = []
        self.reset_cars()
        current_state = self.get_input_distances(car_index=car_index)

        while not self.disqualification_check(car_index=car_index):
            action_vector, decision = self.__carlist[car_index].explore(current_state)
            self.__carlist[car_index].drive(decision)
            new_state = self.get_input_distances(car_index=car_index)
            reward = self.reward(car_index=car_index)
            reward_hist.append(reward)
            score += reward
            done = self.disqualification_check(car_index=car_index)
            self.__carlist[car_index].remember(current_state, action_vector, reward, new_state, done)
            self.__carlist[car_index].learn()
        epsilon =  self.__carlist[car_index].epsilon
        #print(F"Reward Hitory: {reward_hist}")
        return score, epsilon, reward_hist

    def get_model(self, car_index=1):
        self.__carlist[car_index].get_model()

    def set_model(self, this_model, car_index=1):
        self.__carlist[car_index].set_model(this_model)

    def save_model(self, filename, car_index=1):
        self.__carlist[car_index].save_model(F"{filename}")

    def purge_car_histories(self):
        for i in range(0, self.NUM_OF_DRIVERS()):  
            self.__carlist[i].purge_history()

    def load_model(self, filename, car_index=1):
        self.__carlist[car_index].load_model(F"{filename}")

    def race(self):
        for index, car in enumerate(self.__carlist):
            #print(F"The disqualification_check reported: {self.disqualification_check()}.")
            while not self.disqualification_check(car_index=index):
                print(F"The car's position: {car.front_bumper_pos} with state: {self.get_input_distances(car_index=index)}.")
                state = self.get_input_distances(car_index=index)
                action = car.make_decision(state)
                #print(F"State: {state}")
                print(f"The legnth of the car history is {len(car.front_bumper_history)}")
                car.drive(action)
                print(f"The legnth of the car history is {len(car.front_bumper_history)}")
            
            directions = [(-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0),(0,0)]
            for car_x, car_y in car.front_bumper_history:
                print(f"car_x:{car_x} , car_y:{car_y} ")
                for direction in directions:
                    dx, dy = direction
                    self.TRACK_MAT[int(car_x+dx), int(car_y+dy)] = 10
            self.track_displayer()


    def get_cars(self):
        return self.__carlist

    def car_get(self, i):
        result = self.__carlist[i]
        return result

    def reward(self, car_index=0):
        car = self.__carlist[car_index]
        reward = 0
        x, y = car.front_bumper_history[-1][0], car.front_bumper_history[-1][1]
        

        #if car.get_speed() < 1:
        #    #The car stopped.
        #    return -100
        if x > TRACK_COL() or y > TRACK_ROW():
            #The car went off the map.
            return -100
        if self.TRACK_MAT[int(x), int(y)] == 0:
            #print("The spot was in the grass.")
            return -100

        hist_length = len(car.front_bumper_history)
        reward = len(car.front_bumper_history)*10

        if hist_length > 1:
            present_fbumper = car.front_bumper_history[- 1]
            past_fbumper = car.front_bumper_history[-2]
            try:
                slope = (present_fbumper[1] - past_fbumper[1])/(present_fbumper[0] - past_fbumper[0])
            except:
                #cannot caculuate the slope.
                return -10

            for x in range(int(past_fbumper[0]), int(present_fbumper[0]) + 1):
                fx = int(slope*(x-past_fbumper[0]) + past_fbumper[1])
                #print("[x fx] = [", x, " ", fx," ", self.TRACK_MAT[x, fx],"]")
                if self.TRACK_MAT[x][fx] == 2:
                    reward = 1_000 + len(car.front_bumper_history)*20

        return reward

    def get_input_distances(self, car_index=0):
        car = self.__carlist[car_index]
        x, y = car.front_bumper_pos
        #print(F"=====> To fight a bug: This is the (x,y): ({x},{y})")
        return self.__get_distance_to_boundary(x, y)

    def __get_distance_to_boundary(self, real_x, real_y):
        distances = []

        # Define directions (north, 45 degrees northeast, east, etc.)
        #               N:0       NW:1      W:2    SW:3     S:4     SE:5     E:6     NE:7
        directions = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]

        # Iterate through each direction
        for direction in directions:
            #print(direction)
            x, y = real_x, real_y
            dx, dy = direction
            distance = 0
            if x > TRACK_COL() or y > TRACK_ROW():
                    break

            # Move in the current direction until reaching the track boundary
            while self.TRACK_MAT[int(x), int(y)] != 0:  # Assuming 0 represents the grass
                x += dx
                y += dy
                distance += 1
                if x > TRACK_COL() or y > TRACK_ROW():
                    break

            distances.append(distance)
        
        return distances

    def disqualification_check(self, car_index=0):
        car = self.__carlist[car_index]
        #if car.get_speed() == 0:
        #    return True
        # Check if any tire of any car is at the track boundary (pixel value of 0 or 1)
        #tires_coordinates = car.tires_history[-1]
        tires_coordinates = car.get_tires()

        #for tire in tires_coordinates:
        #    x, y = tire
        #    print(F"\t The tire's turf at ({x},{y}) is {self.TRACK_MAT[int(x)][int(y)]}")
        for tire in tires_coordinates:
            x, y = tire
            #if x > TRACK_COL() or y > TRACK_ROW():
            #        print("====> Went off the track!")
            #        return True
                    #pass
            #print(F"\t The tire's turf at ({x},{y}) is {self.TRACK_MAT[int(x)][int(y)]}")
            if self.TRACK_MAT[int(x)][int(y)] == 0:
                return True  # Car is off track.
        return False  # All cars are on the track.
    # Add code for crossing the finish line.

def RACE_TRACK(filename):
    # Edited by Dan.
    image = Image.open(filename)
    data = np.asarray(image)
    print(F"The shape of the image is {data.shape}")
    track = np.zeros(TRACK_COL() * TRACK_ROW()).reshape(TRACK_ROW(), TRACK_COL())

    for x in range(TRACK_COL()):
        for y in range(TRACK_ROW()):
            total = sum(data[y, x])
            r, g, b = data[y, x]
            if total < 90:
                # Asphalt represented by 1
                track[y, x] = 1
            elif total > 90 and total < 350 and (r > (g + b)):
                # Gates represented by 2
                track[y, x] = 2
            elif total > 250 and total < 800:
                # Grass reprsented by 0
                track[y, x] = 0

    # Fix a few single cells that where not asphalt when all four sides were.
    for x in range(1, TRACK_COL() - 1):
        for y in range(1, TRACK_ROW() - 1):
            if track[y + 1, x] == 1 and track[y, x + 1] == 1 and track[y - 1, x] == 1 and track[y, x - 1] == 1:
                track[y, x] = 1
    return np.transpose(track)

#def PIXEL_LENGTH():
#    return 50






