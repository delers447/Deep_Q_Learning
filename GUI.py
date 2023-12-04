import pygame
import sys
#import numpy as np
import threading
import math
class Frame:
    def __init__(self, screen):
        self.screen=screen
        self._online=False
        self._buttonwidth=100
        self._buttonheight=50
        self._buttons=[]
        self._buttontext=[]
        self.__desired_width=1200
        self.__desired_height=720
        self._carstartpointX=675
        self._carstartpointY=134
        self._currentcarX=self._carstartpointX
        self._currentcarY=self._carstartpointY
        self._carwidth=20
        self._carheight=10
        self._car_color=(0,0,255)
        self._car_surface=pygame.Surface((self._carwidth, self._carheight), pygame.SRCALPHA)
        self._car_surface.fill(self._car_color)
        self._toRaceFrame=False
        self._toTrainFrame=False
        self._toMainFrame=False
        self._quitgame=False
        self._buttonfont=pygame.font.Font(None, 36)
        self._buttoncolor=(255, 102, 0)
        self._BUTTON_HOVER_COLOR = (255, 153, 51)
        self._image = pygame.image.load('TrackRaceFinal.jpg')
        self._image = self._image.convert()
        self._image=pygame.transform.scale(self._image, (self.__desired_width, self.__desired_height))
    def _draw(self):
        pass
    def _readbuttoninput(self):
        pass
    def animate(self):
        pass

class MainFrame(Frame):
    def __init__(self, screen):
        super().__init__(screen=screen)
        self.TurnOff=False
        self.titlesize=pygame.Rect(500, 100, 200, 100)
        self._buttontext=["Race", "Train", "Exit"]
        for i in range(3):
            button = pygame.Rect(
                550,
                250+i*100,
                self._buttonwidth,
                self._buttonheight
            )
            self._buttons.append(button) 
        titlefont=pygame.font.Font(size=128)
        self.Title= titlefont.render("Rat Race", True, (0,0,0))
        self.TitleRect=self.Title.get_rect(center=self.titlesize.center)
        
    def _draw(self):
        self.screen.fill((50, 153, 50))
        #Display buttons on screen
        for i, button in enumerate(self._buttons):
            pygame.draw.rect(
                self.screen,
                self._buttoncolor if button.collidepoint(pygame.mouse.get_pos()) else self._BUTTON_HOVER_COLOR,
                button,
                )
            text= self._buttonfont.render(self._buttontext[i], True, (0,0,0))
            text_rect=text.get_rect(center=button.center)
            self.screen.blit(text, text_rect)
        self.screen.blit(self.Title,self.TitleRect)

    def _readbuttoninput(self):
         for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quitgame=True 
            if event.type == pygame.VIDEORESIZE:
                width, height=event.w, event.h
                self.screen=pygame.display.set_mode((width, height),pygame.RESIZABLE) 
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, button in enumerate(self._buttons):
                    if button.collidepoint(event.pos):
                        if i == 0:
                            print("Race button clicked")
                            self._toRaceFrame=True
                        elif i == 1:
                            print("Train button clicked")
                            self._toTrainFrame=True
                        elif i == 2:
                            self.TurnOff=True

class TrainFrame(Frame):
    def __init__(self, screen):
        super().__init__(screen)
        self.__training=False
        self.__SaveFlag=False
        self.__LoadFlag=False
        self._buttontext=["Start", "Save", "Load", "Back"]
        self.__hudcolor=(207,129,27)
        self.__hudrect=pygame.Rect(0, 0, 1200, 100)
        self.__car_path=self.generatePath()
        for i in range(len(self._buttontext)):
            button=pygame.Rect(
            250*(i+1),
            25,
            self._buttonwidth,
            self._buttonheight
            )
            self._buttons.append(button)
    def getSaveFlag(self):
        return self.__SaveFlag
    def setSaveFlag(self, s:bool):
        self.__SaveFlag=s
    def getLoadFlag(self):
        return self.__LoadFlag
    def setLoadFlag(self, l:bool):
        self.__LoadFlag=l
    def _draw(self):
        self.screen.fill((50, 153, 50))
        self.screen.blit(self._image, (0,0))
        pygame.draw.rect(self.screen,self.__hudcolor, self.__hudrect)
       
        for i, button in enumerate(self._buttons):
            pygame.draw.rect(
                self.screen,
                self._buttoncolor if button.collidepoint(pygame.mouse.get_pos()) else self._BUTTON_HOVER_COLOR,
                button,
                )
            text= self._buttonfont.render(self._buttontext[i], True, (0,0,0))
            text_rect=text.get_rect(center=button.center)
            self.screen.blit(text, text_rect)
        if(self.__training==False):
            self.screen.blit(self._car_surface, (self._currentcarX, self._currentcarY))           
    def animate(self):
        angle=0
        while(self.__training):
            for i in (self.__car_path):
                print(i)
                self._currentcarX=(i[0]/60)
                self._currentcarY=(i[1]/60)
                angle+=i[2]/60
                rotated_car=pygame.transform.rotate(self._car_surface, -angle)
                self.screen.blit(rotated_car, (self._currentcarX, self._currentcarY))
        self.__training=False
        self._currentcarX=self._carstartpointX
        self._currentcarY=self._carstartpointY
    def generatePath(self):
        currentx=self._carstartpointX
        currenty=self._carstartpointY
        currentTheta=3
        path=[]
        for i in range(10000):
            position=(currentx,currenty,currentTheta)
            print(position)
            currentx+=1
            currenty+=1
            currentTheta+=3
            path.append(position)
        return path
    def _readbuttoninput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quitgame=True
            if event.type == pygame.VIDEORESIZE:
                width, height=event.w, event.h
                self.screen=pygame.display.set_mode((width, height),pygame.RESIZABLE) 
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, button in enumerate(self._buttons):
                    if button.collidepoint(event.pos):
                        if i == 0 and self.__training==False:
                            self.__training=True
                            #background_thread=threading.Thread(target=self.animate)
                            #background_thread.daemon=True
                            #background_thread.start()
                            print("Start button clicked")
                        elif i == 1:
                            self.__SaveFlag=True
                            print("Save button clicked")
                        elif i == 2:
                            self.__LoadFlag=True
                            print("Load button clicked")
                        elif i==3:
                            print("Back is pressed")
                            self.__SaveFlag=False
                            self.__LoadFlag=False
                            self.__training=False
                            self._toMainFrame=True
                
class RaceFrame(Frame):
    def __init__(self, screen):
        super().__init__(screen)
        self.__racing=False
        self._buttontext=["Race","Back"]
        self.__hudcolor=(207,129,27)
        self.__hudrect=pygame.Rect(0, 0, 1200, 100)
        for i in range(len(self._buttontext)):
            button=pygame.Rect(
            300*(i)+400,
            25,
            self._buttonwidth,
            self._buttonheight
            )
            self._buttons.append(button)
        
    def _draw(self):
        self.screen.fill((50, 153, 50))
        self.screen.blit(self._image, (0,0))
        pygame.draw.rect(self.screen,self.__hudcolor, self.__hudrect)
        for i, button in enumerate(self._buttons):
            pygame.draw.rect(
                self.screen,
                self._buttoncolor if button.collidepoint(pygame.mouse.get_pos()) else self._BUTTON_HOVER_COLOR,
                button,
                )
            text= self._buttonfont.render(self._buttontext[i], True, (0,0,0))
            text_rect=text.get_rect(center=button.center)
            self.screen.blit(text, text_rect)
        if(self.__racing==False):
            self.screen.blit(self._car_surface, (self._currentcarX, self._currentcarY))  
    def animate(self):
        angle=0
        center_x=600
        center_y=450
        car_radius=150
        while(self.__racing):
            car_x = center_x + car_radius * math.cos(math.radians(angle*0.001))
            car_y = center_y + car_radius * math.sin(math.radians(angle*0.001))
            self._currentcarX=car_x
            self._currentcarY=car_y
            rotated_car=pygame.transform.rotate(self._car_surface, -angle*0.001)
            self.screen.blit(rotated_car, (self._currentcarX, self._currentcarY))
            angle+=1
        self.__racing=False
        self._currentcarX=self._carstartpointX
        self._currentcarY=self._carstartpointY
    def _readbuttoninput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quitgame=True
            if event.type == pygame.VIDEORESIZE:
                width, height=event.w, event.h
                self.screen=pygame.display.set_mode((width, height),pygame.RESIZABLE) 
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, button in enumerate(self._buttons):
                    if button.collidepoint(event.pos):
                        if i == 0:
                            print("Race button clicked")
                            self.__racing=True
                            background_thread=threading.Thread(target=self.animate)
                            background_thread.daemon=True
                            background_thread.start()
                        elif i == 1:
                            print("Back button clicked")
                            self.__racing=False
                            self._toMainFrame=True
                
class GUI():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 720), pygame.RESIZABLE)
        pygame.display.set_caption('Rat Race')
        self.MF=MainFrame(self.screen)
        self.TF=TrainFrame(self.screen)
        self.RF=RaceFrame(self.screen)
        self.currentframe=self.MF
        self.running=True
        self.run()
    def train_save():
        pass
    def train_laod():
        pass
    def race_start():
        pass
    def race_setup_canvas():
        pass
    def train_setup_canvas():
        pass
    def animate():
        pass
    def train_start():
        pass
    def run(self):
        while self.running:
            self.currentframe._draw()
            self.currentframe._readbuttoninput()
            pygame.display.flip()
            if (self.currentframe._toTrainFrame):
                self.currentframe._toTrainFrame=False
                self.currentframe=self.TF
            if(self.currentframe._toMainFrame):
                self.currentframe._toMainFrame=False
                self.currentframe=self.MF
            if(self.currentframe._toRaceFrame):
                self.currentframe._toRaceFrame=False
                self.currentframe=self.RF
            if (self.MF.TurnOff or self.currentframe._quitgame):
                self.running=False
        pygame.quit()
        sys.exit()
if __name__ in "__main__":
    GUI()
