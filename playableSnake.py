import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
#import settings

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0,0,0)

BLOCK_SIZE = 40
#SPEED = settings.SNAKE_SPEED
SPEED = 10

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def euclidean_distance(self, p1, p2):
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.press = None
        self._place_food()
        #self.frame_iteration = 0
        #return pygame.surfarray.array3d(self.display)
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self):
        #self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.press = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.press = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.press = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.press = Direction.DOWN
        
        # 2. move
        self._move(self.press) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        #reward = 0
        game_over = False
        if self.is_collision():
        #if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            #reward = -10
            return game_over, self.score
            #return pygame.surfarray.array3d(self.display), reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            #if settings.REWARD_STRUCTURE == 'lengthMax':
                #reward = (0.25*self.score)*10
            #else:    
                #reward = 10
            
            self._place_food()
        else:
            #if settings.REWARD_STRUCTURE == 'findFood':
                #if self.euclidean_distance(self.head, self.food) < self.euclidean_distance(self.snake[1], self.food):
                    #reward = 0.1
                #else:
                    #reward = -0.1
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            if pt == self.snake[0]:
                pygame.draw.rect(self.display, YELLOW, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, YELLOW, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            else:
                pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        '''if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d'''
        #action = action.item()
        if action == Direction.RIGHT: #np.array_equal(action, [1, 0, 0, 0]):
            if self.direction == Direction.LEFT:
                self.direction = Direction.LEFT
            else:
                self.direction = Direction.RIGHT
        elif action == Direction.DOWN: #np.array_equal(action, [0, 1, 0, 0]):
            if self.direction == Direction.UP:
                self.direction = Direction.UP
            else:
                self.direction = Direction.DOWN
        elif action == Direction.LEFT:#np.array_equal(action, [0, 0, 1, 0]):
            if self.direction == Direction.RIGHT:
                self.direction = Direction.RIGHT
            else:
                self.direction = Direction.LEFT
        else: #np.array_equal(action, [0, 0, 0, 1]):
            if self.direction == Direction.DOWN:
                self.direction = Direction.DOWN
            else:
                self.direction = Direction.UP

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

if __name__ == '__main__':
    game = SnakeGameAI()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()