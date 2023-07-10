import pygame, sys 
from settings import *
from player import Player
from objects import Laser, Meteor
from animations import AnimatedStar, Explosion
from random import randint
from support import import_folder, importImgs
from overlay import Overlay
from group import AllSprites
from random import choice
from os import path




# # The main function
# def main():
#     introduction = intro()
#     board = initBoard()
#     pretty = printPretty(board)
#     symbol_1, symbol_2 = sym()
#     full = isFull(board, symbol_1, symbol_2) # The function that starts the game is also in here.
    

    


# # This function introduces the rules of the game Tic Tac Toe
# def intro():
#     print("Hello! Welcome to Pam's Tic Tac Toe game!")
#     print("\n")
#     print("Rules: Player 1 and player 2, represented by X and O, take turns "
#           "marking the spaces in a 3*3 grid. The player who succeeds in placing "
#           "three of their marks in a horizontal, vertical, or diagonal row wins.")
#     print("\n")
#     input("Press enter to continue.")
#     print("\n")



# def initBoard():
# # This function creates a blank playboard
#     print("Here is the playboard: ")
#     board = [[" ", " ", " "],
#              [" ", " ", " "],
#              [" ", " ", " "]]        
#     return board



# def sym():
# # This function decides the players' symbols
#     symbol_1 = input("Player 1, do you want to be X or O? ")
#     if symbol_1 == "X":
#         symbol_2 = "O"
#         print("Player 2, you are O. ")
#     else:
#         symbol_2 = "X"
#         print("Player 2, you are X. ")
#     input("Press enter to continue.")
#     print("\n")
#     return (symbol_1, symbol_2)



# # This function starts the game.
# def startGamming(board, symbol_1, symbol_2, count):
#     # Decides the turn
#     if count % 2 == 0:
#         player = symbol_1
#     elif count % 2 == 1:
#         player = symbol_2
#     print("Player "+ player + ", it is your turn. ")
#     row = int(input("Pick a row:"
#                     "[upper row: enter 0, middle row: enter 1, bottom row: enter 2]:"))
#     column = int(input("Pick a column:"
#                        "[left column: enter 0, middle column: enter 1, right column enter 2]"))


#     # Check if players' selection is out of range, if so, repick.
#     while (row > 2 or row < 0) or (column > 2 or column < 0):
#         outOfBoard(row, column)
#         row = int(input("Pick a row[upper row:"
#                         "[enter 0, middle row: enter 1, bottom row: enter 2]:"))
#         column = int(input("Pick a column:"
#                            "[left column: enter 0, middle column: enter 1, right column enter 2]"))

#         # Check if the square is already filled
#     while (board[row][column] == symbol_1)or (board[row][column] == symbol_2):
#         filled = illegal(board, symbol_1, symbol_2, row, column)
#         row = int(input("Pick a row[upper row:"
#                         "[enter 0, middle row: enter 1, bottom row: enter 2]:"))
#         column = int(input("Pick a column:"
#                             "[left column: enter 0, middle column: enter 1, right column enter 2]"))    
        
#     # Locates player's symbol on the board
#     if player == symbol_1:
#         board[row][column] = symbol_1
            
#     else:
#         board[row][column] = symbol_2
    
#     return (board)



# # This function check if the board is full
# def isFull(board, symbol_1, symbol_2):
#     count = 1
#     winner = True
#     while count < 10 and winner == True:
#         gaming = startGamming(board, symbol_1, symbol_2, count)
#         pretty = printPretty(board)
        
#         if count == 9:
#             print("The board is full. Game over.")
#             if winner == True:
#                 print("There is a tie. ")

#         # Check if here is a winner
#         winner = isWinner(board, symbol_1, symbol_2, count)
#         count += 1
#     if winner == False:
#         print("Game over.")
        
#     # This is function gives a report 
#     report(count, winner, symbol_1, symbol_2)



# # This function tells the players that their selection is out of range
# def outOfBoard(row, column):
#     print("Out of boarder. Pick another one. ")
    
    

# # This function prints the board nice!
# def printPretty(board):
#     rows = len(board)
#     cols = len(board)
#     print("---+---+---")
#     for r in range(rows):
#         print(board[r][0], " |", board[r][1], "|", board[r][2])
#         print("---+---+---")
#     return board



# # This function checks if any winner is winning
# def isWinner(board, symbol_1, symbol_2, count):
#     winner = True
#     # Check the rows
#     for row in range (0, 3):
#         if (board[row][0] == board[row][1] == board[row][2] == symbol_1):
#             winner = False
#             print("Player " + symbol_1 + ", you won!")
   
#         elif (board[row][0] == board[row][1] == board[row][2] == symbol_2):
#             winner = False
#             print("Player " + symbol_2 + ", you won!")
            
            
#     # Check the columns
#     for col in range (0, 3):
#         if (board[0][col] == board[1][col] == board[2][col] == symbol_1):
#             winner = False
#             print("Player " + symbol_1 + ", you won!")
#         elif (board[0][col] == board[1][col] == board[2][col] == symbol_2):
#             winner = False
#             print("Player " + symbol_2 + ", you won!")

#     # Check the diagnoals
#     if board[0][0] == board[1][1] == board[2][2] == symbol_1:
#         winner = False 
#         print("Player " + symbol_1 + ", you won!")

#     elif board[0][0] == board[1][1] == board[2][2] == symbol_2:
#         winner = False
#         print("Player " + symbol_2 + ", you won!")

#     elif board[0][2] == board[1][1] == board[2][0] == symbol_1:
#         winner = False
#         print("Player " + symbol_1 + ", you won!")

#     elif board[0][2] == board[1][1] == board[2][0] == symbol_2:
#         winner = False
#         print("Player " + symbol_2 + ", you won!")

#     return winner
    


# def illegal(board, symbol_1, symbol_2, row, column):
#     print("The square you picked is already filled. Pick another one.")

    
# def report(count, winner, symbol_1, symbol_2):
#     print("\n")
#     input("Press enter to see the game summary. ")
#     if (winner == False) and (count % 2 == 1 ):
#         print("Winner : Player " + symbol_1 + ".")
#     elif (winner == False) and (count % 2 == 0 ):
#         print("Winner : Player " + symbol_2 + ".")
#     else:
#         print("There is a tie. ")

# # Call Main
# if __name__ == "__main__":
#     main()



#####################################################################

class Game:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Ultimate tactoe')
        self.clock = pygame.time.Clock()

        self.all_sprites = AllSprites()
        self.lasers = pygame.sprite.Group()
        self.meteors = pygame.sprite.Group()

        # imports 
        self.star_frames = import_folder('../img/star')
        self.explosion = import_folder('../img/explosion')
        self.meteor_surfaces = import_folder('../img/meteors')
        self.spareImgs = importImgs(os.path.join('../img')#####
        print(self.spareImgs)
        # self.playerimage = pygame.image.load(os.path.join(imgPath, '/img/player.png')).convert_alpha()
        # self.playerimage = pygame.image.load('../img/player.png').convert_alpha()
        # self.broken_surf = pygame.image.load('../img/broken.png').convert_alpha()
        self.broken_rect = self.broken_surf.get_rect(center = (SCREEN_SIZE[0]/2, SCREEN_SIZE[1]/2 - 100))

        self.player = Player(self.all_sprites, (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - 100), self.create_laser, self.playerimage)

        # timer 
        self.meteor_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.meteor_timer, 150)

        # bg stars 
        for i in range(randint(50,70)):
            AnimatedStar(self.all_sprites, self.star_frames)

        # overlay
        self.overlay = Overlay()

        # score 
        self.score = 0
        self.lifes = 3
        self.start_time = 0
        self.game_over = False

        # font
        self.font = pygame.font.Font(None, 40)

        # restart button 
        self.restart_surf = self.font.render('Restart', True, TEXT_COLOR)
        self.restart_rect = self.restart_surf.get_rect(center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - 100))
        self.restart_surf_dark = self.font.render('Restart', True, BG_COLOR)

        # music 
        self.game_music = pygame.mixer.Sound('../audio/game_music.wav')
        self.title_music = pygame.mixer.Sound('../audio/title_music.wav')
        self.laser_sound = pygame.mixer.Sound('../audio/laser.wav')
        self.explosion_sound = pygame.mixer.Sound('../audio/explosion.wav')
        self.damage_sound = pygame.mixer.Sound('../audio/damage.ogg')


    def create_laser(self, pos, direction):

        Laser((self.all_sprites, self.lasers), pos, direction)
        self.laser_sound.play()

    def collisions(self):

        # laser -> meteor
        for laser in self.lasers:
            if pygame.sprite.spritecollide(laser, self.meteors, True, pygame.sprite.collide_mask):
                Explosion(self.all_sprites, self.explosion, laser.rect.midtop)
                laser.kill()
                self.explosion_sound.play()

        # meteor -> player 
        if pygame.sprite.spritecollide(self.player, self.meteors, True, pygame.sprite.collide_mask):
            self.lifes -= 1
            self.damage_sound.play()

        if self.lifes <= 0:
            self.score = pygame.time.get_ticks() - self.start_time
            self.game_over = True
            for meteor in self.meteors:
                meteor.kill()
            self.player.rect.center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - 100)

            self.game_music.stop()
            self.title_music.play() 

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == self.meteor_timer and not self.game_over:
                    Meteor((self.all_sprites, self.meteors), choice(self.meteor_surfaces))

            self.display_surface.fill(BG_COLOR)

            if self.game_over:

                self.display_surface.blit(self.broken_surf,self.broken_rect)	
            # text 
            text_surf = self.font.render(f'Your score: {self.score}', True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2 + 50))
            self.display_surface.blit(text_surf, text_rect)

            # button 
            if self.restart_rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(self.display_surface, TEXT_COLOR, self.restart_rect.inflate(30,30),0,3)
                self.display_surface.blit(self.restart_surf_dark, self.restart_rect)

                if pygame.mouse.get_pressed()[0]:
                    self.game_over = False
                    self.lifes = 3
                    self.start_time = pygame.time.get_ticks()

                    self.title_music.stop() 
                    self.game_music.play()
                else:
                    self.display_surface.blit(self.restart_surf, self.restart_rect)

                pygame.draw.rect(self.display_surface, TEXT_COLOR, self.restart_rect.inflate(30,30), 5,3)
            else:
                dt = self.clock.tick() / 1000

                self.score = pygame.time.get_ticks() - self.start_time
                self.overlay.display_score(self.score)

                self.all_sprites.update(dt)
                self.all_sprites.custom_draw()

                self.collisions()

                self.overlay.display_lifes(self.lifes)

                pygame.display.update()


game = Game()
game.run()