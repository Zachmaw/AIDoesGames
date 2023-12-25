# import pygame


# so the Sim class calls one of the Game classes( which all inherit from the Env class.), but the advanceTime function is called where exactly??
# The Simulation is the one who advances time in the environment by one step/turn when it has the responses from the Agents!

# All environments are either turn based or time based, but...
class Env:
    def __init__(self, id):# The id is given to us from the Simulation
        self.ready = False
        self.id = id
        self.tookTurn = {}### use this somehow to build initiative order based environments.
        self.moves = {}


    def advanceTime(self, actionVector:'list[int]', envRules:'function'):
        '''
        should take in the action of the current agent and\n
        return a list of observations that can be made about the environment\n
        which comes directly from envRules\n
        based on the game being played\n
        Each Agent is always allowed one action per timestep.\n
        It is important to maintain the order of Agent actions.'''
        # which is why I'm going with the initiative method.
        # time advances one tick with each revolution of the initiative tracker.
        ### if this agent is first in the turn order, increase SimRuntime by one tick.
        return envRules(actionVector)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from operator import itemgetter
import random

# ------ DREWCIFER'S HANDY DANDY PYTHON INITIATIVE CALCULATOR FOR D&D -------
# Designed so you enter in your PCs' names once at the start of each session.
# Those names are re-used for each subsequent fight. Monsters are numbered
# simply, this should be tracked by the DM (put numbered tokens next to each
# monster or something to make this easy). At the end of each fight, hit enter
# to start over or 'n' to quit. If you need to add a new PC or make a mistake
# or something, just quit the program (CTRL + C) and run the script again to
# start over.

# Player setup
players = input('How many PCs? ')
player_names = []
for i in range(0, players):
    name = input('PC ' + str(i + 1) + ' name: ')
    player_names.append(name)

# New encounter
new_fight = True

while new_fight is True:

    # Type in each player's initiative total
    player_init = []
    for i in range(0, players):
        pinit = input(player_names[i] + ' initiative total: ')
        player_init.append(pinit)

    # Combines the PC names with their initiative rolls in pairs in a list
    player_list = [None] * players
    for i in range(0, players):
        player_list[i] = ('- ' + player_names[i]), player_init[i]

    # How many enemies do we have?
    enemies = input('How many Monsters? ')

    # Type in each enemy's initiative modifier
    enemy_init = []
    for i in range(0, enemies):
        einit = input('Monster ' + str(i + 1) + ' initiative modifier? ')
        enemy_init.append(einit)

    # Rolls each enemy's initiative, adding their modifier, adds these rolls
    # into a list with the enemy names, in pairs
    monster_list = []
    for i in range(0, enemies):
        init_roll = random.randint(1, (20 + enemy_init[i]))
        monster_list.append(['- Monster ' + str(i + 1), init_roll])

    # Combines the list and sorts by initiative roll
    init_list = player_list + monster_list
    init_list = sorted(init_list, key=itemgetter(1), reverse=True)

    print("\n")
    print("--- Initiative list ---")

    # Prints the list in readable strings without brackets/commas etc
    for entry in init_list:
        print(str(entry).strip("[]()").replace("'", "").replace(",", ":"))

    # Choice to start a new battle with the same group of PCs, or quit
    print("\n")
    again = input('Hit the enter key to start a new battle, or "n" to exit...')
    if again == "n":
        new_fight = False
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DefaultGameObj(Env):
    def __init__(self, id):### add the other atributes a game object would have
        super().__init__(id)
        self.maxPlayers = 7# This game has a player limit
        self.wins = {}# it also can be won.
        self.ties = 0# In case nobody could claim the game.

    def awaitPlayerConnections(self):
        trying = True
        while trying:
            try:
                playerCount = int(input('How many players?'))
                if playerCount > self.maxPlayers:
                    print(f"The number is too high! The highest number of players you can have in this environment is {self.maxPlayers}")
                    Exception("The number is too high!(playerCount above maxPlayers)")
                for i in playerCount:
                    self.tookTurn[f'Player{str(i)}'] = False
                    self.moves[f'Player{str(i)}'] = None
                    self.wins[f'Player{str(i)}'] = 0
                trying = False
            except:
                print('Please, try again.')
        temp2 = input('')
        pass###

    def get_player_move(self, p):
        """
        :param p: [0,1]
        :return: Move
        """
        return self.moves[p]

    def play(self, player, move):
        self.moves[player] = move
        if player == 0:
            self.p1Went = True
        else:
            self.p2Went = True

    def connected(self):
        '''Check self.ready'''
        return self.ready

    def bothWent(self):
        return self.p1Went and self.p2Went

    def winner(self):

        p1 = self.moves[0].upper()[0]
        p2 = self.moves[1].upper()[0]

        winner = -1
        if p1 == "R" and p2 == "S":
            winner = 0
        elif p1 == "S" and p2 == "R":
            winner = 1
        elif p1 == "P" and p2 == "R":
            winner = 0
        elif p1 == "R" and p2 == "P":
            winner = 1
        elif p1 == "S" and p2 == "P":
            winner = 0
        elif p1 == "P" and p2 == "S":
            winner = 1

        return winner

    def resetWent(self):
        self.p1Went = False
        self.p2Went = False











# class Game:
# 	def __init__(self):
# 		pygame.init()
# 		self.display_surface = pygame.display.set_mode(SCREEN_SIZE)
# 		pygame.display.set_caption('Ultimate intro')
# 		self.clock = pygame.time.Clock()

# 		self.all_sprites = AllSprites()
# 		self.lasers = pygame.sprite.Group()
# 		self.meteors = pygame.sprite.Group()
# 		self.player = Player(self.all_sprites, (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - 100), self.create_laser)

# 		# imports
# 		self.star_frames = import_folder('..\graphics\star')
# 		self.explosion = import_folder('..\graphics\explosion')
# 		self.meteor_surfaces = import_folder('..\graphics\meteors')

# 		self.broken_surf = pygame.image.load('../graphics/broken.png').convert_alpha()
# 		self.broken_rect = self.broken_surf.get_rect(center = (SCREEN_SIZE[0]/2, SCREEN_SIZE[1]/2 - 100))

# 		# timer
# 		self.meteor_timer = pygame.USEREVENT + 1
# 		pygame.time.set_timer(self.meteor_timer, 150)

# 		# bg stars
# 		for i in range(randint(50,70)):
# 			AnimatedStar(self.all_sprites, self.star_frames)

# 		# overlay
# 		self.overlay = Overlay()

# 		# score
# 		self.score = 0
# 		self.lifes = 3
# 		self.start_time = 0
# 		self.game_over = False

# 		# font
# 		self.font = pygame.font.Font(None, 40)

# 		# restart button
# 		self.restart_surf = self.font.render('Restart', True, TEXT_COLOR)
# 		self.restart_rect = self.restart_surf.get_rect(center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - 100))
# 		self.restart_surf_dark = self.font.render('Restart', True, BG_COLOR)

# 		# music
# 		self.game_music = pygame.mixer.Sound('../audio/game_music.wav')
# 		self.title_music = pygame.mixer.Sound('../audio/title_music.wav')
# 		self.laser_sound = pygame.mixer.Sound('../audio/laser.wav')
# 		self.explosion_sound = pygame.mixer.Sound('../audio/explosion.wav')
# 		self.damage_sound = pygame.mixer.Sound('../audio/damage.ogg')

# 		if not self.game_over:
# 			self.game_music.play()
# 		else:
# 			self.title_music.play()

# 	def create_laser(self, pos, direction):

# 		Laser((self.all_sprites, self.lasers), pos, direction)
# 		self.laser_sound.play()

# 	def collisions(self):

# 		# laser -> meteor
# 		for laser in self.lasers:
# 			if pygame.sprite.spritecollide(laser, self.meteors, True, pygame.sprite.collide_mask):
# 				Explosion(self.all_sprites, self.explosion, laser.rect.midtop)
# 				laser.kill()
# 				self.explosion_sound.play()

# 		# meteor -> player
# 		if pygame.sprite.spritecollide(self.player, self.meteors, True, pygame.sprite.collide_mask):
# 			self.lifes -= 1
# 			self.damage_sound.play()

# 			if self.lifes <= 0:
# 				self.score = pygame.time.get_ticks() - self.start_time
# 				self.game_over = True
# 				for meteor in self.meteors:
# 					meteor.kill()
# 				self.player.rect.center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - 100)

# 				self.game_music.stop()
# 				self.title_music.play()

# 	def run(self):
# 		while True:
# 			for event in pygame.event.get():
# 				if event.type == pygame.QUIT:
# 					pygame.quit()
# 					sys.exit()
# 				if event.type == self.meteor_timer and not self.game_over:
# 					Meteor((self.all_sprites, self.meteors), choice(self.meteor_surfaces))

# 			self.display_surface.fill(BG_COLOR)

# 			if self.game_over:

# 				self.display_surface.blit(self.broken_surf,self.broken_rect)
# 				# text
# 				text_surf = self.font.render(f'Your score: {self.score}', True, TEXT_COLOR)
# 				text_rect = text_surf.get_rect(center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2 + 50))
# 				self.display_surface.blit(text_surf, text_rect)

# 				# button
# 				if self.restart_rect.collidepoint(pygame.mouse.get_pos()):
# 					pygame.draw.rect(self.display_surface, TEXT_COLOR, self.restart_rect.inflate(30,30),0,3)
# 					self.display_surface.blit(self.restart_surf_dark, self.restart_rect)

# 					if pygame.mouse.get_pressed()[0]:
# 						self.game_over = False
# 						self.lifes = 3
# 						self.start_time = pygame.time.get_ticks()

# 						self.title_music.stop()
# 						self.game_music.play()
# 				else:
# 					self.display_surface.blit(self.restart_surf, self.restart_rect)

# 				pygame.draw.rect(self.display_surface, TEXT_COLOR, self.restart_rect.inflate(30,30), 5,3)
# 			else:
# 				dt = self.clock.tick() / 1000

# 				self.score = pygame.time.get_ticks() - self.start_time
# 				self.overlay.display_score(self.score)

# 				self.all_sprites.update(dt)
# 				self.all_sprites.custom_draw()

# 				self.collisions()

# 				self.overlay.display_lifes(self.lifes)

# 			pygame.display.update()


# game = Game()
# game.run()








# print(pygame.K_0)
# print(pygame.K_w)
# print(pygame.K_LEFT)
# print(pygame.K_SPACE)
# print(pygame.K_DOLLAR)