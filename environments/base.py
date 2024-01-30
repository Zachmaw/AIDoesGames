

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# so the Sim class calls one of the Game classes( which all inherit from the Env class.), but the advanceTime function is called where exactly??
# The Simulation is the one who advances time in the environment by one step/turn when it has the responses from the Agents!

# All environments start off initiative based, but TurnBased modifies it slightly...
# How? Well, let's start be explaining Initiative based order.
# First, the Environment is selected. The Env rules are loaded into the Env class or it's child.
# Then, from envRules, we know the expected NN outputs and player count and can populate the Sim with the NN/Agents and their speed.
# Sim
class Env:
    '''
    All inheritors must declare the following:
    PlayerCount
    '''
    def __init__(self, id):# The id is given to us from the Simulation
        self.ready = False
        self.runid = id
        self.init_order = list()
        # self.init_order.append((50, "LairAction"))
        self.thisRoundResponses = list()# to be filled to player count once an EnvChild inherits it.
        # self.gamestate = dict()
        self.PlayerCount = None

    def setRules(self, gameRules:"function"):
        self.rules = gameRules
    def actAndObserve(self, actionChoice:"int"):
        return self.rules(actionChoice)
    
    def getPlayerCount(self):
        return self.playerCount

    def advance(self):
        ### from whose turn it is in the game, update the internal game state based on actions taken
        # everyone whose turn it is not still recieves game data
        # but they also see it's not their turn.
        # This function is called after all players have submitted their responses.
        # somehow update the gameState from the response from player-whos-turn.
        # and set new observations for... all? just active player?
        #
        # Is it better to represent gamestate as a list of ints with variable length based on environment
        # would a list of int work?? does it have to be dict?
        # yeah, List is probably better. I can itterate over it.
        # I have to convert it to a list of floats anyway for the Agents.
        # I should find a way to just deliver the raw ints to Players.
        # but dict is easier for me to understand? keywise. as opposed to indexwise.
        ### so, from gamestate, get whose turn it is
        move = self.thisRoundResponses[self.gamestate["turnTracker"]]# Sim tracks whose turn it is.
        ### problem here is: roundOfResponses is ordered in sync with initOrder which is not the same as player order.
        if player == 0:
            self.p1Went = True
        else:
            self.p2Went = True


#####
class Chess(Env):
    def __init__(self, id, maxPlayers):### add the other atributes a game object would have( like??)
        super().__init__(id)
        self.playerLimit = maxPlayers# The player limit
        self.playersTookTurn = list()# list of bool with length = maxPlayers
        self.gamestate["turnTracker"] = int()# This game is turn based. int represents index in initOrder.
        # initOrder which has already been sorted so the highest speed has the lowest index. Descending order.

    def awaitPlayerConnections(self):### I don't think Env needs this.
        trying = True
        while trying:
            try:
                playerCount = int(input('How many players?'))
                if playerCount > self.playerLimit:
                    print(f"The number is too high! The highest number of players you can have in this environment is {self.playerLimit}")
                    Exception("The number is too high!(playerCount above maxPlayers)")
                for i in range(playerCount):### add each player to self.players
                    self.tookTurn[f'Player{str(i)}'] = False
                    self.moves[f'Player{str(i)}'] = None
                    self.wins[f'Player{str(i)}'] = 0
                trying = False
            except:
                print('Please, try again.\nJust for this game, ')### incomplete
        temp2 = input('')
        pass###

    def get_player_choice(self, p):### should be a function of Sim, not Env.
        """
        :param p: [0,1]
        :return: Move
        """
        return self.moves[p]


    def connected(self):### Again, Sim probably needs one of these, not Env.
        '''Check self.ready'''
        return self.ready

    def checkWinner(self):
        # cycle through all players answers, returning whose response is closest to the selected number without going over.
        # maybe track responses in a list, sort and return the top result
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


class TurnBased(DefaultGame):
    def __init__(self, id):
        super().__init__(id)
        self.tookTurn = {}
        self.moves = {}
        ### and essentially just make it so if it's not that Agent's turn, it is blindfolded



    def resetWent(self):
        self.p1Went = False
        self.p2Went = False

    def allWent(self):### if not any(list of bool representing players)
        ### make variable to number of players
        return self.p1Went and self.p2Went














# name_to_class = dict(some=SomeClass,
#                      other=OtherClass)

# def factory(name):
#     klass = name_to_class(name)
#     return klass()










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