# import pygame





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




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# so the Sim class calls one of the Game classes( which all inherit from the Env class.), but the advanceTime function is called where exactly??
# The Simulation is the one who advances time in the environment by one step/turn when it has the responses from the Agents!

# All environments start off initiative based, but TurnBased modifies it slightly...
# How? Well, let's start be explaining Initiative based order.
# First, the Environment is selected. The Env rules are loaded into the Env class or it's child.
# Then, from envRules, we know the expected NN outputs and player count and can populate the Sim with the NN/Agents and their speed.
# Sim
class Env:
    def __init__(self, id):# The id is given to us from the Simulation
        self.ready = False
        self.runid = id
        ### What about the initiative tracker program I found? Can't that help? Here's what's left of it.
        self.init_order = list()
        self.init_order.append((20, "LairAction"))

    def addAgent(self, agentID:"tuple(int)", speed:"int"):### why is that a tuple? how big is it suposed to be?
        self.init_order.append(speed, str("LairAction"))
        # sorts by initiative roll
        self.init_order = sorted(self.init_order, key=itemgetter(1), reverse=True)
        #####
        # What I need to do is either put the AgentID in the initiative order list
        # or put the Agent itself in the list?
        # well, when I build the NN from a saved txt file Genome, I have to store it in memory.
        # no point using NN IDs pre generation.
        # Generate NN, store it in dict with key as f"NN{playerNumber}"





        ### IMPLEMENT SPEED GENE
        # NNs can clone, so technichally I don't HAVE to ever keep parents alive, right?
        # If that's the case, I can delete the selected genome from the gene pool the moment I build it into a NN?
        # When I build the NN, place it in initiative. but it needs a speed value...
        # It's time to extend the genome again, I guess... This gives me four bits
        # Let's put it at the beginning and trim it off before it get's to decodeBitstring()
        # Using these 16 permutations, I have 16 values for my initiative order...
        # Env should always have a value of 0 16 or 17, depending on sorting and the binary thing...
        # This speed value should be passed to addAgent along with the same Agents ID.
        #
        # I need to have a gene pool to reference with agentID
        # A gene pool can be:
        # A seperate folder for each Environments gene pool.
        # Where each genome in the pool/folder is a txt file.
        # where each gene in the genome/txt is represented as a string of hexdec characters.
        #
        # The next thing is the naming method for Genomes in storage.
        # The only Genomes in storage are the successful/best ones.
        #####

        # with open("GenePools\\testPool\\")
        #     avaliableGenePool =


    def advanceOneStep(self, actionVector:'list[int]', envRules:'function'):
        '''
        should take in the action of the current agent and\n
        return a list of observations that can be made about the environment\n
        which comes directly from envRules\n
        based on the game being played\n
        Each Agent is always allowed one action per timestep.\n
        It is important to maintain the order of Agent actions.'''
        # which is why I'm going with the initiative method.
        # time advances one tick with each revolution of the initiative tracker.
        ### if this agent is the Environment, increase SimRuntime by one tick.
        # that's it, save for specific Environments.
        return envRules(actionVector)

    def setRules(self, gameRules:"function"):
        self.rules = gameRules
    def actAndObserve(self, actionVector):
        return self.rules(actionVector)


class DefaultGameObj(Env):
    def __init__(self, id, maxPlayers):### add the other atributes a game object would have( like??)
        super().__init__(id)
        self.playerLimit = maxPlayers# The player limit
        self.players = {}# list of bool with length = maxPlayers

    def awaitPlayerConnections(self):
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

    def get_player_choice(self, p):
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

    def checkWinner(self):
        # cycle through all players tracking which response is closest to the selected number
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


class TurnBased(DefaultGameObj):
    def __init__(self, id):
        super().__init__(id)
        self.tookTurn = {}
        self.moves = {}
        ### and essentially just make it so if it's not that Agent's turn, it is blindfolded



    def resetWent(self):
        self.p1Went = False
        self.p2Went = False

    def allWent(self):
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