


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Shmup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def get_input(vector:'pygame.math.Vector2'):
#     keys = pygame.key.get_pressed()
#     vector.update((0,0))# direction reset to 0. If you ain't pressin anythin, y'ain't goin' anywhere.#####
#     if keys[pygame.K_RIGHT]:
#         vector.x = 1
#     if keys[pygame.K_LEFT]:
#         vector.x = -1
#     if keys[pygame.K_DOWN]:
#         vector.y = 1
#     if keys[pygame.K_UP]:
#         vector.y = -1
#     return vector.normalize() if vector.magnitude() != 0 else pygame.math.Vector2()# excuse me, what?# okay so, Normalize the vector UNLESS the vectors magnitude is 0. then just make an empty vector

# def display_score(time_passed):# text data
# 		text_surf = font.render(str(time_passed // 1000), False, 'white')
# 		text_rect = text_surf.get_rect(center = (SCREEN_SIZE[0] / 2, 100))

# 		win.blit(text_surf, text_rect)# score display
# 		pygame.draw.rect(win, 'white', text_rect.inflate(32,32),4,5)# frame
# def display_lives(num_lifes):
# 	for life in range(num_lifes):
# 		x = 20 + life * (icon_surf.get_width() + 4)
# 		y = SCREEN_SIZE[1] - 20
# 		icon_rect = icon_surf.get_rect(bottomleft = (x,y))
# 		win.blit(icon_surf, icon_rect)

# # init
# pygame.mixer.pre_init(44100, 16, 2, 4096) #frequency, size, channels, buffersize
# pygame.init() #turn all of pygame on.
# SCREEN_SIZE = (1280,720)
# win = pygame.display.set_mode(SCREEN_SIZE)
# pygame.display.set_caption('Spaceship')
# clock = pygame.time.Clock()
# font = pygame.font.Font(None, 40)
# img_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ticTacToe\\img')##
# sfx_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sfx')


# # load graphics
# img_loading_initial = pygame.image.load(os.path.join(img_dir, "loading.jpg")).convert()
# current_background = pygame.transform.scale(img_loading_initial, SCREEN_SIZE)
# win.blit(current_background, win.get_rect())
# img_background_menu = pygame.image.load(os.path.join(img_dir, "ColorfulHorizon.jpg")).convert()

# star_bg_surf = pygame.image.load(os.path.join(img_dir, 'star_bg.png')).convert_alpha()
# player_surf = pygame.image.load(os.path.join(img_dir, 'player.png')).convert_alpha()
# laser_surf = pygame.image.load(os.path.join(img_dir, 'laser.png')).convert_alpha()
# meteor_surf = pygame.image.load(os.path.join(img_dir, 'meteor.png')).convert_alpha()
# icon_surf = pygame.image.load(os.path.join(img_dir, 'icon.png')).convert_alpha()
# broken_surf = pygame.image.load(os.path.join(img_dir, 'broken.png')).convert_alpha()

# # bg stars

# # spaceship
# player_rect = player_surf.get_rect(center = (640,360))## was wrapped in an 'pygame.frect()'

# player_direction = pygame.math.Vector2()
# player_speed = 300

# # laser
# laser_speed = 500
# laser_data = []

# # meteor
# meteor_data = []
# meteor_timer = pygame.USEREVENT
# pygame.time.set_timer(meteor_timer, 1500)

# # life system
# lifeCount = 3

# # score system
# score = 0
# start_time = 0
# game_over = False

# # title
# broken_rect = broken_surf.get_rect(center = (SCREEN_SIZE[0]/2, SCREEN_SIZE[1]/2 - 100))
# restart_surf = font.render('Restart', True, 'White')
# restart_surf_dark = font.render('Restart', True, '#3a2e3f')
# restart_rect = restart_surf.get_rect(center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - 100))

# # audio
# game_music = pygame.mixer.Sound(os.path.join(sfx_dir, 'game_music.wav'))### path.join
# title_music = pygame.mixer.Sound(os.path.join(sfx_dir, 'title_music.wav'))
# laser_sound = pygame.mixer.Sound(os.path.join(sfx_dir, 'laser.wav'))
# explosion_sound = pygame.mixer.Sound(os.path.join(sfx_dir, 'explosion.wav'))
# damage_sound = pygame.mixer.Sound(os.path.join(sfx_dir, 'explosion.wav'))
# damage_sound.set_volume(0.5)

# game_music.play()

# while True:
# 	# get delta time
# 	dt = clock.tick() / 1000

# 	# event loop
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			pygame.quit()
# 			exit()
# 		if not game_over:
# 			if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
# 				laser_rect = pygame.rect.Rect(laser_surf.get_rect(midbottom = player_rect.center - pygame.math.Vector2(0,30)))
# 				laser_data.append({'rect':laser_rect, 'dokill': False})
# 				laser_sound.play()
# 			if event.type == meteor_timer:
# 				x,y  = randint(-100, SCREEN_SIZE[0] -100), randint(-300,-100)
# 				meteor_rect = pygame.Rect(meteor_surf.get_rect(center = (x, y)))
# 				meteor_direction = pygame.math.Vector2(randint(-3,3),2)
# 				meteor_speed = randint(300,600)
# 				meteor_data.append({'rect': meteor_rect, 'direction': meteor_direction, 'speed': meteor_speed, 'dokill': False})

# 	# bg color
# 	win.fill('#3a2e3f')

# 	# title screen
# 	if game_over:
# 		win.blit(broken_surf,broken_rect)

# 		# text
# 		text_surf = font.render(f'Your score: {score}', True, 'White')
# 		text_rect = text_surf.get_rect(center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2 + 50))
# 		win.blit(text_surf, text_rect)

# 		# button
# 		if restart_rect.collidepoint(pygame.mouse.get_pos()):
# 			pygame.draw.rect(win, 'white', restart_rect.inflate(30,30),0,3)
# 			win.blit(restart_surf_dark, restart_rect)

# 			if pygame.mouse.get_pressed()[0]:
# 				game_over = False
# 				lifeCount = 3
# 				start_time = pygame.time.get_ticks()

# 				title_music.stop()
# 				game_music.play()
# 		else:
# 			win.blit(restart_surf, restart_rect)

# 		pygame.draw.rect(win, 'white', restart_rect.inflate(30,30), 5,3)

# 	# game logic
# 	else:
# 		win.blit(star_bg_surf,(0,0))

# 		score = pygame.time.get_ticks() - start_time
# 		display_score(score)

# 		# display laser
# 		if laser_data:
# 			for laser_dict in laser_data:
# 				laser_dict['rect'].y -= laser_speed * dt
# 				win.blit(laser_surf, laser_dict['rect'])
# 			laser_data = [laser_dict for laser_dict in laser_data if laser_dict['rect'].y > -100]

# 		# display meteor
# 		if meteor_data:
# 			for meteor_dict in meteor_data:
# 				meteor_dict['rect'].center += meteor_dict['direction'] * meteor_dict['speed'] * dt
# 				win.blit(meteor_surf, meteor_dict['rect'])
# 			meteor_data = [meteor_dict for meteor_dict in meteor_data if meteor_dict['rect'].y < 800]

# 		# display spaceship
# 		player_direction = get_input(player_direction)
# 		player_rect.center += player_direction * player_speed * dt
# 		win.blit(player_surf, player_rect)

# 		# collision
# 		if meteor_data:
# 			for meteor_dict in meteor_data:
# 				# player -> meteor
# 				if player_rect.colliderect(meteor_dict['rect']):
# 					meteor_dict['dokill'] = True
# 					lifeCount -= 1
# 					damage_sound.play()
# 					if lifeCount <= 0:
# 						game_over = True
# 						meteor_data = []
# 						laser_data = []
# 						game_music.stop()
# 						title_music.play()
# 				# laser -> meteor
# 				if laser_data:
# 					for laser_dict in laser_data:
# 						if meteor_dict['rect'].colliderect(laser_dict['rect']):
# 							meteor_dict['dokill'] = True
# 							laser_dict['dokill'] = True
# 							explosion_sound.play()
# 			meteor_data = [meteor_dict for meteor_dict in meteor_data if not meteor_dict['dokill']]
# 			laser_data = [laser_dict for laser_dict in laser_data if not laser_dict['dokill']]
# 		# display life
# 		display_lives(lifeCount)
# 	# update frame
# 	pygame.display.update()

# ~~~~~~~~~~~~~~~~~~~~~~~ End of shmup ~~~~~~~~~



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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gradient descent ~~~~~~~~~~~~~~~

# # # We train the neural network through a process of trial and error.
# # # Adjusting the synaptic weights each time.
# # def trainWeights(self, prediction_result, bestOutcome):
# #     '''Calculates the error in thinking and\n
# #     adjusts weights accordingly'''

# #     # Calculate the difference between the desired output and the expected output).
# #     error = bestOutcome - prediction_result

# #     # Multiply the error by the input and again by the gradient of the Sigmoid curve.
# #     # This means less confident weights are adjusted more.
# #     # This means inputs, which are zero, do not cause changes to the weights.
# #     adjustment = dot(prediction_result.T, error * self.__sigmoid_derivative(prediction_result))

# #     # Adjust the weights.
# #     self.synaptic_weights += adjustment
# ~~~~~~~~~~~~~~~~~~~~~~~ AGAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## gradient descent... I didn't end up using. I went with Genetic.
# from numpy import exp, array, random, dot
# training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# training_set_outputs = array([[0, 1, 1, 0]]).T
# random.seed(1)
# synaptic_weights = 2 * random.random((3, 1)) - 1
# for iteration in xrange(10000):
#     output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
#     synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
# print( 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))

# ~~~~~~~~~~~~~~~~~~~~~~~ End of Gradient descent ~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~ Bird code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def fitnessFun(genomes, config):
#     """
#     runs the simulation of the current population of
#     *!(@&$*&#@ and sets their fitness based on the #*@^%&$%&^ they
#     reach in the *@^#%$.
#     """
#     global WIN, gen
#     win = WIN
#     gen += 1

#     # start by creating lists holding the genome itself, the
#     # neural network associated with the genome and the
#     # bird object that uses that network to play
#     nets = []
#     solutions = []
#     ge = []
#     for genome_id, genome in genomes:
#         genome.fitness = 0  # start with fitness level of 0
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         nets.append(net)
#         solutions.append(Bird(230,350))
#         ge.append(genome)

#     score = 0

#     run = True
#     while run and len(solutions) > 0:
#         clock.tick(30)

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False
#                 break

#         pipe_ind = 0
#         if len(solutions) > 0:
#             if len(pipes) > 1 and solutions[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
#                 pipe_ind = 1                                                                 # pipe on the screen for neural network input

#         for x, bird in enumerate(solutions):  # give each bird a fitness of 0.1 for each frame it stays alive
#             ge[x].fitness += 0.1
#             bird.move()

#             # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
#             output = nets[solutions.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

#             if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
#                 bird.jump()

#         base.move()

#         rem = []
#         add_pipe = False
#         for pipe in pipes:
#             pipe.move()
#             # check for collision
#             for bird in solutions:
#                 if pipe.collide(bird, win):
#                     ge[solutions.index(bird)].fitness -= 1
#                     nets.pop(solutions.index(bird))
#                     ge.pop(solutions.index(bird))
#                     solutions.pop(solutions.index(bird))

#             if pipe.x + pipe.PIPE_TOP.get_width() < 0:
#                 rem.append(pipe)

#             if not pipe.passed and pipe.x < bird.x:
#                 pipe.passed = True
#                 add_pipe = True

#         if add_pipe:
#             score += 1
#             # can add this line to give more reward for passing through a pipe (not required)
#             for genome in ge:
#                 genome.fitness += 5
#             pipes.append(Pipe(WIN_WIDTH))

#         for r in rem:
#             pipes.remove(r)

#         for bird in solutions:
#             if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
#                 nets.pop(solutions.index(bird))
#                 ge.pop(solutions.index(bird))
#                 solutions.pop(solutions.index(bird))

#         print(gamestate)

#         # break if score gets large enough
#         '''if score > 20:
#             pickle.dump(nets[0],open("best.pickle", "wb"))
#             break'''
#     quit()

# def run(config_file):
#     """
#     runs the NEAT algorithm to train a neural network to play flappy bird.
#     :param config_file: location of config file
#     :return: None
#     """
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                          config_file)

#     # Create the population, which is the top-level object for a NEAT run.
#     p = neat.Population(config)

#     # Add a stdout reporter to show progress in the terminal.
#     p.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     p.add_reporter(stats)
#     #p.add_reporter(neat.Checkpointer(5))

#     # Run for up to 50 generations.
#     winner = p.run(fitnessFun, 50)

#     # show final stats
#     print('\nBest genome:\n{!s}'.format(winner))


# if __name__ == '__main__':
#     # Determine path to configuration file. This path manipulation is
#     # here so that the script will run successfully regardless of the
#     # current working directory.
#     local_dir = os.path.dirname(__file__)
#     config_path = os.path.join(local_dir, 'config-feedforward.txt')
#     run(config_path)
