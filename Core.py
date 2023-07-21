# Obtain settings regarding the size/position of the game window.
# call a selected environment from the environments folder
# The Simulation initializes an Environment, checking it's maxPopulation.
# load a nn from text file or pickle or something
# The Sim loads the population of availiable Genomes.
# The Sim adds all players to the Environment, starting with User Agents followed by NN Agents.
# The Sim selects from the population of Genomes based on the Env's max population.
#

# the Simulation communicates between Agent and Environment.
# because like the Agent and the Environment are seperate...

# Then Generates the next population of availiable Genomes.


import pygame, sys
from random import randint
from random import choice

import pygame, os
from sys import exit
from random import randint, uniform


### objects that need defenition
# GameObj
# Agent
    # User
    # NeuNet
# Genome
    # Gene?




class Sim:
    def __init__(self, game:'class') -> None:
        '''Initialize an environment for this simulation'''
        self.environment = game(players)## set to a user defined class which imports from Env
        self.envHistory = dict()### a place to store kept Environments by a name in str and list of settings?
    def advance(self, actions:'list[int]', envRules:'function'):
        '''Each Agent is always allowed one action per timestep.\n
        It is important to maintain the order of Agent actions.'''
        return envRules(actions)### make sure actions.len() can be varied between timesteps.(as long as len(actions) == len(agents))
    def setEnvironment(self,newEnv,keepEnv:'bool'=False):
        '''Set a new environment for this simulation
        (Can be incredibly deadly for Agent objects...)'''
        ### keepEnv
        if keepEnv:
            pass### ugh
        self.environment = newEnv



# class MyNeuralNet():
#     def __init__(self) -> None:
#         pass


class Genome:
    def __init__(self) -> None:
        pass

class Agent:# Recieves rewards and observations, and returns an action
    def __init__(self, geneSeq:'Genome'=None) -> None:
        self.memory = {}
        if geneSeq:### Build a NN from a Genome to handle the object.
            pass
        else:# That there's a Human, probably.
            pass### Make the object recieve input from input devices( wait on the User)



# class Simulation:
#     def __init__(self, environment:'Env') -> None:
#         self.environment = environment
#         self.running = False
#     def advance(self):
#         pass
#     def play(self):
#         # repetedly call advance
#         while self.running:
#             self.advance()
#         pass




# class SpriteSheet():
#     def __init__(self, filename:'str'):
#         try:
#             self.sheet = pygame.image.load(os.path.join(img_dir, filename)).convert()
#             self.filename = filename
#         except pygame.error as e:
#             print(f"Unable to load spritesheet image: {filename}")
#             raise SystemExit(e)
#     def image_at(self, rectangle, colorkey = None):# Load the image from x, y, x+offset, y+offset.
#         rect = pygame.Rect(rectangle)
#         image = pygame.Surface(rect.size).convert()
#         image.blit(self.sheet, (0, 0), rect)
#         if colorkey is not None:
#             if colorkey == -1:
#                 colorkey = image.get_at((0,0))
#             image.set_colorkey(colorkey, pygame.RLEACCEL)
#         return image
#     def images_at(self, rects, colorkey = None):#Load a whole bunch of images and return them as a list.
#         return [self.image_at(rect, colorkey) for rect in rects]
#     def load_strip(self, rect, image_count, colorkey = None):#Load a whole strip of images, and return them as a list.
#         tups = [(rect[0]+rect[2]*x, rect[1], rect[2], rect[3])
#                 for x in range(image_count)]
#         return self.images_at(tups, colorkey)
#     def load_grid_images(self, num_rows, num_cols, x_margin=0, x_padding=0, y_margin=0, y_padding=0):#Load a grid of images. Assumes symmetrical padding on left and right. Calls self.images_at() to get list of images.
#         sheet_rect = self.sheet.get_rect()
#         sheet_width, sheet_height = sheet_rect.size
#         x_sprite_size = ( sheet_width - 2 * x_margin - (num_cols - 1) * x_padding ) / num_cols# To calculate the size of each sprite, subtract the two margins,
#         y_sprite_size = ( sheet_height - 2 * y_margin - (num_rows - 1) * y_padding ) / num_rows# and the padding between each row, then divide by num_cols.
#         sprite_rects = []
#         for row_num in range(num_rows):# Position of sprite rect is margin + one sprite size
#             for col_num in range(num_cols):# and one padding size for each row. Same for y.
#                 x = x_margin + col_num * (x_sprite_size + x_padding)
#                 y = y_margin + row_num * (y_sprite_size + y_padding)
#                 sprite_rect = (x, y, x_sprite_size, y_sprite_size)
#                 sprite_rects.append(sprite_rect)
#         grid_images = self.images_at(sprite_rects)
#         print(f"Loaded {len(grid_images)} images from {self.filename[:-20]}.")
#         return grid_images
#     def fileNameQuery(self):
#         return self.filename
# def loadSSheets():
#     '''
#     make a sprite sheet object for each file listed
#     where the file names determine how the image is disected.
#     '''
#     sheets = {}
#     for ssFileName in SSFN:
#         t = SpriteSheet(ssFileName)
#         num_rows = int(ssFileName[-20:-18])
#         num_cols = int(ssFileName[-18:-16])
#         x_margin = int(ssFileName[-16:-13])
#         x_padding = int(ssFileName[-13:-10])
#         y_margin = int(ssFileName[-10:-7])
#         y_padding = int(ssFileName[-7:-4])
#         print(num_rows, num_cols, x_margin, x_padding, y_margin, y_padding, ssFileName[:-20])
#         gridimg = t.load_grid_images(num_rows, num_cols, x_margin, x_padding, y_margin, y_padding)
#         sheets[ssFileName[:-20]] = gridimg
#     return sheets
# SSFN = []# Buttons0212000000000000.png, GUI1815004004004004.png
# allSpritesheets = loadSSheets()

def get_input(vector:'pygame.math.Vector2'):
    keys = pygame.key.get_pressed()
    vector.update((0,0))# direction reset to 0. If you ain't pressin anythin, y'ain't goin' anywhere.#####
    if keys[pygame.K_RIGHT]:
        vector.x = 1
    if keys[pygame.K_LEFT]:
        vector.x = -1
    if keys[pygame.K_DOWN]:
        vector.y = 1
    if keys[pygame.K_UP]:
        vector.y = -1
    return vector.normalize() if vector.magnitude() != 0 else pygame.math.Vector2()# excuse me, what?# okay so, Normalize the vector UNLESS the vectors magnitude is 0. then just make an empty vector


def display_score(time_passed):
		# text data
		text_surf = font.render(str(time_passed // 1000), False, 'white')
		text_rect = text_surf.get_rect(center = (SCREEN_SIZE[0] / 2, 100))

		# score display
		win.blit(text_surf, text_rect)

		# frame
		pygame.draw.rect(win, 'white', text_rect.inflate(32,32),4,5)

def display_lives(num_lifes):
	for life in range(num_lifes):
		x = 20 + life * (icon_surf.get_width() + 4) 
		y = SCREEN_SIZE[1] - 20
		icon_rect = icon_surf.get_rect(bottomleft = (x,y))
		win.blit(icon_surf, icon_rect)

# init
pygame.mixer.pre_init(44100, 16, 2, 4096) #frequency, size, channels, buffersize
pygame.init() #turn all of pygame on.
SCREEN_SIZE = (1280,720)
win = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Spaceship')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 40)
img_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ticTacToe\\img')##
sfx_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sfx')


# load graphics
img_loading_initial = pygame.image.load(os.path.join(img_dir, "loading.jpg")).convert()
current_background = pygame.transform.scale(img_loading_initial, SCREEN_SIZE)
win.blit(current_background, win.get_rect())
img_background_menu = pygame.image.load(os.path.join(img_dir, "ColorfulHorizon.jpg")).convert()

star_bg_surf = pygame.image.load(os.path.join(img_dir, 'star_bg.png')).convert_alpha()
player_surf = pygame.image.load(os.path.join(img_dir, 'player.png')).convert_alpha()
laser_surf = pygame.image.load(os.path.join(img_dir, 'laser.png')).convert_alpha()
meteor_surf = pygame.image.load(os.path.join(img_dir, 'meteor.png')).convert_alpha()
icon_surf = pygame.image.load(os.path.join(img_dir, 'icon.png')).convert_alpha()
broken_surf = pygame.image.load(os.path.join(img_dir, 'broken.png')).convert_alpha()

# bg stars

# spaceship 
player_rect = player_surf.get_rect(center = (640,360))## was wrapped in an 'pygame.frect()' 

player_direction = pygame.math.Vector2()
player_speed = 300

# laser 
laser_speed = 500
laser_data = []

# meteor 
meteor_data = []
meteor_timer = pygame.USEREVENT
pygame.time.set_timer(meteor_timer, 1500)

# life system
lifeCount = 3

# score system
score = 0
start_time = 0
game_over = False

# title
broken_rect = broken_surf.get_rect(center = (SCREEN_SIZE[0]/2, SCREEN_SIZE[1]/2 - 100))
restart_surf = font.render('Restart', True, 'White')
restart_surf_dark = font.render('Restart', True, '#3a2e3f')
restart_rect = restart_surf.get_rect(center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - 100))

# audio
game_music = pygame.mixer.Sound(os.path.join(sfx_dir, 'game_music.wav'))### path.join
title_music = pygame.mixer.Sound(os.path.join(sfx_dir, 'title_music.wav'))
laser_sound = pygame.mixer.Sound(os.path.join(sfx_dir, 'laser.wav'))
explosion_sound = pygame.mixer.Sound(os.path.join(sfx_dir, 'explosion.wav'))
damage_sound = pygame.mixer.Sound(os.path.join(sfx_dir, 'explosion.wav'))
damage_sound.set_volume(0.5)

game_music.play()

while True:
	# get delta time 
	dt = clock.tick() / 1000

	# event loop
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			exit()
		if not game_over:
			if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
				laser_rect = pygame.rect.Rect(laser_surf.get_rect(midbottom = player_rect.center - pygame.math.Vector2(0,30)))
				laser_data.append({'rect':laser_rect, 'dokill': False})
				laser_sound.play()
			if event.type == meteor_timer:
				x,y  = randint(-100, SCREEN_SIZE[0] -100), randint(-300,-100)
				meteor_rect = pygame.Rect(meteor_surf.get_rect(center = (x, y)))
				meteor_direction = pygame.math.Vector2(randint(-3,3),2)
				meteor_speed = randint(300,600)
				meteor_data.append({'rect': meteor_rect, 'direction': meteor_direction, 'speed': meteor_speed, 'dokill': False})

	# bg color 
	win.fill('#3a2e3f')

	# title screen
	if game_over:
		win.blit(broken_surf,broken_rect)	
			
		# text 
		text_surf = font.render(f'Your score: {score}', True, 'White')
		text_rect = text_surf.get_rect(center = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2 + 50))
		win.blit(text_surf, text_rect)

		# button 
		if restart_rect.collidepoint(pygame.mouse.get_pos()):
			pygame.draw.rect(win, 'white', restart_rect.inflate(30,30),0,3)
			win.blit(restart_surf_dark, restart_rect)

			if pygame.mouse.get_pressed()[0]:
				game_over = False
				lifeCount = 3
				start_time = pygame.time.get_ticks()

				title_music.stop() 
				game_music.play()
		else:
			win.blit(restart_surf, restart_rect)

		pygame.draw.rect(win, 'white', restart_rect.inflate(30,30), 5,3)
	
	# game logic
	else:
		win.blit(star_bg_surf,(0,0))

		score = pygame.time.get_ticks() - start_time
		display_score(score)

		# display laser 
		if laser_data:
			for laser_dict in laser_data:
				laser_dict['rect'].y -= laser_speed * dt
				win.blit(laser_surf, laser_dict['rect'])
			laser_data = [laser_dict for laser_dict in laser_data if laser_dict['rect'].y > -100]

		# display meteor 
		if meteor_data:
			for meteor_dict in meteor_data:
				meteor_dict['rect'].center += meteor_dict['direction'] * meteor_dict['speed'] * dt
				win.blit(meteor_surf, meteor_dict['rect'])
			meteor_data = [meteor_dict for meteor_dict in meteor_data if meteor_dict['rect'].y < 800]

		# display spaceship
		player_direction = get_input(player_direction)
		player_rect.center += player_direction * player_speed * dt
		win.blit(player_surf, player_rect)

		# collision
		if meteor_data:
			for meteor_dict in meteor_data:
				# player -> meteor  
				if player_rect.colliderect(meteor_dict['rect']):
					meteor_dict['dokill'] = True
					lifeCount -= 1
					damage_sound.play()
					if lifeCount <= 0:
						game_over = True
						meteor_data = []
						laser_data = []
						game_music.stop()
						title_music.play()
				# laser -> meteor
				if laser_data:
					for laser_dict in laser_data:
						if meteor_dict['rect'].colliderect(laser_dict['rect']):
							meteor_dict['dokill'] = True
							laser_dict['dokill'] = True
							explosion_sound.play()
			meteor_data = [meteor_dict for meteor_dict in meteor_data if not meteor_dict['dokill']]
			laser_data = [laser_dict for laser_dict in laser_data if not laser_dict['dokill']]
		# display life
		display_lives(lifeCount)
	# update frame 
	pygame.display.update()






	
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


