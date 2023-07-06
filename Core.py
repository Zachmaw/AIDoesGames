# Obtain settings regarding the size/position of the game window.
# load a network from text file or pickle or something
#

import pygame, os
from sys import exit
from random import randint, uniform


# class MyNeuralNet():
#     def __init__(self) -> None:
#         pass




class SpriteSheet():
    def __init__(self, filename):
        try:
            self.sheet = pygame.image.load(os.path.join(img_dir, filename)).convert()
            self.filename = filename
        except pygame.error as e:
            print(f"Unable to load spritesheet image: {filename}")
            raise SystemExit(e)
    def image_at(self, rectangle, colorkey = None):# Load the image from x, y, x+offset, y+offset.
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size).convert()
        image.blit(self.sheet, (0, 0), rect)
        if colorkey is not None:
            if colorkey == -1:
                colorkey = image.get_at((0,0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)
        return image
    def images_at(self, rects, colorkey = None):#Load a whole bunch of images and return them as a list.
        return [self.image_at(rect, colorkey) for rect in rects]
    def load_strip(self, rect, image_count, colorkey = None):#Load a whole strip of images, and return them as a list.
        tups = [(rect[0]+rect[2]*x, rect[1], rect[2], rect[3])
                for x in range(image_count)]
        return self.images_at(tups, colorkey)
    def load_grid_images(self, num_rows, num_cols, x_margin=0, x_padding=0, y_margin=0, y_padding=0):#Load a grid of images. Assumes symmetrical padding on left and right. Calls self.images_at() to get list of images.
        sheet_rect = self.sheet.get_rect()
        sheet_width, sheet_height = sheet_rect.size
        x_sprite_size = ( sheet_width - 2 * x_margin - (num_cols - 1) * x_padding ) / num_cols# To calculate the size of each sprite, subtract the two margins,
        y_sprite_size = ( sheet_height - 2 * y_margin - (num_rows - 1) * y_padding ) / num_rows# and the padding between each row, then divide by num_cols.
        sprite_rects = []
        for row_num in range(num_rows):# Position of sprite rect is margin + one sprite size
            for col_num in range(num_cols):# and one padding size for each row. Same for y.
                x = x_margin + col_num * (x_sprite_size + x_padding)
                y = y_margin + row_num * (y_sprite_size + y_padding)
                sprite_rect = (x, y, x_sprite_size, y_sprite_size)
                sprite_rects.append(sprite_rect)
        grid_images = self.images_at(sprite_rects)
        print(f"Loaded {len(grid_images)} images from {self.filename[:-20]}.")
        return grid_images
    def fileNameQuery(self):
        return self.filename



def get_input(vector):
	keys = pygame.key.get_pressed()
	vector.update((0,0))
	if keys[pygame.K_RIGHT]:
		vector.x = 1
	elif keys[pygame.K_LEFT]:
		vector.x = -1
	if keys[pygame.K_DOWN]:
		vector.y = 1
	elif keys[pygame.K_UP]:
		vector.y = -1
	return vector.normalize() if vector.magnitude() != 0 else pygame.math.Vector2()

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

def loadSSheets():# make a sprite sheet object for each file listed
    sheets = {}
    for ssFileName in SSFN:
        t = SpriteSheet(ssFileName)
        num_rows = int(ssFileName[-20:-18])
        num_cols = int(ssFileName[-18:-16])
        x_margin = int(ssFileName[-16:-13])
        x_padding = int(ssFileName[-13:-10])
        y_margin = int(ssFileName[-10:-7])
        y_padding = int(ssFileName[-7:-4])
        print(num_rows, num_cols, x_margin, x_padding, y_margin, y_padding, ssFileName[:-20])
        gridimg = t.load_grid_images(num_rows, num_cols, x_margin, x_padding, y_margin, y_padding)
        sheets[ssFileName[:-20]] = gridimg
    return sheets

# init
pygame.init()
SCREEN_SIZE = (1280,720)
win = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Spaceship')
clock = pygame.time.Clock()
font = pygame.font.Font(None, 40)
img_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'img')
SSFN = []


# load graphics
img_loading_initial = pygame.image.load(os.path.join(img_dir, "loading.jpg")).convert()
current_background = pygame.transform.scale(img_loading_initial, SCREEN_SIZE)
win.blit(current_background, win.get_rect())
img_background_menu = pygame.image.load(os.path.join(img_dir, "ColorfulHorizon.jpg")).convert()
allSpritesheets = loadSSheets()#####
#####
star_bg_surf = pygame.image.load('../environments/graphics/star_bg.png').convert_alpha()
player_surf = pygame.image.load('../environments/graphics/player.png').convert_alpha()
laser_surf = pygame.image.load('../environments/graphics/laser.png').convert_alpha()
meteor_surf = pygame.image.load('../environments/graphics/meteor.png').convert_alpha()
icon_surf = pygame.image.load('../environments/graphics/icon.png').convert_alpha()
broken_surf = pygame.image.load('../environments/graphics/broken.png').convert_alpha()

# bg stars

# spaceship 
player_rect = pygame.FRect(player_surf.get_rect(center = (640,360)))
player_direction = pygame.math.Vector2()
player_speed = 400

# laser 
laser_speed = 600
laser_data = []

# meteor 
meteor_data = []
meteor_timer = pygame.USEREVENT + 1
pygame.time.set_timer(meteor_timer, 150)

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
game_music = pygame.mixer.Sound('../audio/game_music.wav')
title_music = pygame.mixer.Sound('../audio/title_music.wav')
laser_sound = pygame.mixer.Sound('../audio/laser.wav')
explosion_sound = pygame.mixer.Sound('../audio/explosion.wav')
damage_sound = pygame.mixer.Sound('../audio/damage.ogg')

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
				laser_frect = pygame.FRect(laser_surf.get_rect(midbottom = player_rect.center - pygame.math.Vector2(0,30)))
				laser_data.append({'rect':laser_frect, 'dokill': False})
				laser_sound.play()
			if event.type == meteor_timer:
				x,y  = randint(-100, SCREEN_SIZE[0] -100), randint(-300,-100)
				meteor_frect = pygame.FRect(meteor_surf.get_rect(center = (x, y)))
				meteor_direction = pygame.math.Vector2(uniform(0.1,0.4),1)
				meteor_speed = randint(300,600)
				meteor_data.append({'rect': meteor_frect, 'direction': meteor_direction, 'speed': meteor_speed, 'dokill': False})

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