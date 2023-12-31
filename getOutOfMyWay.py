


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

