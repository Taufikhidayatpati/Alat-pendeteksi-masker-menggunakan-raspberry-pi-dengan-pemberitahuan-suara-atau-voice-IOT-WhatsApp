#Pemutar Audio menggunaka Pygame

import pygame

pygame.mixer.init()
pygame.mixer.music.load("TIDAK.wav")
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
      continue
