import math

import pygame
# from minihex.HexGame import player
import numpy as np
from enum import IntEnum


# Define the colors we will use in RGB format
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (40, 40, 40)
LIGHT = (220, 220, 220)

STRAWBERRY = (251, 41, 67)
AZURE = (6, 154, 243)
BURGUNDY = (137, 0, 55)
ROYAL_BLUE = (10, 10, 200)

class player(IntEnum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2

def _get_colors(dark_mode: bool):
    return {
        'DARK_MODE': dark_mode,
        'BACKGROUND': DARK if dark_mode else WHITE,
        'LINES': LIGHT if dark_mode else BLACK,
        'PLAYER_1': BURGUNDY if dark_mode else STRAWBERRY,
        'PLAYER_2': ROYAL_BLUE if dark_mode else AZURE
    }

# TODO
# Passe Fenster richtig an, wahrscheinlich wird der Cursor nicht richtig
# relativ entdeckt und deswegen kommt keine Rückmeldung

class Gui:
    def __init__(self, board, radius, dark_mode=False):
        self.r = radius
        self.size = [int(self.r * (3 / 2 * board.shape[0] + 3)) + 200, int(self.r * (3 ** (1 / 2) / 2 * board.shape[0] + 3)) + 100]
        self.editor_mode = False  # AI will not move in editor mode
        self.colors = _get_colors(dark_mode)
        self.show_field_text = False
        self.last_field_text = None
        self.winner_text = None

        pygame.init()

        # Set the height and width of the screen
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("HexHex")

        self.clock = pygame.time.Clock()
        # distance of neighboring hexagons
        self.board = board

        pygame.font.init()
        font_path = "fonts/FallingSky-JKwK.otf"
        self.font = pygame.font.Font(font_path, int(radius / 3))
        self.font_large = pygame.font.Font(font_path, int(radius / 2.5))
        self.update_board(board)

    def toggle_colors(self):
        self.colors = _get_colors(not self.colors['DARK_MODE'])

    def quit(self):
        # Be IDLE friendly
        pygame.quit()
        exit(0)

    def pixel_to_pos(self, pixel):
        positions = [(x, y) for x in range(self.board.shape[0]+1) for y in range(self.board.shape[0]+1)]
        def squared_distance(pos1, pos2):
            return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
        # print(positions)
        centers = [(position, self.get_center(position)) for position in positions]
        return min(centers, key=lambda x: squared_distance(x[1], pixel))[0] 

    def get_center(self, pos):
        x = pos[0]
        y = pos[1]
        return [2*self.r + x * self.r / 2 + y * self.r, 2*self.r + math.sqrt(3) / 2 * x * self.r]

    def update_field_text(self, field_text):
        self.last_field_text = field_text

    def update_board(self, board):
        # prepare board
        board = np.pad(board, 1)
        board[:, 0] = player.WHITE
        board[0, :] = player.BLACK
        board[:, -1] = player.WHITE
        board[-1, :] = player.BLACK
        
        print("update")
        # Clear the screen and set the screen background
        self.screen.fill(self.colors['BACKGROUND'])

        text = f"""    a: trigger ai move
{'✓' if self.editor_mode else '   '} e: human vs human mode
{'✓' if self.show_field_text else '   '} s: toggle ai ratings
    z: undo last move
{'✓' if self.colors["DARK_MODE"] else '   '} d: toggle dark mode
    r: restart game"""
        blit_text(self.screen, text, (self.size[0] - 200, 10), self.font, self.colors['LINES'])

        for x in range(board.shape[0]):
            for y in range(board.shape[0]):
                if x in [0, board.shape[0]] and y in [0, board.shape[0]]: # corners
                    continue  # don't draw borders as they don't belong to a single player
                center = self.get_center([x, y])
                angles = [math.pi / 6 + x * math.pi / 3 for x in range(6)]
                points = [[center[0] + math.cos(angle) * self.r / math.sqrt(3),
                           center[1] + math.sin(angle) * self.r / math.sqrt(3)]
                          for angle in angles]
                # breakpoint()
                # print(x,y)
                if board[x, y] == player.BLACK:
                    pygame.draw.polygon(self.screen, self.colors['PLAYER_1'], points, 0)
                elif board[x, y] == player.WHITE:
                    pygame.draw.polygon(self.screen, self.colors['PLAYER_2'], points, 0)
                pygame.draw.polygon(self.screen, self.colors['LINES'], points, 3)

                if self.last_field_text is not None and self.show_field_text:
                    field_text_pos = board.player * (x * board.size + y) + \
                                     (1 - board.player) * (y * board.size + x)
                    if x in range(board.size) and y in range(board.size):
                        text = self.last_field_text[field_text_pos]
                        textsurface = self.font.render(f'{text}', True, self.colors['LINES'])
                        text_size = self.font.size(text)
                        self.screen.blit(textsurface, (center[0] - text_size[0] // 2,
                                                       center[1] - text_size[1] // 2))

        # if self.board.switch:
        #     blit_text(self.screen, "switched!", (self.size[0] - 180, self.size[1] - 100), self.font_large, self.colors['LINES'])

        # if self.board.winner:
        #     blit_text(self.screen, self.winner_text, (self.size[0] - 180, self.size[1] - 50), self.font_large, self.colors['LINES'])

        # Go ahead and update the screen with what we've drawn.
        # This MUST happen after all the other drawing commands.
        pygame.display.flip()

    def wait_for_pressing_r(self):
        while True:
            self.clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # If user clicked close
                    self.quit()
                    exit(0)
                if event.type == pygame.KEYDOWN and event.unicode == 'r':
                    return

    def get_move(self):
        while True:
            # This limits the while loop to a max of 10 times per second.
            # Leave this out and we will use all CPU we can.
            self.clock.tick(10)
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    self.quit()
                    exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    print("click")
                    print(self.pixel_to_pos(event.pos))
                    return self.pixel_to_pos(event.pos)
                if event.type == pygame.KEYDOWN and event.unicode == 'd':
                    self.toggle_colors()
                    return 'redraw'
                if event.type == pygame.KEYDOWN and event.unicode == 'a':
                    return 'ai_move'
                if event.type == pygame.KEYDOWN and event.unicode == 'z':
                    return 'undo_move'
                if event.type == pygame.KEYDOWN and event.unicode == 's':
                    return 'show_ratings'
                if event.type == pygame.KEYDOWN and event.unicode == 'r':
                    return 'restart'
                if event.type == pygame.KEYDOWN and event.unicode == 'e':
                    self.editor_mode = not self.editor_mode
                    return 'redraw'

    def set_winner(self, winner_text):
        self.winner_text = winner_text


# From https://stackoverflow.com/questions/42014195/rendering-text-with-multiple-lines-in-pygame/42015712
def blit_text(surface, text, pos, font, color=pygame.Color('black')):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, True, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.
