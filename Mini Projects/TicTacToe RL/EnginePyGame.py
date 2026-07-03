import numpy as np
import pygame
import sys
 
# ---------------------------------------------------------------------------
# Core game logic (unchanged rules, same as your original Game class)
# ---------------------------------------------------------------------------
class Game():
 
    def start(self):
        self.game = np.zeros([3, 3], dtype=np.int8)
        return "Game Initialized! Player 1 is X and Player 2 is O"
 
    def move(self, square, player):
        if self.game[square[0], square[1]] != 1 and self.game[square[0], square[1]] != 2:
            self.game[square[0], square[1]] = player
            return True
        else:
            return False  # Invalid move, square already occupied
 
    def check(self):
        winningCombinations = [
            [[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]], [[2, 0], [2, 1], [2, 2]],
            [[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]], [[0, 2], [1, 2], [2, 2]],
            [[0, 0], [1, 1], [2, 2]], [[0, 2], [1, 1], [2, 0]]
        ]
 
        for combo in winningCombinations:
            a = self.game[tuple(combo[0])]
            b = self.game[tuple(combo[1])]
            c = self.game[tuple(combo[2])]
            if a != 0 and a == b == c:
                return f"Game Over. Player {a} wins!"
        return None
 
 
# ---------------------------------------------------------------------------
# Pygame front-end
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 600, 700          # extra height at top for status text
BOARD_SIZE = 600
CELL = BOARD_SIZE // 3
LINE_WIDTH = 8
STATUS_HEIGHT = HEIGHT - BOARD_SIZE
 
# Colors
BG_COLOR = (28, 28, 30)
LINE_COLOR = (200, 200, 200)
X_COLOR = (255, 90, 90)
O_COLOR = (90, 170, 255)
TEXT_COLOR = (240, 240, 240)
HOVER_COLOR = (45, 45, 50)
 
 
class PygameGame(Game):
 
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 32)
        self.big_font = pygame.font.SysFont("arial", 48, bold=True)
 
        self.turn = 1
        self.status = self.start()
        self.game_over_msg = None
 
    # -------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------
    def draw_board(self):
        self.screen.fill(BG_COLOR)
 
        # Status bar background
        pygame.draw.rect(self.screen, (18, 18, 20), (0, 0, WIDTH, STATUS_HEIGHT))
 
        # Hover highlight (only while game is still going)
        if self.game_over_msg is None:
            mx, my = pygame.mouse.get_pos()
            if my > STATUS_HEIGHT:
                col = mx // CELL
                row = (my - STATUS_HEIGHT) // CELL
                if 0 <= row < 3 and 0 <= col < 3 and self.game[row, col] == 0:
                    rect = (col * CELL, STATUS_HEIGHT + row * CELL, CELL, CELL)
                    pygame.draw.rect(self.screen, HOVER_COLOR, rect)
 
        # Grid lines
        for i in range(1, 3):
            # vertical
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (i * CELL, STATUS_HEIGHT), (i * CELL, HEIGHT),
                LINE_WIDTH
            )
            # horizontal
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (0, STATUS_HEIGHT + i * CELL), (WIDTH, STATUS_HEIGHT + i * CELL),
                LINE_WIDTH
            )
 
        # X's and O's
        for row in range(3):
            for col in range(3):
                val = self.game[row, col]
                cx = col * CELL + CELL // 2
                cy = STATUS_HEIGHT + row * CELL + CELL // 2
                if val == 1:
                    self.draw_x(cx, cy)
                elif val == 2:
                    self.draw_o(cx, cy)
 
        # Status text
        text = self.game_over_msg if self.game_over_msg else self.status
        color = TEXT_COLOR
        rendered = self.font.render(text, True, color)
        rect = rendered.get_rect(center=(WIDTH // 2, STATUS_HEIGHT // 2))
        self.screen.blit(rendered, rect)
 
        pygame.display.flip()
 
    def draw_x(self, cx, cy):
        pad = CELL // 4
        pygame.draw.line(self.screen, X_COLOR,
                          (cx - pad, cy - pad), (cx + pad, cy + pad), 10)
        pygame.draw.line(self.screen, X_COLOR,
                          (cx + pad, cy - pad), (cx - pad, cy + pad), 10)
 
    def draw_o(self, cx, cy):
        radius = CELL // 4
        pygame.draw.circle(self.screen, O_COLOR, (cx, cy), radius, 10)
 
    # -------------------------------------------------------------
    # Input handling
    # -------------------------------------------------------------
    def handle_click(self, pos):
        if self.game_over_msg is not None:
            return
 
        mx, my = pos
        if my <= STATUS_HEIGHT:
            return  # clicked in the status bar
 
        col = mx // CELL
        row = (my - STATUS_HEIGHT) // CELL
        if not (0 <= row < 3 and 0 <= col < 3):
            return
 
        success = self.move([row, col], self.turn)
        if not success:
            self.status = "Invalid move. Square already occupied."
            return
 
        result = self.check()
        if result:
            self.game_over_msg = result
            return
 
        if np.all(self.game != 0):
            self.game_over_msg = "It's a draw!"
            return
 
        self.turn = 3 - self.turn
        self.status = f"Player {self.turn}'s turn ({'X' if self.turn == 1 else 'O'})"
 
    def reset(self):
        self.turn = 1
        self.status = self.start()
        self.status = f"Player {self.turn}'s turn (X)"
        self.game_over_msg = None
 
    # -------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------
    def run(self):
        self.status = f"Player {self.turn}'s turn (X)"
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
 
            self.draw_board()
 
            if self.game_over_msg:
                # append hint to restart, drawn as part of status text update
                pass
 
            self.clock.tick(60)
 
        pygame.quit()
        sys.exit()
 
 
if __name__ == "__main__":
    game = PygameGame()
    game.run()