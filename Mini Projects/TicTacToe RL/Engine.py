import numpy as np

class Game():

    def start(self):
        self.game = np.zeros([3,3], dtype=np.int8)
        return "Game Initialized! Player 1 is X and Player 2 is O"

    def move(self, square, player):
        if self.game[square[0], square[1]] != 1 and self.game[square[0], square[1]] != 2:
            self.game[square[0], square[1]] = player
        else:
            return "Invalid Move. Square is already occupied."
        
    def check(self):
        winningCombinations = [[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]], [[2, 0], [2, 1], [2, 2]], [[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]], [[0, 2], [1, 2], [2, 2]], [[0, 0], [1, 1], [2, 2]], [[0, 2], [1, 1], [2, 0]]]
        
        for combo in winningCombinations:
            a = self.game[tuple(combo[0])]
            b = self.game[tuple(combo[1])]
            c = self.game[tuple(combo[2])]
            if a != 0 and a == b == c:
                return f"Game Over. Player {a} wins!"
    
class CLIGame(Game):
    def printBoard(self):
        board = self.game.flatten().astype(object)
        for i in range(0,9):
            if board[i] == 1:
                board[i] = "X"
            elif board[i] == 2:
                board[i] = "O"
            else:
                board[i] = " "

        print(f" {board[0]} | {board[1]} | {board[2]} ")
        print("---+---+---")
        print(f" {board[3]} | {board[4]} | {board[5]} ")
        print("---+---+---")
        print(f" {board[6]} | {board[7]} | {board[8]} ")
    
    def callmove(self, turn):
        move = int(input(f"Select your square player {turn} (1-9): "))
        if move not in range(1, 10):
            return "Invalid move. Please pick a number in range 1-9"
        if move%3==0:
            column = 2
        elif move%3==1:
            column = 0
        elif move%3==2:
            column = 1
        
        if move in range(1,4):
            row = 0
        elif move in range(4, 7):
            row = 1
        else:
            row = 2
        
        
        sq = [row, column]
        self.move(sq, turn)


# game loop
turn = 1
game = CLIGame()
print(game.start())

while True:
    game.printBoard()
    game.callmove(turn)

    result = game.check()
    if result:
        game.printBoard()
        print(result)
        break

    if np.all(game.game != 0):
        game.printBoard()
        print("It's a draw!")
        break

    turn = 3 - turn 
        