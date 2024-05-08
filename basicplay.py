import chess
import chess.svg
import chess
import json


class TreeNode:
    def __init__(self, board, turn, parent=None):
        if isinstance(board, str):
            self.board = chess.Board(board)  # Convert FEN string to chess.Board
        else:
            self.board = board
        self.turn = turn
        self.parent = parent
        self.children = []
        self.value = None

    def add_child(self, board, turn):
        """Create a new TreeNode for the child and adds it to this node's children."""
        child_node = TreeNode(board, turn, self)
        self.children.append(child_node)
    
    def is_leaf(self):
        """Returns True if this node is a leaf node (no children)."""
        return len(self.children) == 0

    def __repr__(self):
        """A helper method to output information about the node for debugging."""
        return f"TreeNode(board={self.board}, turn={self.turn})"




class AlphaBetaChessTree:
    def __init__(self, fen):
        """
        Initializes an AlphaBetaChessTree object with a board state.
        The board state is represented in FEN (Forsyth-Edwards Notation) format.
        :param fen: A string representing the chess board in FEN format.
        """
        self.root = TreeNode(fen, 'w' if fen.split()[1] == 'w' else 'b') # Initialize the root node.


    @staticmethod
    def get_supported_evaluations():
        """
        Static method that returns a list of supported evaluation methods.
        :return: A list of strings containing supported evaluation methods.
        """
        return ["material_value", "piece_value", "position_value"]

    def _apply_move(self, move, node, notation="SAN"):
        """
        Applies a chess move to a given game state (node).
        :param move: The move to be applied.
        :param node: The game state to which the move is applied.
        :param notation: The notation system used for the move (default: "SAN" - Standard Algebraic Notation).
        """
        new_board = node.board.copy()  # Make a copy of the board to apply the move.
        try:
            if notation == "SAN":
                move = new_board.parse_san(move)  # Parse the move in SAN format.
            elif notation == "UCI":
                move = chess.Move.from_uci(move)
            new_board.push(move)  # Apply the move to the board.
        except ValueError:
            print("Invalid move:", move)
            return None
        
        #Determining the turn of the next player
        next_turn = 'b' if node.turn == 'w' else 'w'
        new_node = TreeNode(new_board, next_turn, parent=node)  # Create a new node for the new board state.
        node.add_child(new_board, next_turn)  # Add the new node as a child of the current node.
        return new_node
    

    def _get_legal_moves(self, node, notation="SAN"):
        """
        Returns a list of all legal moves from the given game state (node).
        :param node: The game state from which to get legal moves.
        :param notation: The notation system used for the moves (default: "SAN").
        :return: A list of strings representing all legal moves for a given node.
        """
        moves = list(node.board.legal_moves)  # Get all legal moves from the board.
        if notation == "SAN":
            return [node.board.san(move) for move in moves]
        elif notation == "UCI":
            return [move.uci() for move in moves]
        return moves
    
    def get_best_next_move(self, node, depth, notation="SAN"):
        """
        Determines the best next move for the current player using the Alpha-Beta pruning algorithm.
        :param node: The current game state.
        :param depth: The depth of the search tree to explore.
        :param notation: The notation system for the move (default: "SAN").
        :return: The best next move in the format defined by the variable notation.
        """
        best_value = float('-inf') 
        best_move = None

        for move in self._get_legal_moves(node, notation):
            #Create a new node for the move
            new_node = self._apply_move(move, node, notation)
            if new_node is None:
                continue # Skip invalid moves

            #Evaluate the move using alpha-beta pruning
            value, _ = self._alpha_beta(new_node, depth-1, float('-inf'), float('inf'), False)

            #Update the best move if needed
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def _alpha_beta(self, node, depth, alpha, beta, maximizing_player):
        """
        The Alpha-Beta pruning algorithm implementation. This method is used to evaluate game positions.
        :param node: The current node (game state).
        :param depth: The depth of the tree to explore.
        :param alpha: The alpha value for the Alpha-Beta pruning.
        :param beta: The beta value for the Alpha-Beta pruning.
        :param maximizing_player: Boolean indicating if the current player is maximizing or minimizing the score.
        :return: The best score for the current player.
        """
        for move in self._get_legal_moves(node, "SAN"):
            #Create a new node for the move
            new_node = self._apply_move(move, node, "SAN")
            if new_node is None:
                continue # Skip invalid moves
        
        if depth == 0 or node.is_leaf():
            return self._evaluate_position(node, depth), None
        
        if maximizing_player:
            value = float('-inf')
            best_move = None
            for child in node.children:
                child_value, _ = self._alpha_beta(child, depth-1, alpha, beta, False)
                if child_value > value:
                    value = child_value
                    best_move = child
                alpha = max(alpha, value)
                if alpha > beta:
                    break # Beta cut-off
            return value, best_move
        if not maximizing_player:
            value = float('inf')
            best_move = None
            for child in node.children:
                child_value, _ = self._alpha_beta(child, depth-1, alpha, beta, True)
                if child_value < value:
                    value = child_value
                    best_move = child
                beta = min(beta, value)
                if alpha > beta:
                    break # Alpha cut-off
            return value, best_move
        

    def _evaluate_position(self, node, depth):
        """
        Evaluates the position at a given node, taking into account the depth of the node in the decision tree.
        :param node: The game state to evaluate.
        :param depth: The depth of the node in the game tree.
        :return: An evaluation score for the position.
        """
        board = node.board
        if board.is_checkmate():
            return float('inf') if board.turn == chess.BLACK else float('-inf')
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        #Material value weights
        piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
                        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000}


        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                score += piece_values[piece.symbol()]
        
        # Evaluate mobility 
        mobility = len(list(board.legal_moves))
        score += mobility if board.turn == chess.WHITE else -mobility

        # Evaluate king safety
        
        king_safety = self._evaluate_king_safety(board, chess.WHITE) - self._evaluate_king_safety(board, chess.BLACK)
        score += king_safety

        #Center control

        center_control = [chess.D4, chess.E4, chess.D5, chess.E5]
        center_score = sum(1 for square in center_control if board.piece_at(square) and board.color_at(square) == board.turn)
        score += center_score * 10


        #Pawn structure
        score += self._evaluate_pawn_structure(board)

        
        #Add adjustments for positional factors, mobility, king safety, etc.
        return score
    
    def _evaluate_king_safety(self, board, color):
        #Placeholder function to evaluate king safety
        king_square = board.king(color)
        king_safety = 0
        if king_square in [chess.C1, chess.F1, chess.C8, chess.F8]:
            king_safety = - sum(1 for attack_square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]) if board.is_attacked_by(not color, attack_square))
        return king_safety
    
    def _evaluate_pawn_structure(self, board):
        #Placeholder function to evaluate pawn structure
        pawn_structure_value = 0
        #Example: penalize double pawns
        for color in [chess.WHITE, chess.BLACK]:
            squares = board.pieces(chess.PAWN, color)
            file_count = [0] * 8
            for sq in squares:
                file_count[chess.square_file(sq)] += 1
            pawn_structure_value += sum(-10 for count in file_count if count > 1) if color == chess.WHITE else sum(10 for count in file_count if count > 1)
        return pawn_structure_value
    

    def _evaluate_board(self, board):
        """
        Evaluates a provided board and assigns a score.
        :param board: The board to evaluate.
        :return: An evaluation score for the board.
        """
        if board.is_checkmate():
            return float('inf') if board.turn == chess.BLACK else float('-inf')
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
                        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000}
        
        # Calculate the material value of the pieces on the board.  
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                score += piece_values[piece.symbol()]
        
        mobility = len(list(board.legal_moves))
        score += mobility if board.turn == chess.WHITE else -mobility

        return score

    def get_board_visualization(self, board):
        """
        Generates a visual representation of the board.
        :param board: The board to visualize.
        :return: A visual representation of the board.
        """
        return chess.svg.board(board=board)
        

    def visualize_decision_process(self, depth, move, notation="SAN"):
        """
        Visualizes the decision-making process for a particular move up to a certain depth.
        :param depth: The depth of the analysis.
        :param move: The move being analyzed.
        :param notation: The notation system for the move (default: "SAN").
        """
        initial_board = self.root.board
        if notation == "SAN":
            try:
                move = initial_board.parse_san(move)
            except ValueError:
                print("Invalid move provided in SAN notation.")
                
        elif notation == "UCI":
            try:
                move = chess.Move.from_uci(move)
            except ValueError:
                print("Invalid move provided in UCI notation.")
            

        # Apply the move and get the resulting board
        initial_board.push(move)
        node = TreeNode(initial_board, 'w' if initial_board.turn == chess.BLACK else 'b')
        
        print(f"Analyzing move: {move}, Depth: {depth}")
        self._visualize_recursive(node, depth, 0, float('-inf'), float('inf'), initial_board.turn == chess.WHITE)

    def _visualize_recursive(self, node, depth, current_depth, alpha, beta, maximizing_player):
        if depth == 0 or node.is_leaf():
            score = self._evaluate_position(node, depth)
            print(f"Depth {current_depth}: Score {score}")
            return score

        print(f"Depth {current_depth}: Alpha {alpha}, Beta {beta}")

        if maximizing_player:
            value = float('-inf')
            for move in self._get_legal_moves(node):
                new_node = self._apply_move(move, node)
                if new_node is None:
                    continue  # Skip invalid moves
                print(f"Maximizing, considering move: {move}")
                child_value = self._visualize_recursive(new_node, depth - 1, current_depth + 1, alpha, beta, False)
                value = max(value, child_value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    print("Pruning branches below this node due to alpha-beta cutoff.")
                    break
        else:
            value = float('inf')
            for move in self._get_legal_moves(node):
                new_node = self._apply_move(move, node)
                if new_node is None:
                    continue  # Skip invalid moves
                print(f"Minimizing, considering move: {move}")
                child_value = self._visualize_recursive(new_node, depth - 1, current_depth + 1, alpha, beta, True)
                value = min(value, child_value)
                beta = min(beta, value)
                if beta <= alpha:
                    print("Pruning branches above this node due to alpha-beta cutoff.")
                    break
        return value



    def export_analysis(self):
        """
        Exports the analysis performed by the AlphaBetaChessTree to a JSON file.
        """
        analysis_data = {
            "fen": self.root.board.fen(),
            "analysis": []
        }

        def traverse(node, depth=0):
            if node is None:
                return
            node_data = {
                "fen": node.board.fen(),
                "depth": depth,
                "value": node.value,
                "children": []
            }
            for child in node.children:
                node_data["children"].append(traverse(child, depth + 1))
            analysis_data["analysis"].append(node_data)
            return node_data

        traverse(self.root)

        # Write the analysis data to a JSON file
        with open("chess_analysis.json", "w") as file:
            json.dump(analysis_data, file, indent=4)

        print("Analysis has been exported to chess_analysis.json.")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def generate_heatmap(moves,fen):
    """
    Generate a heatmap of moves made on a chessboard.

    Parameters:
    - moves (list): List of moves in UCI notation.

    Returns:
    - None (displays the heatmap)
    """
    # Initialize a 2D array to represent the chessboard
    heatmap = np.zeros((8, 8))

    # Iterate through moves and update the heatmap
    board = chess.Board(fen)
    for move in moves:
        move_obj = board.parse_san(move)
        to_square = move_obj.to_square
        row = to_square // 8
        col = to_square % 8
        heatmap[row][col] += 1
        board.push(move_obj)

    # Plot the heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Move Frequency')
    plt.title('Chess Heatmap')
    plt.xlabel('File')
    plt.ylabel('Rank')
    plt.gca().invert_yaxis()  # Invert y-axis to match chessboard orientation
    plt.xticks(np.arange(8), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    plt.yticks(np.arange(8), np.arange(1, 9))
    plt.show()

def uci_to_san(move, board):
    move_obj = board.parse_uci(move)
    return board.san(move_obj)

def draw_board(board):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    for i in range(8):
        ax.text(i, -0.6, str(chr(97+i)), ha='left', va='bottom', fontsize=12, color='black')
        ax.text(-0.6, i, str(i + 1), ha='left', va='bottom', fontsize=12, color='black')

    # Draw squares
    for i in range(8):
        for j in range(8):
            color = 'white' if (i + j) % 2 == 0 else 'pink'
            ax.add_patch(Rectangle((i, j), 1, 1, color=color))

    # Draw pieces
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece is not None:
                ax.text(i + 0.5, j + 0.5, piece.symbol(), ha='center', va='center', fontsize=20)

    plt.show()


def play_chess(fen , player="Human"):
    board = chess.Board(fen)
    turn = 1
    moves=[]
    originalfen=fen
    
    if player == "Human":
        while not board.is_game_over():
            draw_board(board)
            chess_tree = AlphaBetaChessTree(fen)
            if turn ==1:
                best_move = chess_tree.get_best_next_move(chess_tree.root, 3)
                print("The Computer's Best move is:", best_move)
                move = input("Enter your move: ")
            else:
                move = input("Enter your move: ")
            try:
                board.push_san(move)
                moves.append(move)
                generate_heatmap(moves,originalfen)
                fen=board.fen()
                if turn == 2:
                    turn=1
                else:
                    turn+=1
            except ValueError:
                print("Invalid move! Try again.")
        draw_board(board)
        result = board.result()
        print("Game Over! Result:", result)
        
        
    if player == "Computer":
        while not board.is_game_over():
            draw_board(board)
            chess_tree = AlphaBetaChessTree(fen)
            best_move = chess_tree.get_best_next_move(chess_tree.root, 3)
            print("Computer"+ str(turn) +"'s move:", best_move)
            move = input("Enter your move: ")
            try:
                board.push_san(move)
                fen=board.fen()
                if turn == 2:
                    turn=1
                else:
                    turn+=1
                
            except ValueError:
                print("Invalid move! Try again.")
        #Because right now I don't know how to convert uci to san, so basically we calculate for both sides which is the best move. The computer's move has to be typed by user.
        draw_board(board)
        result = board.result()
        print("Game Over! Result:", result)


    
def main():
    fen = "4k2r/P5r1/8/8/8/8/3R4/R3K3 w Qk - 0 1"
    #chess_tree = AlphaBetaChessTree(fen)
    #best_move = chess_tree.get_best_next_move(chess_tree.root, 3)
    #print("Best move:", best_move)
    gamer=input("Human or Computer: ")
    play_chess(fen,gamer)

    # Please implement your own main function to test the AlphaBetaChessTree class.
    # Try various FEN strings and different depths to see how your algorithm performs.
    # You can also implement your own evaluation functions to test the algorithm with different strategies.
    # For example, you could implement a simple evaluation function that counts the material value of the pieces.
    # (optional) It is recommended to implement methods to visualize your board state and moves.
    # (optional) It is recommended to support the export of tree statistics to understand tree pruning.

    # Feel free to add additional FEN strings and test cases to further evaluate your algorithms below.


if __name__ == "__main__":
    main()
