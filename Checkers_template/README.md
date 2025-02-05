# Checkers

## Game Overview
Checker is a classic two-player strategy game where players move pieces across a checkered board, aiming to either eliminate all of their opponent’s pieces or strategically block them from making a move. The game involves standard pieces and kings, which have enhanced movement capabilities.

## Game Rules

### 1. Board Setup
   - The board size varies depending on the difficulty level:
     - **Low Difficulty:** 6x6 board, offering a simplified experience suitable for beginners.
     - **Medium Difficulty:** 8x8 board, providing the standard gameplay experience.
     - **High Difficulty:** 8x8 board, incorporating advanced AI or competitive strategies.
   - The board consists of alternating light and dark squares.
   - Each player’s pieces are placed on dark squares only.
   - The bottom-right corner of the board should always be a light-colored square.

### 2. **Players and Objective**
   - The game is played between two opponents, each taking turns to move their pieces.
   - The primary objective is to capture all opponent pieces or strategically position pieces so that the opponent has no legal moves left.

### 3. **Piece Movement**
   - **Regular pieces**:
     - Can move diagonally forward by one square.
     - Cannot move backward unless promoted to a king.
     - Can capture an opponent’s piece by jumping over it into an empty space.
   - **Mandatory Capture Rule**:
     - If a capture move is available, the player must make that move.
     - If multiple capture moves are available, the player can choose any of them.
   
### 4. **Capturing Rules**
   - To capture an opponent’s piece, the player’s piece must jump diagonally over it into an empty square.
   - Captured pieces are removed from the board immediately.
   - If possible, players can perform **multiple captures** in a single turn by making consecutive jumps.
   - Chain captures can be executed with both regular pieces and kings.

### 5. **King Promotion**
   - When a regular piece reaches the last row of the opponent’s side of the board, it is promoted to a **king**.
   - A king is indicated by stacking another piece on top.
   - Kings have enhanced movement capabilities:
     - They can move diagonally **both forward and backward**.
     - They can also capture pieces in either direction.

### 6. **Winning Conditions**
   - A player **wins** the game if:
     - They capture all opponent pieces.
     - The opponent has no valid moves left.
   - If neither player can make a move, the game is considered a draw.

## Gameplay Guide
- **Strategic Positioning**: Players should aim to position their pieces strategically to control the board and set up future captures.
- **Forcing Opponent Moves**: Forcing an opponent into a disadvantageous move can turn the tide of the game.
- **Utilizing Kings Effectively**: Kings offer a tactical advantage, so promoting pieces should be a key objective.

## Future Updates
- Implementing an AI opponent with varying difficulty levels.
- Introducing an online multiplayer mode for competitive play.
- Enhancing the user interface for a more immersive gaming experience.

## Contributions
We welcome suggestions and contributions! If you have ideas for improvement or want to contribute code, feel free to submit a Pull Request or an Issue on the project repository.


