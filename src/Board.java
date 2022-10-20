/**
 * Author: Maciej Leciejewski
 */
public class Board {

    private final int height;
    private final int width;
    private final int[][] board;
    private boolean nextTurn;

    public static final int EMPTY_SLOT = 0;
    public static final int PLAYER_1_DISK = 1;
    public static final int PLAYER_2_DISK = 2;

    public static final boolean PLAYER_1_TURN = true;

    public static final int ONGOING = 0;
    public static final int PLAYER_1_WON = 1;
    public static final int PLAYER_2_WON = 2;
    public static final int TIE = 3;

    /**
     * Instantiates a new Board.
     *
     * @param width  the width
     * @param height the height
     */
    public Board(int width, int height) {
        this.width = width;
        this.height = height;
        board = new int[height][width];
        nextTurn = PLAYER_1_TURN;
    }

    /**
     * Instantiates a new Board.
     */
    public Board() {
        this(7, 6);
    }

    /**
     * Instantiates a new Board.
     *
     * @param contents the contents
     * @param nextTurn the next turn
     */
    public Board(int[][] contents, boolean nextTurn) {
        this(contents[0].length, contents.length);
        loadContents(contents);
        this.nextTurn = nextTurn;
    }

    /**
     * Method checks if player is able to do move in given column
     *
     * @param column the column
     * @return the boolean
     */
    public boolean canPlace(int column) {
        return column >= 0 && column < width && board[0][column] == 0;
    }

    /**
     * Method makes move in given column
     *
     * @param column the column
     */
    public void place(int column) {
        int disk = (nextTurn == PLAYER_1_TURN) ? PLAYER_1_DISK : PLAYER_2_DISK;
        if (!canPlace(column)) {
            return;
        }
        int diskHeight = height - 1;
        while (board[diskHeight][column] != EMPTY_SLOT) {
            diskHeight--;
        }
        board[diskHeight][column] = disk;
        nextTurn = !nextTurn;
    }

    /**
     * Gets next state.
     *
     * @param column the column
     * @return updated board, after make move in given column
     */
    public Board getNextState(int column) {
        Board next = this.copy();
        next.place(column);
        return next;
    }

    /**
     * Method sets board state for given contents
     *
     * @param contents the contents
     */
    public void loadContents(int[][] contents) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                board[i][j] = contents[i][j];
            }
        }
    }

    /**
     * Copy board.
     *
     * @return copy of the board with actual state
     */
    public Board copy() {
        return new Board(board, this.nextTurn);
    }

    /**
     * Method checks if a player have won
     *
     * @param playerDisk the player disk
     * @return the boolean
     */
    public boolean didPlayerWin(int playerDisk) {
        int height = board.length;
        int width = board[0].length;
        // check horizontal
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width - 3; j++) {
                for (int k = j; k < j + 4 && board[i][k] == playerDisk; k++) {
                    if (k == j + 3) {
                        return true;
                    }
                }
            }
        }
        // check vertical
        for (int i = 0; i < height - 3; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = i; k < i + 4 && board[k][j] == playerDisk; k++) {
                    if (k == i + 3) {
                        return true;
                    }
                }
            }
        }
        // check diagonal down right
        for (int i = 0; i < height - 3; i++) {
            for (int j = 0; j < width - 3; j++) {
                for (int k = 0; k < 4 && board[i + k][j + k] == playerDisk; k++) {
                    if (k == 3) {
                        return true;
                    }
                }
            }
        }
        // check diagonal down left
        for (int i = 0; i < height - 3; i++) {
            for (int j = 3; j < width; j++) {
                for (int k = 0; k < 4 && board[i + k][j - k] == playerDisk; k++) {
                    if (k == 3) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Method checks if the board is full (no more possible moves)
     *
     * @return the boolean
     */
    public boolean isFull() {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == EMPTY_SLOT) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Method checks current game state
     * ( if one of 2 players have won or the game have ended with a tie
     * or the game have to bo continued )
     *
     * @return the int
     */
    public int currentGameState() {
        return this.didPlayerWin(PLAYER_1_DISK) ? PLAYER_1_WON
                : this.didPlayerWin(PLAYER_2_DISK) ? PLAYER_2_WON
                : this.isFull() ? TIE
                : ONGOING;
    }

    /**
     * Method checks which player have to make a move
     *
     * @return boolean
     */
    public boolean getNextTurn() {
        return nextTurn;
    }

    /**
     * Method implements displaying of a board
     *
     * @return String
     */
    @Override
    public String toString() {
        String result = "|-";
        for (int j = 0; j < width; j++) {
            result += "--|-";
        }
        result = result.substring(0, result.length() - 1) + "\n";
        for (int i = 0; i < height; i++) {
            result += "| ";
            for (int j = 0; j < width; j++) {
                result += (board[i][j] == EMPTY_SLOT ? " " : (board[i][j] == 1 ? "O" : "X")) + " | ";
            }
            result = result.substring(0, result.length() - 1);
            result += "\n|-";
            for (int j = 0; j < width; j++) {
                result += "--|-";
            }
            result = result.substring(0, result.length() - 1);
            result += "\n";
        }
        result += "  0   1   2   3   4   5   6  ";
        return result.substring(0, result.length() - 1);
    }

    /**
     * Gets width.
     *
     * @return int
     */
    public int getWidth() {
        return width;
    }
}
