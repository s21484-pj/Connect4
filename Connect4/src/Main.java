import java.util.Scanner;
import java.util.concurrent.TimeUnit;

/**
 * Author: Maciej Leciejewski
 */
public class Main {

    private static final Scanner SCANNER = new Scanner(System.in);
    private static final long GIVEN_TIME = TimeUnit.SECONDS.toNanos(2);

    /**
     * The entry point of application, contains game loop.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {
        Board board = new Board();
        AI ai = new AI(board, GIVEN_TIME);

        while (board.currentGameState() == Board.ONGOING) {
            System.out.println("\n\n" + board);
            int moveColumn;
            do {
                if (board.getNextTurn() == Board.PLAYER_1_TURN) {
                    System.out.println("Enter your move: ");
                    moveColumn = SCANNER.nextInt();
                } else {
                    System.out.print("AI doing move: ");
                    moveColumn = ai.getOptimalMove();
                    System.out.println(moveColumn);
                }
            } while (!board.canPlace(moveColumn));
            board.place(moveColumn);
            ai.update(moveColumn);
        }

        int gameState = board.currentGameState();
        System.out.println("\n\n");
        System.out.println(board);

        switch (gameState) {
            case Board.PLAYER_1_WON -> System.out.println("You won.\n");
            case Board.PLAYER_2_WON -> System.out.println("AI won.\n");
            default -> System.out.println("Tie.\n");
        }
    }
}
