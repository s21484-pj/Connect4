import java.util.ArrayList;

/**
 * Author: Maciej Leciejewski
 */
public class AI {

    private MonteCarloTreeSearchNode root;
    private final int width;
    private final long GIVEN_TIME;

    /**
     * Instantiates a new Ai.
     *
     * @param board      the board
     * @param GIVEN_TIME the given time
     */
    public AI(Board board, long GIVEN_TIME) {
        this.width = board.getWidth();
        this.GIVEN_TIME = GIVEN_TIME;
        root = new MonteCarloTreeSearchNode(null, board.copy());
    }

    /**
     * Method makes move and sets root to a new board state
     *
     * @param move the move
     */
    public void update(int move) {
        root = root.children[move] != null
                ? root.children[move]
                : new MonteCarloTreeSearchNode(null, root.board.getNextState(move));
    }

    /**
     * Method calculates move power for all possible moves,
     * collect statistics and choose the best one
     *
     * @return the optimal move
     */
    public int getOptimalMove() {
        for (long stop = System.nanoTime() + GIVEN_TIME; stop > System.nanoTime(); ) {
            MonteCarloTreeSearchNode selectedNode = select(root);
            if (selectedNode == null) {
                continue;
            }
            MonteCarloTreeSearchNode expandedNode = expand(selectedNode);
            double result = simulate(expandedNode);
            backPropagate(expandedNode, result);
        }

        int maxIndex = -1;
        for (int i = 0; i < width; i++) {
            if (root.children[i] != null) {
                if (maxIndex == -1 || root.children[i].visits > root.children[maxIndex].visits) {
                    maxIndex = i;
                }
            }
        }
        return maxIndex;
    }

    /**
     * Method calculates move power for given node
     *
     * @param parent the root
     * @return node
     */
    private MonteCarloTreeSearchNode select(MonteCarloTreeSearchNode parent) {
        for (int i = 0; i < width; i++) {
            if (parent.children[i] == null && parent.board.canPlace(i)) {
                return parent;
            }
        }
        double maxSelectionVal = -1;
        int maxIndex = -1;
        for (int i = 0; i < width; i++) {
            if (!parent.board.canPlace(i)) {
                continue;
            }
            MonteCarloTreeSearchNode currentChild = parent.children[i];
            double wins = parent.board.getNextTurn() == Board.PLAYER_1_TURN
                    ? currentChild.player1Wins
                    : (currentChild.visits - currentChild.player1Wins);
            double selectionVal = wins / currentChild.visits
                    + Math.sqrt(2) * Math.sqrt(Math.log(parent.visits) / currentChild.visits);
            if (selectionVal > maxSelectionVal) {
                maxSelectionVal = selectionVal;
                maxIndex = i;
            }
        }
        if (maxIndex == -1) {
            return null;
        }
        return select(parent.children[maxIndex]);
    }

    /**
     * Method creates a node for randomly chosen unvisited child node
     *
     * @param selectedNode selected node
     * @return node
     */
    private MonteCarloTreeSearchNode expand(MonteCarloTreeSearchNode selectedNode) {
        ArrayList<Integer> unvisitedChildrenIndices = new ArrayList<>(width);
        for (int i = 0; i < width; i++) {
            if (selectedNode.children[i] == null && selectedNode.board.canPlace(i)) {
                unvisitedChildrenIndices.add(i);
            }
        }

        int selectedIndex = unvisitedChildrenIndices.get((int) (Math.random() * unvisitedChildrenIndices.size()));
        selectedNode.children[selectedIndex] = new MonteCarloTreeSearchNode(selectedNode, selectedNode.board.getNextState(selectedIndex));
        return selectedNode.children[selectedIndex];
    }

    /**
     * Method makes simulation for given node
     *
     * @param expandedNode expanded node
     * @return result
     */
    private double simulate(MonteCarloTreeSearchNode expandedNode) {
        Board simulationBoard = expandedNode.board.copy();
        while (simulationBoard.currentGameState() == Board.ONGOING) {
            simulationBoard.place((int) (Math.random() * width));
        }

        return switch (simulationBoard.currentGameState()) {
            case Board.PLAYER_1_WON -> 1;
            case Board.PLAYER_2_WON -> 0;
            default -> 0.5;
        };
    }

    /**
     * Method increment visits and player1 score in current node
     * and go back to root node in tree's structure
     *
     * @param expandedNode     expanded node
     * @param simulationResult simulation result
     */
    private void backPropagate(MonteCarloTreeSearchNode expandedNode, double simulationResult) {
        MonteCarloTreeSearchNode currentNode = expandedNode;
        while (currentNode != null) {
            currentNode.incrementVisits();
            currentNode.incrementPlayer1Wins(simulationResult);
            currentNode = currentNode.parent;
        }
    }

    private class MonteCarloTreeSearchNode {

        private final MonteCarloTreeSearchNode parent;
        private final MonteCarloTreeSearchNode[] children;
        private int visits;
        private double player1Wins;
        private final Board board;

        /**
         * Instantiates a new Monte carlo tree search node.
         *
         * @param parent the parent
         * @param board  the board
         */
        public MonteCarloTreeSearchNode(MonteCarloTreeSearchNode parent, Board board) {
            this.parent = parent;
            this.board = board;
            this.visits = 0;
            this.player1Wins = 0;
            children = new MonteCarloTreeSearchNode[width];
        }

        /**
         * Increment visits.
         */
        public void incrementVisits() {
            ++visits;
        }

        /**
         * Increment player 1 wins.
         *
         * @param result the result
         */
        public void incrementPlayer1Wins(double result) {
            player1Wins += result;
        }
    }
}
