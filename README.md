# Neural Network for Checkmates using Pattern Recognition

## Introduction  
As a part of the exploration of the human mind, it is also a common consequence to replicate human performance in tasks through artificial intelligence, including in boardgames such as Chess, Shogi, or Go.  These three games have finite board states, which enables the viability to search through all board states. Game engines such as Stockfish for Chess have implemented domain-specific heuristics such as piece value, opening types, used board advantages, and end-game tablebases to guide its early and late game decisions, and implemented search-space analysis algorithms in order to generate its next move. Thus, the artificial intelligence traverses through differently valued board states through the Monte Carlo Tree Search in order to think of its next move. Stockfish, using its manually tuned algorithm, has proven itself as an engine capable of performing to the level of top human chess player. However, due to the implementation of a search tree algorithm as its sole decision-making guide, it is required to search through millions of positions in order to make decisions.
  
AlphaZero is a relatively new engine that seeks to improve upon the play and the computing time of Stockfish. Instead of using manually adjusted heuristics as in Stockfish, AlphaZero has replaced it with machine learning. (Silver et al. 2018) An untrained neural network from AlphaZero has no knowledge of the meaning of chess pieces, board control, or any other game component that is often considered essential in mastering the game of chess, and learns board state evaluations through machine learning. This method results in artificial intelligence trained by AlphaZero to be much more aggressive in play due to its not assigning a specific value to a piece. Taking inspiration from the neural network portion of AlphaZero, this work seeks to explore the aspect that follows a board evaluation, the generation of a move, through a neural network. The goal of this specific neural network is to generate a move that leads to a checkmate. 
  
  
## Methods  
### Data  
  
A total of 81081 sample board states one move before checkmates were obtained and used. A portion of the chess data was pulled from the Free Internet Chess Server (FICS) database, limited to standard mode games of 2000 or above rating played in June through August of 2020. There was a sparseness in games that ended in checkmates due to the possibility of forfeit by one of the players if in a disadvantaged position. In order to alleviate the sparseness of checkmate moves within the FICS dataset, as well as avoid the length and difficulty processing the dataset properly, the remaining majority of board states were obtained through generation of random legal chess moves between two players through the Chess module in Python 3.7 in order to extract board states one move before checkmate.
  
|![PGN File](/images/pgn.png)|
|:--:| 
|![FEN File](/images/fen.png)|
|![Board img](/images/board.png)|
|Figure 1. (a) An example of a chess game in PGN format. (b) An example of a FEN string of a single board state and (c) its board layout|
   
### Pre-Processing of data  
FICS data was obtained in Portable Game Notation (PGN) format, which contains information of the players involved in the game as well as a string that signifies each move that was made in the game. The generated dataset was obtained in Forsyth-Edwards Notation (FEN) format, a string that signifies the layout of the current board state. (Figure 1) Data in the PGN format was translated to FEN format. The FEN format data was then processed into 25 by 8 by 8 tensors which were then used as inputs in the neural network.  
  
The data was one-hot encoded and represented in 25 channels, in order to encode enough data for possible moves of the pieces. The first layer signified the turn side,  1 for white and 0 for black. The next 12 channels described the positions of each piece. The remaining 12 channels described the positions of legal moves for each piece. One-hot encoding was the best representation to be used in this dataset due to the requirement of representing categorical information of pieces, and using numbers for each piece would lead to greater importance depending on how the pieces were translated into numbers. Through one-hot encoding, each piece is valued the same, and the only differentiation between them is the channel they are assigned to.(Vikström, 2019)  
  
### Network Training and Testing  
The goal of the network was to classify the best next move, or the move that leads to a checkmate on the current board. The basic architecture follows that of many image processing convolutional neural network, but with different tensor sizes to accommodate our data input sizes. In order to prevent over-fitting, dropout layers of probabilities 25% and 50% were defined after each fully connected layer, and in order to compare learning rate, batch normalization layers were defined after each convolutional layer. The Adam optimizer was also used to minimize the cost functions. Pooling layers were not used in our neural network because we wanted to preserve as much of the features as possible. (Oshri and Khandwala, 2018)
  
The moves produced by the output of the network were validated against the best move produced by the UCI chess engine under the Python chess module. 
  
## Results 
|![acc1](/images/acc1.png)|
|:--:| 
|Figure 2. The accuracy of the neural network over time, with training accuracy in blue and test accuracy in red.|

|![loss](/images/loss.png)|
|:--:| 
|Figure 3. The loss of the neural network over time, with training loss in blue and test loss in red.|

|![acc2](/images/acc2.png)|
|:--:| 
|Figure 4. The accuracy of the neural network over time with batch normalization, with training accuracy in blue and test accuracy in red.|
  
The neural networks were run over 100 epochs. The network training was done through a CPU due to the unavailability of a GPU. The results showed that the network was able to attain an accuracy above 98%, and a training accuracy of 81%. In order to compare learning rates with and without batch normalization, we included the function between each convolutional layer. With batch normalization, training accuracy approached near 99.98% and a training accuracy of 90%. An ANOVA analysis of the performance of the two conditions over the 100 epochs it was trained showed that there was a significant difference in performance between the neural netowrk trained with batch normalization and the neural network trained without. (p < 0.001) This analysis was done as an exploratory measure, but nonetheless showed that the inclusion of batch normalization has clear advantages, not only in the speed of learning, but also arriving at a higher accuracy rate than without batch normalization.   
  
|![pieceVcount](/images/pieceVcount.png)|
|:--:| 
|Figure 5. The frequency of correct piece to move for checkmate in the training set.|
  
|![pieceVerror](/images/pieceVerror.png)|
|:--:| 
|Figure 6. The error rate of the neural network in data requiring the corresponding piece needed to move for checkmate.|
  
Analysis was also done on the dataset itself.  Additionally, FEN strings of incorrect outputs were stored for analysis in a separate file. The training dataset was analyzed to observe the distribution of the pieces that were required to move in order to reach checkmate. The overall piece frequency matched recorded data of piece usage towards checkmate in official matches.(Figure 5, Figure 6) There was a high frequency of queen moves as well as sparse presence of king movement in the training dataset, which was also reflected in the underlearning that was evident in the test set. Due to the high presence of queen moves in the training dataset, especially relative to other rarer pieces such as the king, this also may have led to the high error rate in king movement data. This lack of learning or utilization of the king in the neural network was reflected both in the training phase and the testing phase, as there was little to no learning or usage of the king piece and its movement.  
  
## Discussion  
This project was a simple exploration of the ability of neural networks to learn goal-oriented moves through pattern recognition without knowledge of rules of chess, and the current neural network satsifactorily completed its task in finding the required move. The project also served as an exploratory demonstration of the ability of convolutional neural networks to solve problems through a method of pattern recognition and a form of image regression. It also demonstrated the necessity and importance of proper data organization and extraction for the neural network to interpret. The analysis of frequencies of pieces moved for checkmate showed a distribution that had a heavy prevalence of queen movement, followed by rook movement, ending with king movement making a very minimal portion of our dataset. The analysis of the test set revealed an inverse of the frequency of the test set, showing that all checkmate moves that required a king to move had high error rate, and those that required the queen to move had lower error rates.  
  
The inclusion of batch normalization in our neural network generated interesting results. The model that included batch normalization not only learned much faster, starting from the first epoch, but also arrived at higher test accuracy.  Although the effect of batch normalization in this network does not clarify the role of it in the current neural network, it nonetheless led to higher accuracy within the test data set.  
  
However, this neural network is not without its shortcomings. Far from the abilities of AlphaZero, this neural network is simply attempting to make a winning move while one move away. This neural network may be more in line of predicting the next move made by the UCI chess engine, a checkmate, rather than finding the checkmate in a problem solving manner. It is also possible that too much of the data was composed of zeroes in the dataset in encoding. Although the network did give answers at a satsifactory rate, it is possible that the sparseness of the information in the data, as well as the usage of 25 channels led to abnormalities in the process.  
  
Additionally, there may be a need in addressing concerns about the disparity of frequencies between pieces in the dataset, which most likely led to the high error rates in less commonly used pieces. Due to the various ways a game of chess can end, in stalemates, draws, and most commonly, forfeit, there was a very low percentage of pre-existing data that could be used. Still, there may be a need to even out the dataset to include all pieces equally. The current dataset that was generated reflects the real world frequencies of piece usage in a checkmate. However, this disparity in frequency most likely contributed to the underlearning of specific channels in the neural network. A fix involving generating similar number of samples for each piece would require a tremendous number of samples due to the rarity of movements of kings and pawn movement leading to checkmate. An even distribution in the training dataset may lead to a better performance overall across different pieces, as it is likely that the current network had developed a bias toward queen movement.   
  
Future direction of this neural network would include changing the hyperparameters of the neural network, and changing the encoding representation of each data. Specifically to the movement towards a checkmate, a control or manipulation of distribution of training samples may be needed in order to increase testing accuracy.   
  
  
## References  
Oshri, B., & Khandwala, N. (2016). Predicting moves in chess using convolutional neural networks.    
  
Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Lillicrap, T. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 362(6419), 1140-1144.  
  
Vikström, J. (2019). Training a Convolutional Neural Network to Evaluate Chess Positions.  
