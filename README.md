# Neural Hangman: Seq2Seq Model for Hangman Prediction

## Abstract
This project explores the application of sequence-to-sequence (Seq2Seq) modeling to enhance the classic word-guessing game Hangman. The primary objective is to surpass a baseline algorithm's win rate significantly. This baseline uses a training dictionary alignment and letter frequency for guesses. Our approach employs Seq2Seq modeling to predict successive guesses by analyzing the changing sequence of game states and guesses, incorporating letter frequency and contextual gameplay patterns. An LSTM-based initial implementation shows promise, achieving approximately a 22% success rate. The ongoing challenge is to refine the Seq2Seq model, integrating frequency analysis with nuanced gameplay patterns for improved performance. This project demonstrates the versatility of Seq2Seq models in sequential decision-making and pattern recognition contexts, offering AI-driven strategies for various games.

## Algorithm

### Neural Hangman: Seq2Seq Model for Hangman Prediction
**Require:** Training Dictionary `D`, Vocabulary Size `V`  
**Ensure:** Trained Model for Hangman Prediction

```plaintext
1: function GENERATETRAININGDATA(D)
2:     Initialize DataSet TrainingData ← ∅
3:     for each word w in D do
4:         Sample w based on its length
5:         Create game states for w and add to TrainingData
6:     end for
7:     return TrainingData
8: end function

9: function TRAINMODEL(TrainingData, V)
10:    Initialize Seq2Seq Model with Vocabulary Size V
11:    for each batch B in TrainingData do
12:        Extract sequences and labels from B
13:        Conduct forward pass with sequences through the Model
14:        Calculate Loss and MissPenalty using CALCULATELOSS(...)
15:        Update Model weights through backpropagation
16:    end for
17:    return Seq2Seq Model
18: end function

19: function CALCULATELOSS(output, labels, lengths, misses, V)
20:    Apply activation function to output
21:    Compute miss penalty based on incorrect guesses
22:    Calculate Binary Cross-Entropy Loss with weights
23:    return Loss, MissPenalty
24: end function

25: function PREDICTGUESS(Model, GameState)
26:    Analyze GameState using Model
27:    Predict the most likely letter as the next guess
28:    return Predicted Letter
29: end function

30: TrainingData ← GENERATETRAININGDATA(D)
31: HangmanModel ← TRAINMODEL(TrainingData, V)
