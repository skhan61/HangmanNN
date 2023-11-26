from pathlib import Path

import matplotlib.pyplot as plt


def plot_word_stats(word_length_stats, epoch=None, save_path=None):
    # Validate input format
    if not isinstance(word_length_stats, dict) or not all(isinstance(value, dict) for value in word_length_stats.values()):
        raise ValueError(
            "word_length_stats must be a dictionary of dictionaries")

    # Prepare data for plotting
    word_lengths = sorted(word_length_stats.keys())
    wins = [word_length_stats[length]['wins'] for length in word_lengths]
    losses = [word_length_stats[length]['losses'] for length in word_lengths]
    total_attempts = [word_length_stats[length]['total_attempts']
                      for length in word_lengths]
    avg_attempts = [word_length_stats[length]['total_attempts'] /
                    word_length_stats[length]['games'] for length in word_lengths]

    # Creating subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Wins and Losses
    axs[0, 0].bar(word_lengths, wins, label='Wins', color='green', alpha=0.6)
    axs[0, 0].bar(word_lengths, losses, label='Losses',
                  color='red', alpha=0.6, bottom=wins)
    axs[0, 0].set_title('Wins and Losses by Word Length')
    axs[0, 0].set_xlabel('Word Length')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].legend()

    # Total Attempts
    axs[0, 1].bar(word_lengths, total_attempts, color='blue', alpha=0.6)
    axs[0, 1].set_title('Total Attempts by Word Length')
    axs[0, 1].set_xlabel('Word Length')
    axs[0, 1].set_ylabel('Total Attempts')

    # Average Attempts
    axs[1, 0].bar(word_lengths, avg_attempts, color='purple', alpha=0.6)
    axs[1, 0].set_title('Average Attempts per Game by Word Length')
    axs[1, 0].set_xlabel('Word Length')
    axs[1, 0].set_ylabel('Average Attempts')

    # Wins vs. Losses Ratio
    win_loss_ratio = [wins[i] / max(losses[i], 1) for i in range(len(wins))]
    axs[1, 1].plot(word_lengths, win_loss_ratio, color='orange', marker='o')
    axs[1, 1].set_title('Win/Loss Ratio by Word Length')
    axs[1, 1].set_xlabel('Word Length')
    axs[1, 1].set_ylabel('Ratio')

    # Adjusting layout
    plt.tight_layout()

    # Determine and create save path
    if save_path is None:
        save_filename = f'epoch_{epoch}.png' if epoch is not None else 'stats_plot.png'
        save_path = Path(f'plots/wordwise_stats/{save_filename}')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the plots
    plt.savefig(save_path)

    # Close the plot
    plt.close()

# Example usage
# word_length_stats = {...}  # Use your word_length_stats dictionary here
# plot_word_stats(word_length_stats, epoch=1)
