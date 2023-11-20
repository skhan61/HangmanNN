import matplotlib.pyplot as plt
from pathlib import Path

def plot_word_stats(word_length_stats, epoch, save_path=None):
    # Prepare data for plotting
    word_lengths = list(word_length_stats.keys())
    wins = [word_length_stats[length]['wins'] for length in word_lengths]
    losses = [word_length_stats[length]['losses'] for length in word_lengths]
    total_attempts = [word_length_stats[length]['total_attempts'] for length in word_lengths]
    avg_attempts = [word_length_stats[length]['avg_attempts'] for length in word_lengths]

    # Creating subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Wins and Losses
    axs[0, 0].bar(word_lengths, wins, label='Wins', color='green', alpha=0.6)
    axs[0, 0].bar(word_lengths, losses, label='Losses', color='red', alpha=0.6, bottom=wins)
    axs[0, 0].set_title('Wins and Losses')
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
    axs[1, 0].set_title('Average Attempts by Word Length')
    axs[1, 0].set_xlabel('Word Length')
    axs[1, 0].set_ylabel('Average Attempts')

    # Adjusting layout
    plt.tight_layout()

    # Determine and create save path
    if save_path is None:
        save_path = Path(f'plots/wordwise_stats/epoch_{epoch}.png')
    save_path = Path(save_path)  # Ensure Path object
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Save the plots
    plt.savefig(save_path)

    # Close the plot
    plt.close()