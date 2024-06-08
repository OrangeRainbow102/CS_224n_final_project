
import matplotlib.pyplot as plt


def graph_data(y_data, x_data, title, save_path, y_label='Loss', label='Multiple Negatives Ranking Loss'):

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    ax.plot(x_data, y_data, label=label, color='b', linewidth=2, linestyle='-')

    # Customize the plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Customize the ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    # Show the plot
    plt.savefig(save_path)


def main():
    loss = [0.4963, 0.3591, 0.3078, 0.1516, 0.081, 0.0774, 0.0379, 0.027]
    steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    graph_data(loss, steps, "Training Loss on Large Synthetic Dataset", "LARGE_loss.png")

if __name__ == '__main__':
    main()