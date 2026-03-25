import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def get_latest_file(folder="data"):
    """Return the newest file inside the folder."""
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        raise FileNotFoundError("No data files found in 'data/'")
    latest = max(files, key=os.path.getmtime)
    return latest

def line_plot(file_path):
    data = np.loadtxt(file_path)
    t = data[:, 0]

    _, ax = plt.subplots()
    for col in data[:, 1:].T:
        ax.plot(t, col)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [rad]")
    ax.set_title("Satellite Attitude Angles")
    ax.grid(True)

    plt.show()

def main(argv):
    # If no arguments → automatically plot latest file
    if len(argv) == 1:
        latest = get_latest_file()
        print(f"Plotting latest file: {latest}")
        line_plot(latest)
        return

    # Manual mode (old behavior)
    if len(argv) == 3:
        plot_type = argv[1]
        file_path = argv[2]
        if plot_type == 'lineplot':
            line_plot(file_path)
        else:
            print("Plot type not supported yet.")
    else:
        print("Wrong number of arguments. Expected 2 (plot_type, file_path) or none.")

if __name__ == "__main__":
    main(sys.argv)
