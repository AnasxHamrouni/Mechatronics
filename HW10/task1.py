import numpy as np
import matplotlib.pyplot as plt
import csv

def load_rated_torques(csv_path):
    rated = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            joint = row["Joint"]
            rated[joint] = float(row["MaxTorqueNm"])
    return rated

def load_static_torques():
    np.random.seed(0)
    static_torques = []
    means = [2.0, 10.0, 6.0, 0.8, 0.1, 0.05] 
    for m in means:
        static_torques.append(m + 0.5 * np.random.randn(200))
    return static_torques

if __name__ == "__main__":
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    rated = load_rated_torques("motors-data.csv")   
    static_torques = load_static_torques()         
    data = [np.abs(tau) for tau in static_torques]  # use |tau|

    # Violin plot 
    fig, ax = plt.subplots(figsize=(8, 4))
    vp = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)
    ax.set_xticks(np.arange(1, len(joint_names) + 1))
    ax.set_xticklabels(joint_names, rotation=45, ha="right")
    ax.set_ylabel("Static torque |τ| [Nm]")
    ax.set_title("Static torque distribution vs rated torque (per joint)")

    # Overlay rated torque as horizontal lines
    for i, jname in enumerate(joint_names, start=1):
        if jname in rated:
            ax.hlines(rated[jname], i - 0.4, i + 0.4,
                      colors="r", linestyles="--", linewidth=2,
                      label="Rated torque" if i == 1 else None)

    ax.grid(axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.show()

    # Bar plot of means 
    means = [np.mean(d) for d in data]
    rated_vals = [rated[j] for j in joint_names]

    x = np.arange(len(joint_names))
    width = 0.35

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(x - width/2, means, width, label="Measured mean |τ|")
    ax2.bar(x + width/2, rated_vals, width, label="Rated torque")

    ax2.set_xticks(x)
    ax2.set_xticklabels(joint_names, rotation=45, ha="right")
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_title("Static torque vs rated torque (mean per joint)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()
