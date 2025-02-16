import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

pd.options.mode.copy_on_write = True


def get_features(multifocal):
    return (
        [
            "OD Sphere",
            "OD Cylinder",
            "OD Axis",
            "OD Add",
            "OS Sphere",
            "OS Cylinder",
            "OS Axis",
            "OS Add",
        ]
        if multifocal
        else [
            "OD Sphere",
            "OD Cylinder",
            "OD Axis",
            "OS Sphere",
            "OS Cylinder",
            "OS Axis",
        ]
    )


def reverse_wraparound_axis(axis_value):
    return 180 - axis_value if axis_value > 90 else axis_value


def read_data(file_pattern, multifocal, location):
    # Use glob to get all file paths matching the pattern
    file_paths = glob.glob(file_pattern)

    # Read and concatenate all CSV files into one DataFrame
    data_frames = [pd.read_csv(file) for file in file_paths]
    data = pd.concat(data_frames, ignore_index=True)
    if multifocal:
        data = data[data["Type"] == "multifocal"]
    else:
        data = data[data["Type"] == "single"]
    data = data[data["Location"] == location]
    # Create a comparison DataFrame
    data["Added date (in CST)"] = pd.to_datetime(data["Added date (in CST)"])
    data["dispension date (in CST)"] = pd.to_datetime(data["dispension date (in CST)"])

    # Add new columns for original axis data
    data["OD Axis Original"] = data["OD Axis"].apply(lambda x: 0 if x == 180 else x)
    data["OS Axis Original"] = data["OS Axis"].apply(lambda x: 0 if x == 180 else x)

    # Handle reverse wraparound for axis
    data["OD Axis"] = data["OD Axis"].apply(reverse_wraparound_axis)
    data["OS Axis"] = data["OS Axis"].apply(reverse_wraparound_axis)

    return data


def clean_data(data, multifocal, refererence=None):
    X = data[get_features(multifocal)]

    # Handle missing values (if any)
    X.fillna(0, inplace=True)

    # Normalize the data
    scaler = StandardScaler()
    scaler.fit(refererence[get_features(multifocal)] if refererence is not None else X)
    X_scaled = scaler.transform(X)
    # Calculate Z-scores
    z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0))

    # Filter out rows where any Z-score is above the threshold (e.g., 3)
    threshold = 2.8
    new_data = data[(z_scores < threshold).all(axis=1)]
    # print("Number of pre-filtered glasses:", len(data) - len(new_data))
    return new_data


def scale_data(data, multifocal):
    X = data[get_features(multifocal)]
    # Standardize the filtered data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def clustering(data, X_scaled, num_clusters):
    # Perform hierarchical clustering on filtered data
    Z = linkage(X_scaled, method="ward")

    data["cluster"] = fcluster(Z, t=num_clusters, criterion="maxclust")
    data["cluster"] = data["cluster"].astype(str)

    # Plot the dendrogram with a smaller figure size and truncate mode
    plt.figure(figsize=(8, 3))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=num_clusters * 2.5,
        leaf_rotation=90.0,
        leaf_font_size=12.0,
        color_threshold=Z[-num_clusters, 2],
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Glasses")
    plt.ylabel("Distance")
    plt.ylim(0, 60)  # Set y-axis limit
    plt.show()


def pca(data, multifocal):
    # Apply PCA
    pca = PCA(n_components=2)
    X_scaled, _ = scale_data(data, multifocal)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Loading scores
    loading_scores = pd.DataFrame(
        pca.components_.T, columns=["PC1", "PC2"], index=get_features(multifocal)
    )
    print("Loading scores:\n", loading_scores)

    # Plot the PCA components
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=data["cluster"].astype(int),
        cmap="viridis",
        alpha=0.5,
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Dispense Report Data")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

    # Plot loading scores
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    loading_scores["PC1"].plot(kind="bar", ax=ax[0])
    ax[0].set_title("Loading Scores for PC1")
    ax[0].set_ylabel("Loading Score")
    ax[0].set_xlabel("Feature")
    ax[0].set_ylim(-0.5, 0.5)

    loading_scores["PC2"].plot(kind="bar", ax=ax[1])
    ax[1].set_title("Loading Scores for PC2")

    plt.tight_layout()
    plt.show()


def process_inventory(data, multifocal, location, scaler, X_scaled):
    # Load and preprocess inventory data similarly
    inventory_data = read_data("inventory*.csv", multifocal, location)

    X_inventory = inventory_data[get_features(multifocal)].fillna(0)
    X_inventory_scaled = scaler.transform(X_inventory)

    # Compute cluster centroids from dispensed data
    centroids = []
    for cluster_id in sorted(data["cluster"].unique(), key=int):
        cluster_points = X_scaled[data["cluster"] == cluster_id]
        centroids.append(cluster_points.mean(axis=0))
    centroids = np.array(centroids)

    # Assign each inventory item to the nearest centroid
    distances = np.sqrt(((X_inventory_scaled[:, None] - centroids) ** 2).sum(axis=2))
    nearest_cluster_indices = distances.argmin(axis=1)
    inventory_data["cluster"] = (nearest_cluster_indices + 1).astype(str)
    return inventory_data


def compare_clusters(dispense_data, unsuccessful_data, inventory_data):
    # Compute absolute and relative frequency in the dispensed data
    dispense_cluster_count = (
        dispense_data["cluster"].value_counts().rename("dispense_cluster_count")
    )
    # Compute absolute and relative frequency in the unsuccessful data
    unsuccessful_cluster_counts = (
        unsuccessful_data["cluster"].value_counts().rename("unsuccessful_cluster_count")
    )

    # Compute absolute and relative frequency in the inventory data
    inventory_cluster_count = (
        inventory_data["cluster"].value_counts().rename("inventory_cluster_count")
    )
    inventory_cluster_freq = (
        inventory_data["cluster"]
        .value_counts(normalize=True)
        .rename("inventory_cluster_frequency")
    )

    # Calculate the total count of data
    total_count = len(dispense_data) + len(unsuccessful_data)

    # Calculate the percentage of dispensed and unsuccessful glasses against the total count
    dispense_cluster_percent_total = (
        dispense_cluster_count / total_count * 100
    ).rename("dispense_cluster_percent_total")
    unsuccessful_cluster_percent_total = (
        unsuccessful_cluster_counts / total_count * 100
    ).rename("unsuccessful_cluster_percent_total")

    # Create a comparison DataFrame
    comparison_df = pd.DataFrame(
        {
            "dispense_cluster_count": dispense_cluster_count.astype(int),
            "unsuccessful_cluster_count": unsuccessful_cluster_counts.astype(int),
            "inventory_cluster_count": inventory_cluster_count.astype(int),
            "inventory_cluster_percent": inventory_cluster_freq * 100,
            "dispense_cluster_percent_total": dispense_cluster_percent_total,
            "unsuccessful_cluster_percent_total": unsuccessful_cluster_percent_total,
        }
    ).fillna(0)

    print(
        (
            comparison_df["dispense_cluster_count"]
            + comparison_df["unsuccessful_cluster_count"]
        ).astype(int)
    )
    return comparison_df


def plot_compared_clusters(comparison_df):
    # Plot cluster frequencies as percentages
    fig, ax = plt.subplots(figsize=(10, 4))

    # We'll plot a grouped bar chart
    bar_width = 0.2
    clusters = comparison_df.index
    x_positions = range(len(clusters))

    ax.bar(
        [x - bar_width for x in x_positions],
        comparison_df["dispense_cluster_percent_total"],
        width=bar_width,
        label="Dispensed (%)",
        color="green",
    )

    ax.bar(
        x_positions,
        comparison_df["inventory_cluster_percent"],
        width=bar_width,
        label="Inventory (%)",
        color="black",  # Color for inventory
    )

    ax.bar(
        [x + bar_width for x in x_positions],
        comparison_df["unsuccessful_cluster_percent_total"],
        width=bar_width,
        label="Unsuccessful (%)",
        color="red",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(clusters)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(
        "Comparison of Cluster Frequencies: Dispensed vs. Inventory vs. Unsuccessful"
    )
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_absolute_compared_clusters(comparison_df):
    fig, ax = plt.subplots(figsize=(10, 4))

    # We'll plot a grouped bar chart
    bar_width = 0.2
    clusters = comparison_df.index
    x_positions = range(len(clusters))

    ax.bar(
        [x - bar_width for x in x_positions],
        comparison_df["dispense_cluster_count"],
        width=bar_width,
        label="Dispensed (#)",
        color="green",
    )

    ax.bar(
        x_positions,
        comparison_df["inventory_cluster_count"],
        width=bar_width,
        label="Inventory (#)",
        color="black",  # Color for inventory
    )

    ax.bar(
        [x + bar_width for x in x_positions],
        comparison_df["unsuccessful_cluster_count"],
        width=bar_width,
        label="Unsuccessful (#)",
        color="red",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(clusters)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title(
        "Comparison of Cluster Counts: Dispensed vs. Inventory vs. Unsuccessful"
    )
    ax.legend()

    plt.tight_layout()
    plt.show()


def randomforest(data, multifocal):
    # Prepare the data
    X = data[get_features(multifocal)]
    y = data["cluster"]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled2 = scaler.fit_transform(X)

    # Get unique clusters
    clusters = y.unique()
    clusters.sort()

    # Determine the number of features
    num_features = len(X.columns)

    # Plot feature importances for each cluster
    fig, axes = plt.subplots(
        1, len(clusters), figsize=(len(clusters) * 5, num_features * 0.75)
    )

    for i, cluster in enumerate(clusters):
        # Create binary labels for the current cluster
        y_binary = (y == cluster).astype(int)

        # Train a Random Forest classifier
        rf = RandomForestClassifier(n_estimators=150)
        rf.fit(X_scaled2, y_binary)

        # Get feature importances
        feature_importances = rf.feature_importances_
        features = X.columns

        # Plot feature importances
        axes[i].barh(features, feature_importances)
        axes[i].set_xlabel("Feature Importance")
        axes[i].set_xlim(0, 0.5)
        axes[i].invert_yaxis()  # Reverse the y-axis
        if i == 0:
            axes[i].set_ylabel("Feature")
        else:
            axes[i].set_yticklabels([])
        axes[i].set_title(f"Feature Importances for Cluster {cluster}")

    plt.tight_layout(pad=1.0)
    plt.show()


def read_dispense(multifocal, location):
    data = read_data("dispense_report*.csv", multifocal, location)
    data = data[data["dispense type"] == "DISPENSED"]
    data = clean_data(data, multifocal)
    print(f"{len(data)} dispensed glasses loaded.")
    return data


def read_unsuccessful_searches(multifocal, location, dispense_data):
    data = read_data("unsuccessful_*.csv", multifocal, location)
    data = data.sort_values(by="Added date (in CST)")
    # necessary bc earlier data returned wrong results due to frontend bugs in reims2
    data = data[data["Added date (in CST)"] >= "2024-02-01"]
    data = data[data["isBal"] == "DISABLE_NONE"]
    data = data[data["highTolerance"] == False]
    data = remove_close_timestamps(data)
    data = clean_data(data, multifocal, dispense_data)
    print(f"{len(data)} unsuccessful searches loaded.")
    return data


def remove_close_timestamps(data):
    data["time_diff"] = data["Added date (in CST)"].diff().dt.total_seconds().abs()
    filtered_data = data[(data["time_diff"] > 150) | (data["time_diff"].isna())]

    removed_count = len(data) - len(filtered_data)
    filtered_data = filtered_data.drop(columns=["time_diff"])
    # print(
    # f"{removed_count} unsuccessful searches removed due to very close timestamps."
    # )

    return filtered_data


def remove_close_unsuccessful_searches(
    unsuccessful_data, combined_data, threshold=6 * 60
):
    unsuccessful_data["time_diff"] = (
        unsuccessful_data["Added date (in CST)"].diff().dt.total_seconds().abs()
    )
    filtered_unsuccessful_data = unsuccessful_data[
        (unsuccessful_data["time_diff"] > threshold)
        | (unsuccessful_data["time_diff"].isna())
    ]

    removed_count = len(unsuccessful_data) - len(filtered_unsuccessful_data)
    filtered_unsuccessful_data = filtered_unsuccessful_data.drop(columns=["time_diff"])
    # print(
    #     f"{removed_count} unsuccessful searches removed due to close timestamps and cluster."
    # )

    return filtered_unsuccessful_data, combined_data


def plot_cluster_feature_distributions(
    dispense_data, unsuccessful_data, multifocal, location
):
    features = get_features(multifocal)
    features = [
        feature.replace("OD Axis", "OD Axis Original").replace(
            "OS Axis", "OS Axis Original"
        )
        for feature in features
    ]
    combined_data = pd.concat([dispense_data, unsuccessful_data], ignore_index=True)
    clusters = combined_data["cluster"].unique()
    clusters.sort()

    for cluster in clusters:
        dispense_data_cluster = dispense_data[dispense_data["cluster"] == cluster]
        unsuccessful_data_cluster = unsuccessful_data[
            unsuccessful_data["cluster"] == cluster
        ]

        # Create subplots
        fig, axes = plt.subplots(2, len(features) // 2, figsize=(14, 8))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]

            # Determine bin width and fixed min/max based on feature type
            if "Axis" in feature:
                bin_width = 10
                min_val, max_val = 0, 180
                xticks = np.arange(min_val, max_val + bin_width, 1 * bin_width)
                xtick_labels = [
                    f"{x:.0f}" if i % 2 == 0 else "" for i, x in enumerate(xticks)
                ]
            elif "Sphere" in feature:
                bin_width = 0.25
                min_val, max_val = -4, 4
                xticks = np.arange(min_val, max_val + bin_width, 0.5)
                xtick_labels = [
                    f"{x:.1f}" if i % 4 == 0 else "" for i, x in enumerate(xticks)
                ]
            elif "Cylinder" in feature:
                bin_width = 0.25
                min_val, max_val = -3, 0
                xticks = np.arange(min_val, max_val + bin_width, 0.25)
                xtick_labels = [
                    f"{x:.1f}" if i % 2 == 0 else "" for i, x in enumerate(xticks)
                ]
            elif "Add" in feature:
                bin_width = 0.25
                min_val, max_val = 0, 4
                xticks = np.arange(min_val, max_val + bin_width, 0.25)
                xtick_labels = [
                    f"{x:.1f}" if i % 2 == 0 else "" for i, x in enumerate(xticks)
                ]

            bins = np.arange(min_val, max_val + 2 * bin_width, bin_width)

            # Plot total distribution
            ax.hist(
                combined_data[feature],
                bins=bins,
                color="gray",
                alpha=0.3,
                label="Total",
                align="mid" if "Axis" in feature else "left",
            )

            # Plot dispense and unsuccessful data as stacked bar graphs
            ax.hist(
                [dispense_data_cluster[feature], unsuccessful_data_cluster[feature]],
                bins=bins,
                stacked=True,
                color=["green", "red"],
                label=[
                    f"Dispense",
                    f"Unsuccessful",
                ],
                align="mid" if "Axis" in feature else "left",
            )

            ax.set_title(feature)
            ax.legend()

            # Set xticks based on feature type
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
            # plt.setp(ax.get_xticklabels(), rotation=45, ha="left")

        plt.suptitle(f"Feature Distributions for Cluster {cluster}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"plots/features_{location}_{multifocal}_{cluster}.png")
        plt.show()


def launch(multifocal, location, cluster_count):
    print(
        f"Analyzing {location} data for {'multifocal' if multifocal else 'single'} glasses."
    )
    dispense_data = read_dispense(multifocal, location)
    unsuccessful_data = read_unsuccessful_searches(multifocal, location, dispense_data)

    # Reset indices to ensure alignment
    dispense_data = dispense_data.reset_index(drop=True)
    unsuccessful_data = unsuccessful_data.reset_index(drop=True)

    combined_data = pd.concat([dispense_data, unsuccessful_data], ignore_index=True)

    X_scaled, scaler = scale_data(combined_data, multifocal)
    clustering(combined_data, X_scaled, num_clusters=cluster_count)

    # Assign cluster labels back to dispense_data and unsuccessful_data
    dispense_data["cluster"] = combined_data.loc[
        : len(dispense_data) - 1, "cluster"
    ].values
    unsuccessful_data["cluster"] = combined_data.loc[
        len(dispense_data) :, "cluster"
    ].values

    # Remove unsuccessful searches that were relatively close in time
    unsuccessful_data, combined_data = remove_close_unsuccessful_searches(
        unsuccessful_data, combined_data
    )
    inventory_data = process_inventory(
        combined_data, multifocal, location, scaler, X_scaled
    )

    comparison = compare_clusters(dispense_data, unsuccessful_data, inventory_data)
    plot_compared_clusters(comparison)

    randomforest(combined_data, multifocal=multifocal)

    plot_cluster_feature_distributions(
        dispense_data, unsuccessful_data, multifocal, location
    )

    plot_absolute_compared_clusters(comparison)

    # return combined_data, inventory_data
