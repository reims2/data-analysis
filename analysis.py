import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

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
    return data


def clean_data(data, multifocal):
    X = data[get_features(multifocal)]

    # Handle missing values (if any)
    X.fillna(0, inplace=True)

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Calculate Z-scores
    z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0))

    # Filter out rows where any Z-score is above the threshold (e.g., 3)
    threshold = 3.5
    new_data = data[(z_scores < threshold).all(axis=1)]
    print("Number of filtered glasses:", len(data) - len(new_data))
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
        p=num_clusters * 2,
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


def compare_clusters(data, inventory_data):
    # Compute absolute and relative frequency in the dispensed data
    dispense_cluster_count = (
        data["cluster"].value_counts().rename("dispense_cluster_count")
    )
    dispense_cluster_freq = (
        data["cluster"]
        .value_counts(normalize=True)
        .rename("dispense_cluster_frequency")
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

    # Create a comparison DataFrame
    comparison_df = pd.DataFrame(
        {
            "dispense_cluster_count": dispense_cluster_count,
            "dispense_cluster_percent": dispense_cluster_freq * 100,
            "inventory_cluster_count": inventory_cluster_count,
            "inventory_cluster_percent": inventory_cluster_freq * 100,
        }
    ).fillna(0)

    print(comparison_df["dispense_cluster_count"])

    # Plot cluster frequencies as percentages
    fig, ax = plt.subplots(figsize=(10, 4))

    # We'll plot a grouped bar chart
    bar_width = 0.3
    clusters = comparison_df.index
    x_positions = range(len(clusters))

    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        comparison_df["dispense_cluster_percent"],
        width=bar_width,
        label="Dispensed (%)",
    )

    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        comparison_df["inventory_cluster_percent"],
        width=bar_width,
        label="Inventory (%)",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(clusters)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Comparison of Cluster Frequencies: Dispensed vs. Inventory")
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
    return data


def read_unsuccessful_searches(multifocal, location):
    data = read_data("unsuccessful_*.csv", multifocal, location)
    data = clean_data(data, multifocal)
    data = remove_close_timestamps(data, "Added date (in CST)")
    return data


def remove_close_timestamps(data, time_column):
    # Convert the time column to datetime
    data[time_column] = pd.to_datetime(data[time_column])

    data = data.sort_values(by=time_column)
    data["time_diff"] = data[time_column].diff().dt.total_seconds().abs()
    filtered_data = data[(data["time_diff"] > 60) | (data["time_diff"].isna())]

    removed_count = len(data) - len(filtered_data)
    filtered_data = filtered_data.drop(columns=["time_diff"])
    print(
        f"Number of removed unsuccessful searches that were very close in time: {removed_count}"
    )

    return filtered_data


def launch(multifocal, location, cluster_count):
    data = read_dispense(multifocal, location)
    unsucc_data = read_unsuccessful_searches(multifocal, location)
    data = pd.concat([data, unsucc_data], ignore_index=True)
    print("Number of glasses used for clustering:", len(data))
    X_scaled, scaler = scale_data(data, multifocal)
    clustering(data, X_scaled, num_clusters=cluster_count)
    # pca(data, multifocal)
    inventory_data = process_inventory(data, multifocal, location, scaler, X_scaled)
    compare_clusters(data, inventory_data)
    randomforest(data, multifocal=multifocal)
    return data, inventory_data
