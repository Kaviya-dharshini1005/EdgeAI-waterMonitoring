import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_FILE = "Dataset/water_potability.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("WATER QUALITY AUTOENCODER TRAINING")
print("=" * 60)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

# Remove unwanted Excel/CSV columns such as Unnamed: 1
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. SELECT FEATURES
# ============================================================

cols = ["ph", "TDS", "Turbidity"]

# Make sure required columns exist
for col in cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

if "Potability" not in df.columns:
    raise ValueError("Missing 'Potability' column")


# ============================================================
# 4. SELECT NORMAL WATER ONLY
# ============================================================

# IMPORTANT:
# The autoencoder learns the pattern of NORMAL water.
#
# Potability == 1 is being treated as the normal/safe class
# according to our current dataset labeling.

normal = df[df["Potability"] == 1][cols].dropna()

print("\nNormal water samples:", len(normal))

if len(normal) == 0:
    raise ValueError("No normal water samples found.")


# ============================================================
# 5. NORMALIZATION
# ============================================================

print("\nNormalizing data to 0-1...")

scaler = MinMaxScaler()

X = scaler.fit_transform(normal).astype(np.float32)

print("Input shape:", X.shape)

print("\nNormalization parameters:")

for i, col in enumerate(cols):
    print(
        f"{col}: "
        f"min={scaler.data_min_[i]:.6f}, "
        f"max={scaler.data_max_[i]:.6f}"
    )


# ============================================================
# 6. AUTOENCODER ARCHITECTURE
# ============================================================

print("\n" + "=" * 60)
print("AUTOENCODER ARCHITECTURE")
print("=" * 60)

print("""
Input:       3
Encoder:     3 -> 8 -> 4 -> 2
Decoder:     2 -> 4 -> 8 -> 3

Complete architecture:

3 -> 8 -> 4 -> 2 -> 4 -> 8 -> 3

Latent/Bottleneck layer = 2 neurons
""")


# ============================================================
# 7. BUILD MODEL
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),

    # Encoder
    tf.keras.layers.Dense(8, activation="relu", name="encoder_8"),
    tf.keras.layers.Dense(4, activation="relu", name="encoder_4"),
    tf.keras.layers.Dense(2, activation="relu", name="latent"),

    # Decoder
    tf.keras.layers.Dense(4, activation="relu", name="decoder_4"),
    tf.keras.layers.Dense(8, activation="relu", name="decoder_8"),
    tf.keras.layers.Dense(3, activation="linear", name="output")
])


# ============================================================
# 8. COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

print("\nModel summary:")
model.summary()


# ============================================================
# 9. TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = model.fit(
    X,
    X,
    epochs=500,
    batch_size=32,
    shuffle=True,
    verbose=1
)


# ============================================================
# 10. TRAINING LOSS
# ============================================================

losses = history.history["loss"]

final_loss = losses[-1]

print("\nFinal training loss:", final_loss)


# ============================================================
# 11. CALCULATE NORMAL RECONSTRUCTION ERROR
# ============================================================

print("\nCalculating reconstruction errors...")

reconstructed = model.predict(X, verbose=0)

errors = np.mean(
    np.square(X - reconstructed),
    axis=1
)

mean_error = np.mean(errors)
std_error = np.std(errors)

# Conservative anomaly threshold
threshold = mean_error + 3 * std_error

print("\nNormal water reconstruction error:")
print(f"Mean = {mean_error:.8f}")
print(f"Std  = {std_error:.8f}")

print(f"\nRecommended anomaly threshold = {threshold:.8f}")


# ============================================================
# 12. TEST UNSAFE WATER
# ============================================================

unsafe = df[df["Potability"] == 0][cols].dropna()

print("\nUnsafe samples:", len(unsafe))

if len(unsafe) > 0:

    X_unsafe = scaler.transform(
        unsafe
    ).astype(np.float32)

    unsafe_reconstructed = model.predict(
        X_unsafe,
        verbose=0
    )

    unsafe_errors = np.mean(
        np.square(X_unsafe - unsafe_reconstructed),
        axis=1
    )

    detection_rate = np.mean(
        unsafe_errors > threshold
    ) * 100

    print("\nUnsafe water reconstruction error:")
    print(f"Mean = {np.mean(unsafe_errors):.8f}")
    print(f"Std  = {np.std(unsafe_errors):.8f}")

    print(
        f"\nDetection rate = {detection_rate:.2f}%"
    )

else:
    unsafe_errors = np.array([])

    print(
        "\nNo unsafe samples available for testing."
    )


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "water_autoencoder.keras"
)

model.save(model_path)

print("\nTensorFlow model saved:")
print(model_path)


# ============================================================
# 14. EXPORT WEIGHTS AND BIASES FOR STM32
# ============================================================

print("\n" + "=" * 60)
print("EXPORTING MODEL PARAMETERS")
print("=" * 60)


def write_matrix(f, name, array):

    rows, cols = array.shape

    f.write(
        f"static const float {name}[{rows}][{cols}] = {{\n"
    )

    for i in range(rows):

        values = ", ".join(
            f"{value:.8f}f"
            for value in array[i]
        )

        comma = "," if i < rows - 1 else ""

        f.write(
            f"    {{{values}}}{comma}\n"
        )

    f.write("};\n\n")


def write_vector(f, name, array):

    size = len(array)

    f.write(
        f"static const float {name}[{size}] = {{\n"
    )

    values = ", ".join(
        f"{value:.8f}f"
        for value in array
    )

    f.write(
        f"    {values}\n"
    )

    f.write("};\n\n")


weights_path = os.path.join(
    OUTPUT_DIR,
    "weights.h"
)


with open(weights_path, "w") as f:

    f.write("#ifndef WEIGHTS_H\n")
    f.write("#define WEIGHTS_H\n\n")

    f.write(
        "/* Autoencoder: 3 -> 8 -> 4 -> 2 -> 4 -> 8 -> 3 */\n\n"
    )

    # --------------------------------------------------------
    # Encoder 3 -> 8
    # --------------------------------------------------------

    w1, b1 = model.get_layer(
        "encoder_8"
    ).get_weights()

    write_matrix(
        f,
        "ae_enc_w1",
        w1
    )

    write_vector(
        f,
        "ae_enc_b1",
        b1
    )


    # --------------------------------------------------------
    # Encoder 8 -> 4
    # --------------------------------------------------------

    w2, b2 = model.get_layer(
        "encoder_4"
    ).get_weights()

    write_matrix(
        f,
        "ae_enc_w2",
        w2
    )

    write_vector(
        f,
        "ae_enc_b2",
        b2
    )


    # --------------------------------------------------------
    # Encoder 4 -> 2
    # --------------------------------------------------------

    w3, b3 = model.get_layer(
        "latent"
    ).get_weights()

    write_matrix(
        f,
        "ae_enc_w3",
        w3
    )

    write_vector(
        f,
        "ae_enc_b3",
        b3
    )


    # --------------------------------------------------------
    # Decoder 2 -> 4
    # --------------------------------------------------------

    w4, b4 = model.get_layer(
        "decoder_4"
    ).get_weights()

    write_matrix(
        f,
        "ae_dec_w1",
        w4
    )

    write_vector(
        f,
        "ae_dec_b1",
        b4
    )


    # --------------------------------------------------------
    # Decoder 4 -> 8
    # --------------------------------------------------------

    w5, b5 = model.get_layer(
        "decoder_8"
    ).get_weights()

    write_matrix(
        f,
        "ae_dec_w2",
        w5
    )

    write_vector(
        f,
        "ae_dec_b2",
        b5
    )


    # --------------------------------------------------------
    # Decoder 8 -> 3
    # --------------------------------------------------------

    w6, b6 = model.get_layer(
        "output"
    ).get_weights()

    write_matrix(
        f,
        "ae_dec_w3",
        w6
    )

    write_vector(
        f,
        "ae_dec_b3",
        b6
    )

    f.write("#endif\n")


print("\nWeights and biases exported:")
print(weights_path)


# ============================================================
# 15. EXPORT NORMALIZATION PARAMETERS
# ============================================================

normalization_path = os.path.join(
    OUTPUT_DIR,
    "normalization.h"
)

with open(normalization_path, "w") as f:

    f.write("#ifndef NORMALIZATION_H\n")
    f.write("#define NORMALIZATION_H\n\n")

    f.write(
        f"#define PH_MIN {scaler.data_min_[0]:.8f}f\n"
    )

    f.write(
        f"#define PH_MAX {scaler.data_max_[0]:.8f}f\n\n"
    )

    f.write(
        f"#define TDS_MIN {scaler.data_min_[1]:.8f}f\n"
    )

    f.write(
        f"#define TDS_MAX {scaler.data_max_[1]:.8f}f\n\n"
    )

    f.write(
        f"#define TURBIDITY_MIN {scaler.data_min_[2]:.8f}f\n"
    )

    f.write(
        f"#define TURBIDITY_MAX {scaler.data_max_[2]:.8f}f\n\n"
    )

    f.write("#endif\n")


print(
    "Normalization parameters exported:"
)
print(normalization_path)


# ============================================================
# 16. EXPORT ANOMALY THRESHOLD
# ============================================================

threshold_path = os.path.join(
    OUTPUT_DIR,
    "threshold.h"
)

with open(threshold_path, "w") as f:

    f.write("#ifndef THRESHOLD_H\n")
    f.write("#define THRESHOLD_H\n\n")

    f.write(
        f"#define ANOMALY_THRESHOLD {threshold:.8f}f\n\n"
    )

    f.write("#endif\n")


print(
    "Anomaly threshold exported:"
)
print(threshold_path)


# ============================================================
# 17. PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.title(
    "Autoencoder Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.grid(True)

loss_plot = os.path.join(
    OUTPUT_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. PLOT RECONSTRUCTION ERRORS
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    errors,
    bins=30,
    alpha=0.7,
    label="Normal Water"
)

if len(unsafe_errors) > 0:

    plt.hist(
        unsafe_errors,
        bins=30,
        alpha=0.7,
        label="Unsafe Water"
    )

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.6f}"
)

plt.xlabel(
    "Reconstruction Error (MSE)"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Water Quality Anomaly Detection"
)

plt.legend()

error_plot = os.path.join(
    OUTPUT_DIR,
    "reconstruction_errors.png"
)

plt.savefig(
    error_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 19. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("""
Architecture:
3 -> 8 -> 4 -> 2 -> 4 -> 8 -> 3

Input features:
1. pH
2. TDS
3. Turbidity

Latent space:
2 neurons

Activation:
ReLU in hidden layers
Linear in output layer

Training:
Normal water samples only

Output files:
outputs/
    water_autoencoder.keras
    weights.h
    normalization.h
    threshold.h
    training_loss.png
    reconstruction_errors.png
""")

print("=" * 60)