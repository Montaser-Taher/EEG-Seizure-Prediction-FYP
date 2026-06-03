import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt


# Allows Python to find local project files if needed
sys.path.append(str(Path(__file__).resolve().parent))


# APP SETTINGS

st.set_page_config(
    page_title="EEG Seizure Prediction Monitor",
    layout="wide"
)


# CUSTOM CSS

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }

    .big-title {
        font-size: 38px;
        font-weight: 800;
        color: #0b1f3a;
    }

    .subtitle {
        font-size: 18px;
        color: #4b5563;
    }

    .status-card {
        padding: 22px;
        border-radius: 18px;
        background-color: white;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
        text-align: center;
    }

    .safe {
        color: #15803d;
        font-size: 28px;
        font-weight: 800;
    }

    .warning {
        color: #ca8a04;
        font-size: 28px;
        font-weight: 800;
    }

    .danger {
        color: #dc2626;
        font-size: 28px;
        font-weight: 800;
    }

    .small-text {
        font-size: 14px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# HELPER FUNCTIONS

@st.cache_resource
def load_cnn_model(model_path):
    # Loads the trained Keras model only once for better performance
    model = tf.keras.models.load_model(model_path)
    return model


def load_npz_file(uploaded_file):
    # Loads processed EEG windows and labels from a .npz file
    data = np.load(uploaded_file)
    X = data["X"]

    if "y" in data:
        y = data["y"]
    else:
        y = None

    return X, y


def prepare_for_cnn_single_window(window):
    # Converts one EEG window from (channels, timepoints) to (1, timepoints, channels)
    window = np.asarray(window, dtype=np.float32)
    window = np.transpose(window, (1, 0))
    window = np.expand_dims(window, axis=0)
    return window


def prepare_for_cnn_batch(X):
    # Converts a batch from (windows, channels, timepoints) to (windows, timepoints, channels)
    X = np.asarray(X, dtype=np.float32)
    X = np.transpose(X, (0, 2, 1))
    return X


def get_risk_status(probability, threshold):
    # Converts model probability into a readable risk level
    if probability >= threshold:
        return "HIGH RISK", "danger"
    elif probability >= threshold * 0.7:
        return "MEDIUM RISK", "warning"
    else:
        return "LOW RISK", "safe"


def plot_live_probability(probabilities, threshold):
    # Plots prediction probabilities over time
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(probabilities, linewidth=2)
    ax.axhline(threshold, linestyle="--", label="Threshold")

    ax.set_ylim(0, 1)
    ax.set_title("Preictal Probability Timeline")
    ax.set_xlabel("EEG Window")
    ax.set_ylabel("Preictal Probability")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def plot_single_eeg_window(window):
    # Plots a small preview of EEG channels from one window
    fig, ax = plt.subplots(figsize=(12, 4))

    channels_to_plot = min(5, window.shape[0])

    for i in range(channels_to_plot):
        ax.plot(window[i] + i * 5, linewidth=0.8)

    ax.set_title("EEG Window Preview")
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("EEG Channels")
    ax.grid(True)

    st.pyplot(fig)


# SIDEBAR

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Upload & Preview",
        "Run Offline Prediction",
        "Live Simulation Monitor",
        "Model Explanation"
    ]
)

st.sidebar.markdown("---")

model_path = st.sidebar.text_input(
    "Model path",
    value="models/baseline_cnn_prediction.keras"
)

threshold = st.sidebar.slider(
    "Prediction threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

simulation_speed = st.sidebar.slider(
    "Live simulation speed",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

max_live_windows = st.sidebar.slider(
    "Number of windows to simulate",
    min_value=10,
    max_value=300,
    value=80,
    step=10
)


# SESSION STATE

if "X" not in st.session_state:
    st.session_state.X = None

if "y" not in st.session_state:
    st.session_state.y = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "predicted_classes" not in st.session_state:
    st.session_state.predicted_classes = None


# HOME PAGE

if page == "Home":

    st.markdown(
        '<div class="big-title">EEG Seizure Prediction Monitoring System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        This interface demonstrates how an EEG-based seizure prediction model can be used
        for offline testing and simulated real-time monitoring.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="status-card">
                <h3>Offline Testing</h3>
                <p>Upload processed EEG files and test the trained model.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="status-card">
                <h3>Live Simulation</h3>
                <p>Simulate incoming EEG windows one by one.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="status-card">
                <h3>Risk Alert</h3>
                <p>Display low, medium, or high preictal risk.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("System Flow")

    st.code(
        """
Processed EEG file (.npz)
        ↓
Load EEG windows
        ↓
Prepare input for the model
        ↓
Apply trained model
        ↓
Display preictal probability
        ↓
Show risk level
        """
    )

    st.info(
        """
        This interface is not connected to a real EEG headset. It simulates real-time
        prediction by reading one EEG window at a time from processed EEG files.
        """
    )


# UPLOAD PAGE

elif page == "Upload & Preview":

    st.title("Upload & Preview EEG Data")

    uploaded_file = st.file_uploader(
        "Upload a processed EEG .npz file",
        type=["npz"]
    )

    if uploaded_file is not None:

        X, y = load_npz_file(uploaded_file)

        st.session_state.X = X
        st.session_state.y = y

        st.success("File loaded successfully.")

        st.subheader("Data Shape")

        col1, col2, col3 = st.columns(3)

        col1.metric("Windows", X.shape[0])
        col2.metric("Channels", X.shape[1])
        col3.metric("Timepoints", X.shape[2])

        st.write("X shape:", X.shape)

        if y is not None:
            st.write("y shape:", y.shape)

            unique_labels, counts = np.unique(y, return_counts=True)

            label_df = pd.DataFrame({
                "Label": unique_labels,
                "Count": counts
            })

            label_df["Meaning"] = label_df["Label"].map({
                0: "Interictal",
                1: "Preictal"
            })

            st.subheader("Label Distribution")
            st.dataframe(label_df)

        else:
            st.warning("No labels found in this file.")

        st.subheader("Quality Check")

        st.write("NaN values:", np.isnan(X).any())
        st.write("Infinite values:", np.isinf(X).any())
        st.write("Minimum value:", float(np.min(X)))
        st.write("Maximum value:", float(np.max(X)))

        st.subheader("First EEG Window Preview")

        if X.ndim == 3:
            plot_single_eeg_window(X[0])


# OFFLINE PREDICTION PAGE

elif page == "Run Offline Prediction":

    st.title("Run Offline CNN Prediction")

    if st.session_state.X is None:
        st.warning("Please upload a processed .npz file first.")

    else:
        X = st.session_state.X

        st.subheader("Input Check")
        st.write("Original X shape:", X.shape)

        if X.ndim != 3:
            st.error("X must be 3D: (windows, channels, timepoints)")

        elif X.shape[1] != 17:
            st.error("Expected 17 EEG channels.")

        else:
            st.success("Input shape looks correct.")

            X_cnn = prepare_for_cnn_batch(X)

            st.write("CNN input shape after transpose:", X_cnn.shape)

            if st.button("Run Offline Prediction"):

                try:
                    model = load_cnn_model(model_path)

                    probabilities = model.predict(X_cnn).flatten()
                    predicted_classes = (probabilities >= threshold).astype(int)

                    st.session_state.probabilities = probabilities
                    st.session_state.predicted_classes = predicted_classes

                    st.success("Prediction completed.")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            if st.session_state.probabilities is not None:

                probabilities = st.session_state.probabilities
                predicted_classes = st.session_state.predicted_classes

                preictal_count = np.sum(predicted_classes == 1)
                interictal_count = np.sum(predicted_classes == 0)

                st.subheader("Prediction Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric("Interictal Windows", int(interictal_count))
                col2.metric("Preictal Windows", int(preictal_count))
                col3.metric("Average Preictal Probability", f"{np.mean(probabilities):.3f}")

                st.subheader("Prediction Timeline")
                plot_live_probability(probabilities, threshold)

                result_df = pd.DataFrame({
                    "Window": np.arange(len(probabilities)),
                    "Preictal Probability": probabilities,
                    "Predicted Class": predicted_classes
                })

                result_df["Meaning"] = result_df["Predicted Class"].map({
                    0: "Interictal",
                    1: "Preictal"
                })

                st.subheader("Prediction Table")
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="offline_prediction_results.csv",
                    mime="text/csv"
                )


# LIVE SIMULATION PAGE

elif page == "Live Simulation Monitor":

    st.title("Live EEG Prediction Simulation")

    st.info(
        """
        This page simulates real-time seizure prediction by sending one EEG window
        at a time to the trained model.
        """
    )

    if st.session_state.X is None:
        st.warning("Please upload a processed .npz file first.")

    else:
        X = st.session_state.X
        y = st.session_state.y

        if X.ndim != 3:
            st.error("X must be 3D: (windows, channels, timepoints).")

        elif X.shape[1] != 17:
            st.error("Expected 17 EEG channels for this model.")

        else:
            st.success("EEG data is ready for live simulation.")

            total_windows = min(max_live_windows, X.shape[0])

            col_start, col_info = st.columns([1, 3])

            with col_start:
                start_simulation = st.button("Start Live Simulation")

            with col_info:
                st.write(f"Simulation will run for **{total_windows} windows**.")

            status_placeholder = st.empty()
            metric_placeholder = st.empty()
            eeg_placeholder = st.empty()
            chart_placeholder = st.empty()
            table_placeholder = st.empty()
            alert_placeholder = st.empty()

            if start_simulation:

                try:
                    model = load_cnn_model(model_path)

                    live_probabilities = []
                    live_classes = []
                    live_rows = []

                    for i in range(total_windows):

                        current_window = X[i]
                        X_live = prepare_for_cnn_single_window(current_window)

                        probability = float(model.predict(X_live, verbose=0).flatten()[0])
                        predicted_class = int(probability >= threshold)

                        live_probabilities.append(probability)
                        live_classes.append(predicted_class)

                        risk_text, risk_class = get_risk_status(probability, threshold)

                        if y is not None:
                            true_label = int(y[i])
                        else:
                            true_label = None

                        live_rows.append({
                            "Window": i,
                            "Preictal Probability": probability,
                            "Predicted Class": predicted_class,
                            "Prediction Meaning": "Preictal" if predicted_class == 1 else "Interictal",
                            "True Label": true_label
                        })

                        with status_placeholder.container():

                            st.markdown("### Patient EEG Monitoring Status")

                            st.markdown(
                                f"""
                                <div class="status-card">
                                    <div class="{risk_class}">{risk_text}</div>
                                    <div class="small-text">Current Window: {i + 1} / {total_windows}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with metric_placeholder.container():

                            col1, col2, col3, col4 = st.columns(4)

                            col1.metric("Current Probability", f"{probability:.3f}")
                            col2.metric("Threshold", f"{threshold:.2f}")
                            col3.metric("Predicted Class", predicted_class)
                            col4.metric("Preictal Count", int(np.sum(np.array(live_classes) == 1)))

                        with eeg_placeholder.container():

                            st.subheader("Live EEG Signal Window")
                            plot_single_eeg_window(current_window)

                        with chart_placeholder.container():

                            st.subheader("Live Prediction Timeline")
                            plot_live_probability(live_probabilities, threshold)

                        with table_placeholder.container():

                            live_df = pd.DataFrame(live_rows)
                            st.subheader("Live Prediction Log")
                            st.dataframe(live_df.tail(10), use_container_width=True)

                        with alert_placeholder.container():

                            if probability >= threshold:
                                st.error("ALERT: High preictal probability detected. The patient may be entering a preictal state.")
                            elif probability >= threshold * 0.7:
                                st.warning("Warning: Probability is increasing. Continue monitoring.")
                            else:
                                st.success("Stable: Current EEG window is mostly interictal.")

                        time.sleep(simulation_speed)

                    final_df = pd.DataFrame(live_rows)

                    st.success("Live simulation completed.")

                    csv = final_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Live Simulation Results",
                        data=csv,
                        file_name="live_simulation_results.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Live simulation failed: {e}")


# MODEL EXPLANATION PAGE

elif page == "Model Explanation":

    st.title("Model Explanation")

    st.info(
        """
        This page explains how the EEG seizure prediction system works, from processed EEG windows
        to model prediction and risk monitoring.
        """
    )

    st.subheader("1. What the Model Predicts")

    st.write(
        """
        The system predicts whether an EEG window is **interictal** or **preictal**.

        **Interictal** means normal brain activity between seizures.

        **Preictal** means the period before a seizure, where the EEG signal may contain
        patterns that suggest a seizure could happen soon.
        """
    )

    st.code(
        """
0 = Interictal
1 = Preictal
        """
    )

    st.subheader("2. Input Data")

    st.write(
        """
        The model receives processed EEG windows. Each window represents a short segment
        of EEG signal.

        In this project, each EEG window uses:

        - 17 selected EEG channels
        - 1280 timepoints
        - 5 seconds of EEG signal
        - 256 Hz sampling rate
        """
    )

    st.code(
        """
Original processed shape:
(windows, channels, timepoints)

Model input shape:
(windows, timepoints, channels)
        """
    )

    st.warning(
        """
        The transpose step is important because the CNN expects the time dimension first,
        followed by the EEG channels.
        """
    )

    st.subheader("3. Preprocessing Pipeline")

    st.write(
        """
        Before training or prediction, the raw EEG recordings are cleaned and converted
        into a format suitable for machine learning.
        """
    )

    st.code(
        """
Raw EDF EEG files
        ↓
Load EEG recordings
        ↓
Apply notch filter to reduce powerline noise
        ↓
Apply bandpass filter to keep useful EEG frequencies
        ↓
Select FINAL_17 common EEG channels
        ↓
Normalize EEG signals
        ↓
Split EEG into overlapping 5-second windows
        ↓
Assign labels: interictal or preictal
        ↓
Save processed data as .npz files
        """
    )

    st.subheader("4. Baseline CNN Model")

    st.write(
        """
        The baseline model is a 1D Convolutional Neural Network.

        The CNN learns local EEG signal patterns such as waveform changes, short oscillations,
        and abnormal signal behaviour that may be related to the preictal stage.
        """
    )

    st.code(
        """
EEG Window
        ↓
Conv1D layers
        ↓
Batch normalization
        ↓
Max pooling
        ↓
Dropout
        ↓
Dense layer
        ↓
Sigmoid output
        ↓
Preictal probability
        """
    )

    st.subheader("5. CNN + Frequency Model")

    st.write(
        """
        In addition to the baseline CNN, the project also uses frequency-domain features.

        EEG signals contain useful information in different frequency ranges. Because of this,
        frequency features are extracted and combined with the CNN output to support prediction.
        """
    )

    st.code(
        """
EEG window
        ↓
CNN branch learns time-domain patterns

Frequency features
        ↓
Dense branch learns spectral information

CNN features + frequency features
        ↓
Final prediction
        """
    )

    st.subheader("6. CNN-LSTM Sequence Model")

    st.write(
        """
        The final model improves on the CNN by adding LSTM layers.

        A CNN looks at one EEG window at a time. The CNN-LSTM model looks at a sequence of EEG
        windows, which helps it learn how EEG patterns change over time before a seizure.
        """
    )

    st.code(
        """
Sequence of EEG windows
        ↓
CNN extracts features from each window
        ↓
LSTM learns changes across time
        ↓
Frequency features are combined
        ↓
Final preictal probability
        """
    )

    st.subheader("7. Prediction Threshold")

    st.write(
        """
        The model outputs a probability between 0 and 1.

        A threshold is then used to convert this probability into a final prediction.
        """
    )

    st.code(
        """
If probability >= threshold:
    Prediction = Preictal

If probability < threshold:
    Prediction = Interictal
        """
    )

    st.write(
        """
        In the interface, the threshold can be changed from the sidebar.
        A lower threshold may detect more possible preictal windows, while a higher threshold
        may reduce false alarms.
        """
    )

    st.subheader("8. Risk Levels")

    st.write(
        """
        To make the output easier to understand, the system converts the probability into
        a simple risk level.
        """
    )

    st.code(
        """
Low Risk:
    Probability is clearly below the threshold

Medium Risk:
    Probability is getting close to the threshold

High Risk:
    Probability is equal to or above the threshold
        """
    )

    st.subheader("9. Offline Prediction and Live Simulation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Offline Prediction

            In offline prediction, the system processes all EEG windows at once.

            This is useful for:
            - testing the model
            - analysing complete EEG files
            - exporting prediction results
            """
        )

    with col2:
        st.markdown(
            """
            ### Live Simulation

            In live simulation, the system reads one EEG window at a time.

            This simulates:
            - real-time monitoring
            - continuous probability updates
            - warning alerts when risk increases
            """
        )

    st.subheader("10. Important Limitation")

    st.warning(
        """
        This interface is currently a simulated real-time system. It does not connect directly
        to a physical EEG headset yet.

        Instead, it uses preprocessed EEG files and sends the windows to the model one by one
        to demonstrate how real-time seizure prediction could work.
        """
    )

    st.subheader("11. Summary")

    st.success(
        """
        Overall, this interface demonstrates how the trained EEG seizure prediction models can
        be used for offline testing and simulated real-time monitoring. It connects the machine
        learning part of the project with a practical interface that shows predictions,
        probabilities, EEG signals, and risk alerts.
        """
    )