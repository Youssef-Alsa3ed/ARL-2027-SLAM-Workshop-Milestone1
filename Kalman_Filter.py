import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import Filter_visualizer


class KalmanFilter2D:

    def __init__(
        self,
        gps_measurement_var,
        speedometere_measurement_var,
        position_process_noise,
        velocity_process_noise,
        dt=1,
    ):
        self.dt = dt
        # state transition
        self.A = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        # acceleration
        acc = 0.5 * dt**2
        # control matrix
        self.B = np.array([[acc, 0], [0, acc], [dt, 0], [0, dt]], dtype=float)
        # measurement matrix
        self.H = np.eye(4)
        # process noise covariance
        self.Q = np.diag(
            [
                position_process_noise,
                position_process_noise,
                velocity_process_noise,
                velocity_process_noise,
            ]
        ).astype(float)
        # measurement noise covariance matrix
        self.R = np.diag(
            [
                gps_measurement_var,
                gps_measurement_var,
                speedometere_measurement_var,
                speedometere_measurement_var,
            ]
        ).astype(float)


    def predict(self, X_past, P_past, U):
        X_pred = self.A @ X_past + self.B @ U
        P_pred = self.A @ P_past @ self.A.T + self.Q
        return X_pred, P_pred

    def correct(self, X_pred, P_pred, Z):
        # Innovation Residual
        y = Z - self.H @ X_pred
        # Innovation Covariance
        s = self.H @ P_pred @ self.H.T + self.R
        # Kalman Gain
        k = P_pred @ self.H.T @ np.linalg.inv(s)
        # State Update 
        X_update = X_pred + k @ y
        # Covariance Update
        P_update = (np.eye(4) - k @ self.H) @ P_pred
        return X_update, P_update


if __name__ == "__main__":
 
    scenarios = [
        {
            "name": "1. Noisy Prediction",
            "x0": [0, 0, 10, 5],
            "p0_pos": 1.0, "p0_vel": 1.0,
            "u": [2.0, 1.0],
            "z": [10.5, 5.2, 9.8, 4.9],
            "q_pos": 3.0, "q_vel": 3.0,
            "r_gps": 2.0, "r_speedo": 2.0,
        },
        {
            "name": "2. Noisy Measurement",
            "x0": [0, 0, 10, 5],
            "p0_pos": 1.0, "p0_vel": 1.0,
            "u": [2.0, 1.0],
            "z": [10.5, 5.2, 12.0, 6.0],
            "q_pos": 0.05, "q_vel": 0.1,
            "r_gps": 9.0, "r_speedo": 9.0,
        },
        {
            "name": "3. GPS Outlier Glitch",
            "x0": [0, 0, 10, 5],
            "p0_pos": 1.0, "p0_vel": 1.0,
            "u": [2.0, 1.0],
            "z": [35.0, 28.0, 10.1, 5.0],
            "q_pos": 0.05, "q_vel": 0.1,
            "r_gps": 20.0, "r_speedo": 1.0,
        },
        {
            "name": "4. Cold Start (Docked)",
            "x0": [0, 0, 0, 0],
            "p0_pos": 0.01, "p0_vel": 0.01,
            "u": [0.0, 0.0],
            "z": [0.01, 0.0, 0.0, 0.0],
            "q_pos": 0.001, "q_vel": 0.01,
            "r_gps": 0.04, "r_speedo": 0.01,
        },
    ]
    
    for i, scenario in enumerate(scenarios, start = 1):
        filter = KalmanFilter2D(gps_measurement_var=scenario["r_gps"], speedometere_measurement_var=scenario["r_speedo"], position_process_noise=scenario["q_pos"], velocity_process_noise=scenario["q_vel"])
        X0 = np.array(scenario["x0"], dtype=float).reshape(4, 1);
        P0 = np.diag([scenario["p0_pos"], scenario["p0_pos"], scenario["p0_vel"], scenario["p0_vel"]]).astype(float)
        U = np.array(scenario["u"], dtype=float).reshape(2, 1)
        Z = np.array(scenario["z"], dtype=float).reshape(4,1)

        X_pred, P_pred = filter.predict(X0, P0, U)
        X_upd, P_upd = filter.correct(X_pred, P_pred, Z)

        print(f"Scenario {i} {scenario['name']}:")
        print(f"  X_pred = {X_pred.ravel()}")
        print(f"  X_upd  = {X_upd.ravel()}")

        Filter_visualizer.plot_1d_gaussians(
            X_pred, P_pred, Z, filter.R, X_upd, P_upd,
            state_idx=0, label="Position X"
        )
        plt.savefig(f"Images/scenario_{i}_position_x.png")
        plt.close()

        Filter_visualizer.plot_1d_gaussians(
            X_pred, P_pred, Z, filter.R, X_upd, P_upd,
            state_idx=2, label="Velocity X", xlabel="m/s"
        )
        plt.savefig(f"Images/scenario_{i}_velocity_x.png")
        plt.close()
        print(f"  saved Images/scenario_{i}_position_x.png + velocity_x.png")


    pass
