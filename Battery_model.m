% Battery model parameters
R0 = 0.1;           % Battery internal resistance (Ohms)
C = 1000;           % Battery capacity (Ah)
alpha = 0.9;        % SoH decay factor

% Measurement noise covariance
Q = diag([0.01, 0.01]);  % Process noise covariance
R = diag([0.05, 0.05]);  % Measurement noise covariance

% Initial estimates and covariances
x_est = [0.5; 0.9];          % Initial estimates of SoC and SoH
P = diag([0.1, 0.1]);        % Initial covariance matrix

% Simulated current and voltage measurements
time = 0:0.1:10;
current = sin(time);        % Simulated current profile
voltage = R0 * current + C * alpha * (1 - exp(-time/alpha)) + randn(size(time))*sqrt(R(2,2));

% Kalman filter estimation
for i = 1:length(time)
    % Prediction step
    x_prd = x_est;                        % State prediction
    P_prd = P + Q;                        % Covariance prediction

    % Update step
    H = [x_prd(1), 0; 0, x_prd(2)];       % Measurement matrix
    z = [voltage(i); current(i)];         % Measurement vector

    S = H * P_prd * H' + R;                % Innovation covariance
    K = P_prd * H' / S;                    % Kalman gain

    x_est = x_prd + K * (z - H * x_prd);   % State estimation
    P = (eye(2) - K * H) * P_prd;          % Covariance estimation

    % Store the estimated SoC and SoH
    estimated_soc(i) = x_est(1);
    estimated_soh(i) = x_est(2);
end

% Plot the results
figure;
subplot(2,1,1);
plot(time, voltage, 'b', 'LineWidth', 1.5);
hold on;
grid on;
plot(time, R0 * current + C * estimated_soh, 'r--', 'LineWidth', 1.5);
ylabel('Voltage (V)');
legend('Measured Voltage', 'Estimated Voltage');
title('Battery Voltage');

subplot(2,1,2);
plot(time, estimated_soc, 'b', 'LineWidth', 1.5);
hold on;
grid on;
plot(time, ones(size(time))*0.5, 'r--', 'LineWidth', 1.5);
ylabel('State of Charge (SoC)');
xlabel('Time (s)');
legend('Estimated SoC', 'True SoC');
title('Battery State of Charge');