# 58%
class PIDTuned1:
    def __init__(self):
        super(PIDTuned1, self).__init__()
        self.Fe_PID = PIDHelper(0.001, 0, 0.001)
        self.psi_PID = PIDHelper(0.08, 0.001, 10)
        self.Fs_theta_PID = PIDHelper(5, 0, 6)

    def pid_algorithm(self, s):
        dx, dy, vel_x, vel_y, theta, omega, leg_contact_left, leg_contact_right = s

        Fe = self.Fe_PID.compute_output(min(abs(dx), 0.3)*0.4 - dy*0.2)
        Fs = self.Fs_theta_PID.compute_output(theta*5)
        psi = self.psi_PID.compute_output(theta + dx/5)

        if leg_contact_left and leg_contact_right:  # legs have contact
            Fe = 0
            Fs = 0

        return Fe, Fs, psi


# 24-33.5%
class PIDTuned2:
    def __init__(self):
        super(PIDTuned2, self).__init__()
        self.Fe_PID = PIDHelper(5, 0, 5)
        self.psi_PID = PIDHelper(1, 0.001, 10)
        self.Fs_theta_PID = PIDHelper(5, 0, 6)

    def pid_algorithm(self, s):
        dx, dy, vel_x, vel_y, theta, omega, leg_contact_left, leg_contact_right = s

        Fe = self.Fe_PID.compute_output(min(abs(dx), 0.3)*0.4 - dy*0.2)
        Fs = self.Fs_theta_PID.compute_output(theta*5)
        psi = self.psi_PID.compute_output(theta + dx/5)

        if leg_contact_left and leg_contact_right:  # legs have contact
            Fe = 0
            Fs = 0

        return Fe, Fs, psi


class PIDHelper:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.accumulated_error = 0
        self.prev_error = 0

    def increment_integral_error(self, error, limit=3):
        self.accumulated_error = self.accumulated_error + error
        if self.accumulated_error > limit:
            self.accumulated_error = limit
        elif self.accumulated_error < limit:
            self.accumulated_error = -limit

    def compute_output(self, error):
        self.increment_integral_error(error)
        dt_error = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.accumulated_error + self.Kd * dt_error
