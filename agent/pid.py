class PIDHelper:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.accumulated_error = 0
        self.prev_error = 0

    def increment_intregral_error(self, error, pi_limit=3):
        self.accumulated_error = self.accumulated_error + error
        if self.accumulated_error > pi_limit:
            self.accumulated_error = pi_limit
        elif self.accumulated_error < pi_limit:
            self.accumulated_error = -pi_limit

    def compute_output(self, error):
        self.increment_intregral_error(error)
        output = self.Kp * error + self.Ki * self.accumulated_error + self.Kd * (error - self.prev_error)
        self.prev_error = error
        return error


class PIDTuned:
    def __init__(self):
        super(PIDTuned, self).__init__()
        self.Fe_PID = PIDHelper(0.001, 0, 0.001)
        self.psi_PID = PIDHelper(0.085, 0.001, 10.55)
        self.Fs_theta_PID = PIDHelper(5, 0, 6)

    def pid_algorithm(self, s):
        dx, dy, vel_x, vel_y, theta, omega, legContact_left, legContact_right = s

        if dx > 0.3:
            dx_tmp = 0.3
        elif dx < -0.3:
            dx_tmp = -0.3
        else:
            dx_tmp = dx
        Fe = self.Fe_PID.compute_output(abs(dx_tmp)*0.4 - dy*0.2)
        Fs = self.Fs_theta_PID.compute_output(theta*5)
        psi = self.psi_PID.compute_output(theta + dx/5)

        if legContact_left and legContact_right:  # legs have contact
            Fe = 0
            Fs = 0

        return Fe, Fs, psi
