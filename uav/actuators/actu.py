def actuator(u_ref, U, k, fs, T):
    ref1 = U[0] + (1 / (fs * T)) * (u_ref[0] - U[0])
    ref2 = U[1] + (1 / (fs * T)) * (u_ref[1] - U[1])
    ref3 = U[3] + (1 / (fs * T)) * (u_ref[2] - U[3])
    return ref1, ref2, ref3

