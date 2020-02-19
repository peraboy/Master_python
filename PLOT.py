import matplotlib
import numpy as np
matplotlib.use("TkAgg")
def PLOT(total):
    if state_space == 'full':
        C = 180/np.pi
        # Euler angles:_____________________________________________________________________________________________________
        fig, axs = plt.subplots(3, 1)
        axs[0].plot( total['sim0']['t'], total['sim0']['roll']*C, total['sim1']['t'], total['sim1']['roll']*C, total['sim0']['t'] ,total['sim0']['ref_roll']*C, '--')
        axs[0].legend(['Roll', 'Roll_wear', 'Ref'])
        axs[0].set_title('Roll, Pitch and Yaw')
        # axs[0].set_xlim()
        axs[0].set_ylabel('Roll [Deg]')
        axs[0].grid(True)

        axs[1].plot(total['sim0']['t'], total['sim0']['pitch']*C, total['sim1']['t'], total['sim1']['pitch']*C , total['sim0']['t'] ,total['sim0']['ref_pitch']*C, '--')
        axs[1].legend(['Pitch', 'Pitch_wear', 'Ref'])
        #axs[0].set_title('Roll, Pitch and Yaw')
        # axs[0].set_xlim()
        axs[1].set_ylabel('Pitch [Deg]')
        axs[1].grid(True)

        axs[2].plot(total['sim0']['t'], total['sim0']['yaw']*C, total['sim1']['t'], total['sim1']['yaw']*C , total['sim0']['t'] ,total['sim0']['ref_yaw']*C, '--')
        axs[2].legend(['Yaw', 'Yaw_wear', 'Ref'])
        #axs[0].set_title('Roll, Pitch and Yaw')
        # axs[0].set_xlim()
        axs[2].set_ylabel('Yaw [Deg]')
        axs[2].grid(True)
        fig.tight_layout()
        plt.interactive(False)
        plt.show()

        # Inputs________________________________________________________________________________________________________
        fig, axs = plt.subplots(3, 1)
        axs[0].plot( total['sim0']['t'], total['sim0']['U'][0,:]*C, total['sim1']['t'], total['sim1']['U'][0,:]*C)
        axs[0].legend([r'$\delta_a$', '$\delta_a^w$'])
        axs[0].set_title('Inputs')
        # axs[0].set_xlim()
        axs[0].set_ylabel('Ailerons [Deg]')
        axs[0].grid(True)

        axs[1].plot( total['sim0']['t'], total['sim0']['U'][1,:]*C, total['sim1']['t'], total['sim1']['U'][1,:]*C)
        axs[1].legend([r'$\delta_e$', '$\delta_e^w$'])
        # axs[0].set_xlim()
        axs[1].set_ylabel('Elevator [Deg]')
        axs[1].grid(True)

        axs[2].plot( total['sim0']['t'], total['sim0']['U'][3,:]*C, total['sim1']['t'], total['sim1']['U'][3,:]*C)
        axs[2].legend([r'$\delta_t$', '$\delta_t^w$'])
        # axs[0].set_xlim()
        axs[2].set_ylabel('Throttle [Deg]')
        axs[2].grid(True)
        fig.tight_layout()
        plt.interactive(False)
        plt.show()

        # SSA, AOA and airspeed_________________________________________________________________________________________
        fig, axs = plt.subplots(3, 1)
        axs[0].plot( total['sim0']['t'], total['sim0']['v_r'], total['sim1']['t'], total['sim1']['v_r'], total['sim0']['t'], total['sim0']['v_ref'][0:len(total['sim0']['t'])], '--')
        axs[0].legend(['Vr', r'$Vr^w$'])
        #axs[0].set_title('')
        # axs[0].set_xlim()
        axs[0].set_ylabel('Airspeed [Deg]')
        axs[0].grid(True)

        axs[1].plot( total['sim0']['t'], total['sim0']['aoa']*C, total['sim1']['t'], total['sim1']['aoa']*C)
        axs[1].legend([r'$\alpha$', r'$ \alpha^w $'])
        # axs[0].set_xlim()
        axs[1].set_ylabel('AoA [Deg]')
        axs[1].grid(True)

        axs[2].plot( total['sim0']['t'], total['sim0']['ssa']*C, total['sim1']['t'], total['sim1']['ssa']*C)
        axs[2].legend([r'$\beta $', r'$\beta^w$'])
        # axs[0].set_xlim()
        axs[2].set_ylabel('SSA [Deg]')
        axs[2].grid(True)
        fig.tight_layout()
        plt.interactive(False)
        plt.show()








        # NED position______________________________________________________________________________________________________
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(t, X[0, :])
        # axs[0].set_title('Inputs')
        # axs[0].set_xlim()
        axs[0].set_ylabel('NED x-axis [m]')
        axs[0].grid(True)

        axs[1].plot(t, X[1, :])
        # axs[1].set_xlim()
        axs[1].set_ylabel('NED y-axis [m]')
        axs[1].grid(True)

        axs[2].plot(t, X[2, :])
        # axs[2].set_xlim()
        axs[2].set_ylabel('NED z-axis [m]')
        axs[2].set_xlabel('Time [s]')
        axs[2].grid(True)
        fig.tight_layout()
        plt.interactive(False)
        plt.show()
    else:
        return 'Nothing to plot'