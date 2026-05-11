''' This python file contains the various normalizers for input data of the format:
tensor( events, {energy, vx, vy}, x, y, tau})
'''

import numpy as np
class NormFn:
    def __init__(self):
        return

    @staticmethod
    def etau_array(data: np.ndarray,  # in form {n_events, {E,vx,vy}, x, y, t, {E,vx,vy}}
                    tau0: float,
                    tau_step: float,
                    eps: float = 1e-6
                    ) -> np.ndarray:
        ''' input: data of shape {n_events, {e,vx,vy}, x, y, t}, time 0, time steps size, eps
            output: data of shape {n_events, {e*tau, vx, vy}, x, y, t}
        '''
        nT = data.shape[-1] 
        T_tau = np.arange(tau0, tau0 + nT*tau_step, tau_step).reshape(1,1,1,-1)
        # v2 = data[:,1,:,:,:]**2 + data[:,2,:,:,:]**2
        # v2[v2 >= 1] = 1-eps
        # gamma = 1./np.sqrt(1-v2) # gamma = 1/sqrt(1-vx^2-vy^2)
        T_e = data[:,0,:,:,:] * T_tau # E*\tau*\gamma
        # T_ux = T_e * data[:,1,:,:,:] # E*\tau*\gamma*vx
        # T_uy = T_e * data[:,2,:,:,:] # E*\tau*\gamma*vy
        return np.stack((T_e, data[:,1,:,:,:], data[:,2,:,:,:]), axis=1)

    @staticmethod
    def inv_etau_array(data: np.ndarray,# in form {n_events, {E*gamma*tau, E*gamma*tau*vx, E*gamma*tau*vy}, x, y, t}
                        tau0: float, 
                        tau_step: float, 
                        eps: float = 1e-6
                        ) -> np.ndarray:
        ''' input: {n_events, {e*tau, vx, vy}, x, y, t}, time 0, time step size, eps
            output: {n_events, {e, vx, vy}, x, y, t}
        '''
        nT = data.shape[-1] 
        T_tau = np.arange(tau0, tau0 + nT*tau_step, tau_step).reshape(1,1,1,-1)
        etaugamma = data[:,0,:,:,:]
        etaugamma[etaugamma <= 0] = eps
        # T_vx = data[:,1,:,:,:] / etaugamma # vx = E*gamma*tau*vx / E*gamma*tau
        # T_vy = data[:,2,:,:,:] / etaugamma # vy = E*gamma*tau*vy / E*gamma*tau
        # v2 = T_vx**2 + T_vy**2
        # v2[v2 >= 1] = 1-eps
        # gamma = 1./np.sqrt(1 - v2)
        T_e = data[:,0,:,:,:] / T_tau # E = E*gamma*tau / (tau*gamma)
        return np.stack((T_e, data[:,1,:,:,:], data[:,2,:,:,:]), axis=1)

    @staticmethod
    def etaugamma_array(data: np.ndarray,  # in form {n_events, {E,vx,vy}, x, y, t, {E,vx,vy}}
                    tau0: float,
                    tau_step: float,
                    eps: float = 1e-6
                    ) -> np.ndarray:
        ''' input: data of shape {n_events, {e,vx,vy}, x, y, t}, time 0, time steps size, eps
            output: data of shape {n_events, {e*tau*gamma, vx, vy, x, y, t}
        '''
        nT = data.shape[-1] 
        T_tau = np.arange(tau0, tau0 + nT*tau_step, tau_step).reshape(1,1,1,-1)
        v2 = data[:,1,:,:,:]**2 + data[:,2,:,:,:]**2
        v2[v2 >= 1] = 1-eps
        gamma = 1./np.sqrt(1-v2) # gamma = 1/sqrt(1-vx^2-vy^2)
        T_e = data[:,0,:,:,:] * T_tau * gamma # E*\tau*\gamma
        # T_ux = T_e * data[:,1,:,:,:] # E*\tau*\gamma*vx
        # T_uy = T_e * data[:,2,:,:,:] # E*\tau*\gamma*vy
        # return np.stack((T_e, T_ux, T_uy), axis=1)
        return np.stack((T_e, data[:,1,:,:,:], data[:,2,:,:,:]), axis=1)

    @staticmethod
    def inv_etaugamma_array(data: np.ndarray,# in form {n_events, {E*gamma*tau, E*gamma*tau*vx, E*gamma*tau*vy}, x, y, t}
                        tau0: float, 
                        tau_step: float, 
                        eps: float = 1e-6
                        ) -> np.ndarray:
        ''' input: {n_events, {e*tau*gamma, e*gamma*tau*vx, e*gamma*tau*vy}, x, y, t}, time 0, time step size, eps
            output: {n_events, {e, vx, vy}, x, y, t}
        '''
        nT = data.shape[-1] 
        T_tau = np.arange(tau0, tau0 + nT*tau_step, tau_step).reshape(1,1,1,-1)
        etaugamma = data[:,0,:,:,:]
        etaugamma[etaugamma <= 0] = eps
        v2 = data[:,1,:,:,:]**2 + data[:,2,:,:,:]**2
        v2[v2 >= 1] = 1-eps
        gamma = 1./np.sqrt(1 - v2)
        T_e = data[:,0,:,:,:] / (T_tau * gamma) # E = E*gamma*tau / (tau*gamma)
        # return np.stack((T_e, T_vx, T_vy), axis=1)
        return np.stack((T_e, data[:,1,:,:,:], data[:,2,:,:,:]), axis=1)

    @staticmethod
    def energy_mom_array(data: np.ndarray,  # in form {n_events, {E,vx,vy}, x, y, t, {E,vx,vy}}
                    tau0: float,
                    tau_step: float,
                    eps: float = 1e-6
                    ) -> np.ndarray:
        ''' input: data of shape {n_events, {e,vx,vy}, x, y, t}, time 0, time steps size, eps
            output: data of shape {n_events, {e*tau*gamma, e*gamma*tau*vx, e*gamma*tau*vy}, x, y, t}
        '''
        nT = data.shape[-1] 
        T_tau = np.arange(tau0, tau0 + nT*tau_step, tau_step).reshape(1,1,1,-1)
        v2 = data[:,1,:,:,:]**2 + data[:,2,:,:,:]**2
        v2[v2 >= 1] = 1-eps
        gamma = 1./np.sqrt(1-v2) # gamma = 1/sqrt(1-vx^2-vy^2)
        T_e = data[:,0,:,:,:] * T_tau * gamma # E*\tau*\gamma
        T_ux = T_e * data[:,1,:,:,:] # E*\tau*\gamma*vx
        T_uy = T_e * data[:,2,:,:,:] # E*\tau*\gamma*vy
        return np.stack((T_e, T_ux, T_uy), axis=1)

    @staticmethod
    def inv_energy_mom_array(data: np.ndarray,# in form {n_events, {E*gamma*tau, E*gamma*tau*vx, E*gamma*tau*vy}, x, y, t}
                        tau0: float, 
                        tau_step: float, 
                        eps: float = 1e-6
                        ) -> np.ndarray:
        ''' input: {n_events, {e*tau*gamma, e*gamma*tau*vx, e*gamma*tau*vy}, x, y, t}, time 0, time step size, eps
            output: {n_events, {e, vx, vy}, x, y, t}
        '''
        nT = data.shape[-1] 
        T_tau = np.arange(tau0, tau0 + nT*tau_step, tau_step).reshape(1,1,1,-1)
        etaugamma = data[:,0,:,:,:]
        etaugamma[etaugamma <= 0] = eps
        T_vx = data[:,1,:,:,:] / etaugamma # vx = E*gamma*tau*vx / E*gamma*tau
        T_vy = data[:,2,:,:,:] / etaugamma # vy = E*gamma*tau*vy / E*gamma*tau
        v2 = T_vx**2 + T_vy**2
        v2[v2 >= 1] = 1-eps
        gamma = 1./np.sqrt(1 - v2)
        T_e = data[:,0,:,:,:] / (T_tau * gamma) # E = E*gamma*tau / (tau*gamma)
        return np.stack((T_e, T_vx, T_vy), axis=1)

    @staticmethod
    def e0p75(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        ''' input: data of shape {n_events, {e,vx,vy}, x, y, t}, eps
            output: data of shape {n_events, {e**0.75, vx, vy}, x, y, t}
        '''
        dat = data.copy()
        dat[:,0,:,:,:][dat[:,0,:,:,:] > eps] = dat[:,0,:,:,:][dat[:,0,:,:,:] > eps] ** 0.75
        return dat

    @staticmethod
    def inv_e0p75(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        ''' input: data of shape {n_events, {e,vx,vy}, x, y, t}, eps
            output: data of shape {n_events, {e**0.75, vx, vy}, x, y, t}
        '''
        dat = data.copy()
        dat[:,0,:,:,:][dat[:,0,:,:,:] > eps] = dat[:,0,:,:,:][dat[:,0,:,:,:] > eps] ** (1./0.75)
        return dat

    # do the unit gaussian
    @staticmethod
    def unit_gaussian(data: np.ndarray, min_energy:float=0.1, 
                    train_ratio:float=0.75, eps:float=1e-6, mean=None, std=None) -> (np.ndarray, np.ndarray, np.ndarray):

        if mean is None and std is None:
            ntrain = int(data.shape[0] * train_ratio)
            ntau = data.shape[-1]

            # get the stats
            E = data[:ntrain].copy()
            E[:,0,:,:,:][E[:,0,:,:,:] < min_energy] = np.nan

            mean = np.array([np.nanmean(E[:,0,:,:,T]) for T in range(ntau)], dtype=float)
            std = np.array([np.nanstd(E[:,0,:,:,T]) for T in range(ntau)], dtype=float)

        dat = data.copy()
        print(dat.shape)
        # the below process pushes dat[:0,:,:,:] to have shape only dat[ev,x,y,t] and the broadcast
        # expands time along the three axes
        dat[:,0,:,:,:] = (dat[:,0,:,:,:] - mean.reshape(1,1,1,-1)) / (std.reshape(1,1,1,-1) + eps)
        return dat, mean, std

    @staticmethod
    def inv_unit_gaussian(data: np.ndarray, mean: np.ndarray, std: np.ndarray, 
                        eps: float = 1e-6) -> np.ndarray:
        '''
        take mean and std and inverse the unit gaussian transformation
        '''
        dat = data.copy()
        dat[:,0,:,:,:] = dat[:,0,:,:,:] * (std.reshape(1,1,1,-1) + eps) + mean.reshape(1,1,1,-1)
        return dat
        
    @staticmethod
    def mean_max_per_tau(data, mean: np.array=None):
        ''' Get the mean and std per time step for the maximum e function. '''
        if mean is None:
            max_per_ev_per_tau = np.nanmax(data[:,0,:,:,:], axis=(1,2))
        # print(max_per_ev_per_tau.shape)
            mean_max_per_tau = np.nanmean(max_per_ev_per_tau, axis=0)
        else:
            mean_max_per_tau = mean
        # print(mean_max_per_tau.shape)
        # print(mean_max_per_tau)
        dat = data.copy()
        dat[:,0,:,:,:] = dat[:,0,:,:,:] / mean_max_per_tau.reshape(1, 1, 1, -1)
        return dat, mean_max_per_tau
    def inv_mean_max_per_tau(data, mean):
        ''' Inverse transformation to get back the original data from scaled data. '''
        dat = data.copy()
        dat[:,0,:,:,:] = dat[:,0,:,:,:] * mean.reshape(1, 1, 1, -1)
        return dat

    @staticmethod
    def energy_mom_tensor(data: np.ndarray,
                    tau0: float,
                    tau_step: float,
                    eps: float = 1e-6
                    ) -> np.ndarray:
        ''' input: data of shape {n_events, {e, vx, vy}, x, y, t}
            output: data of shape {n_events, {e*tau*gamma, e*tau*gamma*vx, e*tau*gamma*vy}, x, y, t}
        '''
        nT = data.shape[-1] 
        T_tau = np.arange(tau0, tau0 + nT*tau_step, tau_step).reshape(1,1,1,-1)
        v2 = data[:,1,:,:,:]**2 + data[:,2,:,:,:]**2
        v2[v2 >= 1] = 1-eps
        gamma = 1./np.sqrt(1-v2) # gamma = 1/sqrt(1-vx^2-vy^2)
        T_e = data[:,0,:,:,:] * T_tau * gamma
        T_ux = T_e * data[:,1,:,:,:] # E*\tau*\gamma*vx
        T_uy = T_e * data[:,2,:,:,:] # E*\tau*\gamma*vy
        return np.stack((T_e, T_ux, T_uy), axis=1)

    @staticmethod
    def etau_3d(data: np.ndarray,
                tau_min: float,
                dtau: float,
                ) -> np.ndarray:
        ''' input: data of shape {nevents, {ed, vx, vy, ...}, x, y, eta, tau}
            output: data of shape {nevents, {ed*tau, vx, vy, ...}, x, y, eta, tau}
            where tau = tau_min + taubin * dtau
        '''
        nT = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, 1, -1)
        out = data.copy()
        out[:, 0] = data[:, 0] * tau
        return out

    @staticmethod
    def etau75_3d(data: np.ndarray,
                tau_min: float,
                dtau: float,
                ) -> np.ndarray:
        ''' input: data of shape {nevents, {ed, vx, vy, ...}, x, y, eta, tau}
            output: data of shape {nevents, {ed^0.75*tau, vx, vy, ...}, x, y, eta, tau}
            where tau = tau_min + taubin * dtau
        '''
        nT = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, 1, -1)
        out = data.copy()
        out[:, 0] = data[:, 0]**0.75 * tau
        return out

    @staticmethod
    def inv_etau75_3d(data: np.ndarray,
                    tau_min: float,
                    dtau: float,
                    eps: float = 1e-6,
                    ) -> np.ndarray:
        ''' input: data of shape {nevents, {ed^0.75*tau, vx, vy, ...}, x, y, eta, tau}
            output: data of shape {nevents, {ed, vx, vy, ...}, x, y, eta, tau}
        '''
        nT = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, 1, -1)
        out = data.copy()
        out[:, 0] = (data[:, 0] / (tau + eps))**(1/0.75)
        return out

    @staticmethod
    def inv_etau_3d(data: np.ndarray,
                    tau_min: float,
                    dtau: float,
                    eps: float = 1e-6,
                    ) -> np.ndarray:
        ''' input: data of shape {nevents, {ed*tau, vx, vy, ...}, x, y, eta, tau}
            output: data of shape {nevents, {ed, vx, vy, ...}, x, y, eta, tau}
        '''
        nT = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, 1, -1)
        out = data.copy()
        out[:, 0] = data[:, 0] / (tau + eps)
        return out

    @staticmethod
    def inv_energy_mom_tensor(data: np.ndarray,
                    tau0: float,
                    tau_step: float,
                    eps: float = 1e-6
                    ) -> np.ndarray:
        ''' input: data of shape {n_events, {e*tau*gamma, e*tau*gamma*vx, e*tau*gamma*vy}
            output: data of shape {n_events, {e, vx, vy}, x, y, t}
        '''
        nT = data.shape[-1] 
        T_tau = np.arange(tau0, tau0 + nT*tau_step, tau_step).reshape(1,1,1,-1)
        etaugamma = data[:,0,:,:,:]
        etaugamma[etaugamma <= 0] = eps
        T_vx = data[:,1,:,:,:] / etaugamma # vx = E*gamma*tau*vx / E*gamma*tau
        T_vy = data[:,2,:,:,:] / etaugamma # vy = E*gamma*tau*vy / E*gamma*tau
        v2 = T_vx**2 + T_vy**2
        v2[v2 >= 1] = 1-eps
        gamma = 1./np.sqrt(1 - v2)
        T_e = data[:,0,:,:,:] / T_tau / gamma # E = E*gamma*tau / (tau*gamma)
        return np.stack((T_e, T_vx, T_vy), axis=1)