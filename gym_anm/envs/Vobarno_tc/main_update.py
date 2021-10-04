import warnings
import random
import numpy as np
import pandas as pd
from gym import spaces
from scipy.sparse.linalg import MatrixRankWarning
import matplotlib.pyplot as plt
import seaborn as sns

from gym_anm.envs.anm_env_tc import ANMEnv_tc
from gym_anm.errors import EnvNextVarsError
from gym_anm.simulator.components import Load
from gym_anm.envs.Vobarno_tc.calendario import calendar


class NetworkGenerator():

    # aggiungo come parametro la data a cui saranno associate le curve di carico e generazione
    # aggiungo come parametro l'errore da aggiungere a ogni episodio ai carichi e ai generatori
    # L'errore in pu è costante lungo tutto il giorno e viene moltiplicato per la potenza nominale del carioc.
    # Un unico errore per i generatori idroelettrici, uno per gli impianti PV, singoli per ciascun carico
    def __init__(self, day_date, error_gen, error_load):
        self.error_load = error_load
        self.error_gen = error_gen
        bus = pd.read_csv('bus.csv', sep=';')
        bus = bus.to_numpy()

        device_df = pd.read_csv('device1.csv', sep=';')
        # Prendo le colonne necessarie per network, dopo aggiungerò le sette mancanti
        device = device_df[['dev_ID', 'bus', 'Type', 'Q/P', 'Pmax', 'Pmin', 'Qmax', 'Qmin']]

        self.dev_len = len(device)
        self.gen_len = 43
        self.load_len = self.dev_len - self.gen_len - 1
        df_code = calendar()
        self.code_cal = df_code.loc[(df_code['Date'] == day_date), 'code']
        self.code_cal = self.code_cal.values

        self.gen_dic = []
        self.car_dic = []
        self.car_dic_q = []
        self.tipo_prof = 7  # uno di questi è il flat_group però
        self.tipo_gen = 2
        self.k = 8  # numero di profili per carico
        self.day_interv = 96  # quarter-an-hour-data
        self.gen_pv = self.gen_len - 2  # due nodi sono collegati a impianti idroelettrici
        self.flat = np.ones(self.day_interv)

        gen = device_df[['Curve_car_gen', 'Pmax', 'DEVICE']]
        gen = gen[gen.DEVICE.str.contains('Generatore')]
        self.gen_tipo = gen['Curve_car_gen'].tolist()
        self.gen_p = gen['Pmax'].tolist()
        self.d_gen = {'GenSolare': 0, 'GenIdro': 1}
        self.lst_gen = [self.d_gen[k] for k in self.gen_tipo]


        car = device_df[['Curve_car_gen', 'p_media', 'DEVICE']]
        carichi = car[car.DEVICE.str.contains('Carico')]
        self.carico_tipo = carichi['Curve_car_gen'].tolist()
        self.carico_p_media = carichi['p_media'].tolist()

        self.d_car = {'IlluminazionePublica': 0, 'PrelievoDomestico': 1, 'PrelievoNonDomesticoSopra55kW': 2,
             'PrelievoNonDomesticoSotto6kW': 3,
             'PrelievoNonDomesticoTra16kWe55kW': 4, 'PrelievoNonDomesticoTra6kWe16kW': 5, 'FlatLoadGroup': 6}
        self.lst_car = [self.d_car[k] for k in self.carico_tipo]

        device = device.to_numpy()

        # AGGIUNGERE 7 COLONNE LEGATE AI GENERATORI CHE SERVONO DA INPUT ALLA LIBRERIA
        param_not_useful = np.zeros((self.dev_len, 7))
        device = np.hstack((device, param_not_useful))
        self.m = self.dev_len - 1

        branch = pd.read_csv('branch.csv', sep=';')
        branch = branch.to_numpy()

        self.network = {
            'baseMVA': 40,
            'bus': bus,
            'device': device,
            'branch': branch
        }

    def get_network(self):
        return self.network

    # Definisco il periodo dell'anno in cui calcolare i carichi

    def get_p_time_series(self):
        """Return the fixed 24-hour time-series for the load injections."""

        '''GENERATORI'''
        curve_gen_df = pd.read_csv('gen_car.csv', sep=';')
        curve_gen = np.array(curve_gen_df)
        curve_dic_gen = np.zeros((self.tipo_gen, self.day_interv))
        for i in range(self.tipo_gen):
            curve_dic_gen[i, :] = curve_gen[i * self.k + self.code_cal, :]  # Dato che ci sono 8 profili a seconda della stagione bisogna scalare di k

        for i in range(self.gen_len):
            tipo_gen = self.lst_gen[i]
            p = curve_dic_gen[tipo_gen, :] * self.gen_p[i]
            p += self.error_gen[tipo_gen] * p
            self.gen_dic.append(p)

        P_gen_dic = np.array(self.gen_dic)

        '''CARICHI'''
        curve_carico_df = pd.read_csv('p_car.csv', sep=';')
        curve_carico = np.array(curve_carico_df)
        curve_dic = np.zeros((self.tipo_prof - 1, self.day_interv))
        for i in range(self.tipo_prof - 1):
            curve_dic[i, :] = curve_carico[i * self.k + self.code_cal, :]
        curve_dic = np.vstack([curve_dic, self.flat])

        for i in range(self.load_len):
            tipo = self.lst_car[i]
            p = curve_dic[tipo, :] * self.carico_p_media[i]
            p -= self.error_load[i] * p
            self.car_dic.append(p)

        P_car_dic = np.array(self.car_dic)

        P_dev = np.vstack((P_gen_dic, P_car_dic))
        assert P_dev.shape == (self.m, self.day_interv)

        return P_dev

    def get_q_time_series(self):
        """Return the fixed 24-hour time-series for the load injections."""

        '''GENERATORI'''
        # I generatori sono tipicamente usati con fattore di potenza unitario
        P_gen_dic_q = np.zeros((self.gen_len, self.day_interv))

        '''CARICHI'''
        curve_q_df = pd.read_csv('q_car.csv', sep=';')
        curve_q = np.array(curve_q_df)
        curve_dic_q = np.zeros((self.tipo_prof - 1, 96))
        for i in range(self.tipo_prof - 1):
            curve_dic_q[i, :] = curve_q[i * self.k + self.code_cal, :]

        curve_dic_q = np.vstack([curve_dic_q, self.flat / 3])
        for i in range(self.load_len):
            tipo = self.lst_car[i]
            q = curve_dic_q[tipo, :] * self.carico_p_media[i]
            q -= self.error_load[i] * q
            self.car_dic_q.append(q)

        P_car_dic_q = np.array(self.car_dic_q)
        Q_loads = np.vstack((P_gen_dic_q, P_car_dic_q))
        assert Q_loads.shape == (self.m, self.day_interv)

        return Q_loads


class TCEnvironment(ANMEnv_tc):
    """An example of a simple 2-bus custom gym-anm environment."""

    def __init__(self, my_network, p_dev, q_loads, init_tc):
        # observation = [('bus_v_magn', [1, 92, 148, 267, 204], 'pu')]
        observation = [('bus_v_magn', 'all', 'pu')]
        # observation = [('bus_p', [0], 'MW'), ('bus_q', [0], 'MVAr')]
        K = 1                             # 1 auxiliary variable
        delta_t = 0.25                    # 15min intervals
        gamma = 0.9                       # discount factor
        lamb = 100                       # penalty weighting hyperparameter
        PUN = 60
        penalty_tc = 10
        bus_from = 0  # lato primario del TC
        bus_to = 1
        init_v_SB = np.array([1, 1]) # input in forma di array
        aux_bounds = np.array([[0, 10]])  # bounds on auxiliary variable
        costs_clipping = (100, 10000)         # reward clipping parameters
        seed = 1                          # random seed

        super().__init__(my_network, observation, K, delta_t, gamma, lamb, PUN, penalty_tc,
                         bus_from, bus_to, init_v_SB, aux_bounds, costs_clipping, seed)

        self.tap_interval = 0.015
        self.upper_bound = 1.164  # Valore appena inferiore per problemi con l'approssimazione
        self.lower_bound = 0.836  # Valore appena superiore per problemi con l'approssimazione
        self.bus_slack = 0
        self.branch_tc = 0

        self.P_dev = p_dev
        self.Q_loads = q_loads
        self.tc = init_tc
        self.simulator.update_tc(init_tc)
        self.action_space = spaces.Discrete(3)
        self.n_dev = 2792  # Aggiungi numero di dispositivi di network.device

    def init_state(self):
        """Return a state vector with random values in [0, 1]."""

        state = np.zeros(2 * self.n_dev + self.K)
        t_0 = 0
        state[-1] = t_0
        # Load (P, Q) injections.
        numbers = list(range(1, self.n_dev))
        for dev_id, p_dev in zip(numbers, self.P_dev):
            state[dev_id] = p_dev[t_0]
        for dev_id, q_load in zip(numbers, self.Q_loads):
            state[self.n_dev + dev_id] = q_load[t_0]
        return state

    def next_vars(self, s_t):
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))

        vars = []
        for p_dev in self.P_dev:
            vars.append(p_dev[aux])

        for q_load in self.Q_loads:
            vars.append(q_load[aux])

        vars.append(aux)
        return np.array(vars)

    def step(self, action):
        """
        Take a control action and transition from state :math:`s_t` to state :math:`s_{t+1}`.

        Parameters
        ----------
        action : numpy.ndarray
            The action vector :math:`a_t` taken by the agent.

        Dovrei fare a = +1 tap_changer pos. TC =+ 0.0125

        Returns
        -------
        obs : numpy.ndarray
            The observation vector :math:`o_{t+1}`.
        reward : float
            The reward associated with the transition :math:`r_t`.
        done : bool
            True if a terminal state has been reached; False otherwise.
        info : dict
            A dictionary with further information (used for debugging).
        """

        self.simulator.update_current_action(action)

        # 0. Remain in a terminal state and output reward=0 if the environment
        # has already reached a terminal state.
        if self.done:
            obs = self._terminal_state(self.observation_N)
            return obs, 0., self.done, {}

        # 1a. Sample the internal stochastic variables.
        vars = self.next_vars(self.state)
        expected_size = 2 * self.simulator.N_load + self.K

        if vars.size != expected_size:
            msg = 'Next vars vector has size %d but expected is %d' % \
                  (vars.size, expected_size)
            raise EnvNextVarsError(msg)

        P_dev = vars[:self.simulator.N_load]
        Q_load = vars[self.simulator.N_load: 2 * self.simulator.N_load]
        aux = vars[2 * self.simulator.N_load:]
        err_msg = 'Only {} auxiliary variables are generated, but K={} are ' \
                  'expected.'.format(len(aux), self.K)
        assert len(aux) == self.K, err_msg

        # 1b. Convert internal variables to dictionaries.
        load_idx, gen_idx = 0, 0
        P_dev_dict, Q_load_dict = {}, {}
        for dev_id, dev in self.simulator.devices.items():
            if isinstance(dev, Load):
                P_dev_dict[dev_id] = P_dev[load_idx]
                Q_load_dict[dev_id] = Q_load[load_idx]
                load_idx += 1

        if self.upper_bound > self.tc > self.lower_bound:
            if action == 0:
                self.tc -= self.tap_interval
            elif action == 1:
                self.tc = self.tc
            elif action == 2:
                self.tc += self.tap_interval
        elif self.tc > self.upper_bound:
            if action == 0:
                self.tc -= self.tap_interval
            elif action == 1:
                self.tc = self.tc
        elif self.tc < self.lower_bound:
            if action == 2:
                self.tc += self.tap_interval
            elif action == 1:
                self.tc = self.tc

        # 3a. Apply the action in the simulator.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', MatrixRankWarning)
            self.simulator.update_tc(self.tc)
            _, r, e_loss, penalty, pfe_converged = \
                self.simulator.transition(P_dev_dict, Q_load_dict)

            # A terminal state has been reached if no solution to the power
            # flow equations is found.
            self.done = not pfe_converged

        # 3b. Clip the reward.
        if not self.done:
            self.e_loss = np.sign(e_loss) * np.clip(np.abs(e_loss), 0,
                                                    self.costs_clipping[0])  # mette comunque un limite
            self.penalty = np.clip(penalty, 0, self.costs_clipping[1])
            r = - (self.e_loss + self.penalty)
        else:
            # Very large reward if a terminal state has been reached.
            r = - self.costs_clipping[1] / (1 - self.gamma)
            self.e_loss = self.costs_clipping[0]
            self.penalty = self.costs_clipping[1]

        # 4. Construct the state and observation vector.
        if not self.done:
            for k in range(self.K):
                self.state[k - self.K] = aux[k]
            self.state = self._construct_state()
            obs = self.observation(self.state)

            err_msg = "Observation %r (%s) invalid." % (obs, type(obs))
            assert self.observation_space.contains(obs), err_msg

        # Cast state and obs vectors to 0 (arbitrary) if a terminal state is
        # reached.
        else:
            self.state = self._terminal_state(self.state_N)
            obs = self._terminal_state(self.observation_N)

        # 5. Update the timestep.
        self.timestep += 1

        return obs, r, self.done, {}, self.tc

    def update_values(self, slack, tc):
        self.network["bus"][self.bus_slack, 3] = slack
        self.network["branch"][self.branch_tc, 6] = tc


# FUNZIONE PER DISCRETIZZARE LE OSSERVAZIONI, NEL NOSTRO CASO IN 8 INTERVALLI
def discrete_state(observation, num_ranges, step):

    range_vett = []
    for j in range(0, num_ranges - 1):
        half = num_ranges / 2
        k = half - j
        n = 1 - (k - 1) * step
        range_vett.append(n)
    v_range = np.digitize(observation, range_vett)
    val = 0
    for i in range(1, len(v_range) + 1):
        val += v_range[-i] * num_ranges ** (i - 1)

    return val


if __name__ == '__main__':
    dti = pd.date_range('2022-07-12', periods=3, freq='d')
    day = dti[0]
    n_loads = 2748
    types_gen = 2
    year_len = 365
    T = 96
    gen_err_zero = np.zeros((types_gen, 1))  # se si vuole testare la rete con errore nullo
    load_err_zero = np.zeros((n_loads, 1))  # se si vuole testare la rete con errore nullo

    np.random.seed(19970905)
    gen_err_seq = np.random.randn(types_gen, year_len) / 10
    load_err_seq = np.random.randn(n_loads, year_len) / 10

    gen_err = gen_err_seq[:, 0]
    load_err = load_err_seq[:, 0]

    # Se si vuole testare la rete con errore nullo usa gen_err_zero e load_err_zero
    generator = NetworkGenerator(day_date=day, error_gen=gen_err_zero, error_load=load_err_zero)
    network = generator.get_network()

    tc = 1  # Valore iniziale del tc
    env = TCEnvironment(my_network=network, p_dev=generator.get_p_time_series(),
                        q_loads=generator.get_q_time_series(), init_tc=tc)
    env.reset()
    init_obs = env.reset()
    print('Environment reset and ready')

    v_at_df = pd.read_csv('v_at.csv', sep=',')  # tensione lato at
    v_at = np.array(v_at_df)

    # slack = 1
    rew_vett = []
    voltage_vett = []
    env.simulator.v_SB = v_at[301, :]

    for t in range(T):
        a = env.action_space.sample()
        # a = 1
        env.v_SB = env.simulator.v_SB[0]
        slack = env.v_SB
        env.update_values(slack=slack, tc=tc)
        o, r, done, _, tc = env.step(a)
        rew_vett.append(r)
        voltage_vett.append(o)
        print(f't={t}, r_t={r:.3}')

    voltage_vett = np.array(voltage_vett)
    voltage_df = pd.DataFrame(voltage_vett)
    voltage_vett = np.delete(voltage_vett, 0, 1)

    voltage_df.plot(kind='box', color=dict(medians='black', whiskers='darkred'),
                    medianprops=dict(linestyle='-', linewidth=1.5, color='black'))
    '''
    #boxprops=dict(linestyle='-', linewidth=1),
    flierprops=dict(linestyle='-', linewidth=1.5),
    medianprops=dict(linestyle='-', linewidth=1.5),
    whiskerprops=dict(linestyle='-', linewidth=1.5),
    capprops=dict(linestyle='-', linewidth=0.5),
    showfliers=False, grid=True, rot=0,
    color=dict(boxes='c', whiskers='red', medians='black', caps='b'),
    )'''
    plt.xticks(np.arange(1, 275, 50))
    plt.show()


