from collections import OrderedDict
import numpy as np
import copy
from scipy.sparse import csc_matrix
from logging import getLogger

from .components import Load, TransmissionLine, \
    Bus, ClassicalGen
from .components.constants_tc import DEV_H
from . import check_network
from . import solve_load_flow
from .components.devices import Generator

logger = getLogger(__file__)


class Simulator(object):
    """
    A simulator of an AC electricity distribution network.

    Attributes
    ----------
    baseMVA : int
        The base power of the system (MVA).
    delta_t : float
        The fraction of an hour corresponding to the time interval between two
        consecutive time steps :math:`\\Delta t` (e.g., 0.25 means an interval of 15 minutes).
    lamb : int
        The penalty factor associated with violating operating constraints :math:`\\lambda`,
        used in the reward signal.
    buses : dict of {int : :py:class:`Bus`}
        The buses of the grid, where for each {key: value} pair, the key is a
        unique bus ID.
    branches : dict of {(int, int) : `TransmissionLine`}
        The transmission lines of the grid, where for each {key: value} pair, the
        key is a unique transmission line ID.
    devices : dict of {int : `Device`}
        The devices connected to the grid, where for each {key: value} pair, the
        key is a unique device ID.
    N_bus, N_device : int
        The number of buses :math:`|\\mathcal N|` and electrical devices :math:`|\\mathcal D|`
        in the network.
    N_load, N_non_slack_gen, N_des, N_gen_rer : int
        The number of load :math:`|\\mathcal D_L|`, non-slack generators :math:`|\\mathcal D_G|-1`,
        DES devices :math:`|\\mathcal D_{DES}|`, and renewable energy generators :math:`|\\mathcal D_{DER}|`.
    Y_bus : scipy.sparse.csc_matrix
        The (N_bus, N_bus) nodal admittance matrix :math:`\\mathbf Y` of the network as a sparse
        matrix.
    state_bounds : dict of {str : dict}
        The lower and upper bounds that each electrical quantity may take. This
        is a nested dictionary with keys [quantity][ID][unit], where quantity
        is the electrical quantity of interest, ID is the unique bus/device/branch
        ID and unit is the units in which to return the quantity.
    state : dict of {str : numpy.ndarray}
        The current state of the system.
    pfe_converged : bool
        True if the load flow converged; otherwise False (possibly infeasible
        problem).

    Methods
    -------
    reset(np_random, init_soc)
        Reset the simulator.
    get_network_specs()
        Get the operating characteristics of the network.
    get_action_space()
        Get the control action space available.
    transition(P_load, P_potential, P_curt_limit, desired_alpha, Q_storage)
        Simulate a transition of the system from time t to time (t+1).
    """

    def __init__(self, network, delta_t, lamb, PUN, penalty_tc, bus_from, bus_to, v_SB):
        """
        Parameters
        ----------
        network : dict of {str : numpy.ndarray}
            A network dictionary describing the power grid.
        delta_t : float
            The interval of time between two consecutive time steps :math:`\\delta t`,
            in fraction of hour.
        lamb : int
            A constant factor :math:`\\lambda` multiplying the penalty associated
            with violating operational constraints.
        """

        self.delta_t = delta_t
        self.lamb = lamb
        self.PUN = PUN
        self.penalty_tc = penalty_tc
        # Check the correctness of the input network file.
        check_network.check_network_specs(network)

        # Load network.
        self.baseMVA, self.buses, self.branches, self.devices = \
            self._load_case(network)

        # Number of elements in all sets.
        self.N_bus = len(self.buses)
        self.N_device = len(self.devices)
        self.N_load = len([0 for d in self.devices.values() if isinstance(d, Load)])

        # Build the nodal admittance matrix.
        self.Y_bus = self._build_admittance_matrix()

        # Compute the range of possible (P, Q) injections of each bus.
        self._compute_bus_bounds()

        # Summarize the operating range of the network.
        self.state_bounds = self.get_state_space()

        self.state = None
        self.pfe_converged = None

        self.tc = None
        self.current_action = None

        ''' Sistemare lo slack intorno a riga 400'''
        # self.v_SB = np.random.uniform(low=0.999, high=1.001, size=(97,))
        self.v_SB = v_SB

        # Mettere i due bus del tap changer
        self.bus_from = bus_from
        self.bus_to = bus_to

    def update_tc(self, new_tc):
        self.tc = new_tc

    def update_current_action(self, action):
        self.current_action = action

    def _load_case(self, network):
        """
        Initialize the network model based on parameters given in a network file.

        Parameters
        ----------
        network : dict of numpy.ndarray
            A network dictionary describing the power grid.

        Returns
        -------
        baseMVA : int or float
            The base power of the system (MVA).
        buses : OrderedDict of {int : `Bus`}
            The buses of the network, ordered by unique bus ID.
        branches : dict of {(int, int) : `TransmissionLine`}
            The branches of the network.
        devices : OrderedDict of {int : `Device`}
            The electrical devices connected to the grid, ordered by unique device
            ID.

        Raises
        ------
        NoImplementedError
            When the feature DEV_TYPE of a device is not valid.
        """

        baseMVA = network['baseMVA']

        buses = {}
        for bus_spec in network['bus']:
            bus = Bus(bus_spec)
            buses[bus.id] = bus

        # Create an OrderedDict ordered by bus ID.
        buses = OrderedDict(sorted(buses.items(), key=lambda t: t[0]))

        bus_ids = list(buses.keys())

        branches = OrderedDict()  # order branches in the order they are provided.
        for br_spec in network['branch']:
            branch = TransmissionLine(br_spec, baseMVA, bus_ids)
            branches[(branch.f_bus, branch.t_bus)] = branch

        devices = {}
        for dev_spec in network['device']:
            dev_type = int(dev_spec[DEV_H['DEV_TYPE']])

            if dev_type == -1:
                dev = Load(dev_spec, bus_ids, baseMVA)

            elif dev_type == 0:
                dev = ClassicalGen(dev_spec, bus_ids, baseMVA)

            else:
                raise NotImplementedError

            devices[dev.dev_id] = dev

        # Create an OrderedDict sorted by device ID.
        devices = OrderedDict(sorted(devices.items(), key=lambda t: t[0]))

        return baseMVA, buses, branches, devices

    def _build_admittance_matrix(self):
        """
        Build the nodal admittance matrix of the network (in p.u.).
        """
        n = max([i for i in self.buses.keys()])
        Y_bus = np.zeros((n + 1, n + 1), dtype=np.complex)

        for (f, t), br in self.branches.items():
            # Fill an off-diagonal elements of the admittance matrix Y_bus.
            Y_bus[f, t] = - br.series / np.conjugate(br.tap)
            Y_bus[t, f] = - br.series / br.tap

            # Increment diagonal element of the admittance matrix Y_bus.
            Y_bus[f, f] += (br.series + br.shunt) / (np.abs(br.tap) ** 2)
            Y_bus[t, t] += br.series + br.shunt

        return csc_matrix(Y_bus)

    def _compute_bus_bounds(self):
        """
        Compute the range of (P, Q) possible injections at each bus.
        """

        P_min = {i:0 for i in self.buses.keys()}
        P_max = copy.copy(P_min)
        Q_min, Q_max = copy.copy(P_min), copy.copy(P_min)

        # Iterate over all devices connected to the power grid.
        for dev in self.devices.values():
            P_min[dev.bus_id] += dev.p_min
            P_max[dev.bus_id] += dev.p_max
            Q_min[dev.bus_id] += dev.q_min
            Q_max[dev.bus_id] += dev.q_max

        # Update each bus with its operation range.
        for bus_id, bus in self.buses.items():
            if bus_id in P_min.keys():
                bus.p_min = P_min[bus_id]
                bus.p_max = P_max[bus_id]
                bus.q_min = Q_min[bus_id]
                bus.q_max = Q_max[bus_id]

    def reset(self, init_state):
        """
        Reset the simulator.

        The :code:`init_state` vector should have power injections in MW or MVAr and
        state of charge in MWh.

        Parameters
        ----------
        init_state : numpy.ndarray
            The initial state vector :math:`s_0` of the environment.

        Returns
        -------
        pfe_converged : bool
            True if a feasible solution was reached (within the specified
            tolerance) for the power flow equations. If False, it might indicate
            that the network has collapsed (e.g., voltage collapse).
        """

        self.state = None

        # 1. Extract variables to pass to `transition` function.
        P_dev = init_state[:self.N_device]
        Q_dev = init_state[self.N_device: 2 * self.N_device]

        P_load = {}
        Q_load = {}

        for idx, (dev_id, dev) in enumerate(self.devices.items()):
            if isinstance(dev, Load):
                P_load[dev_id] = P_dev[idx]
                Q_load[dev_id] = Q_dev[idx]

        # 3. Compute all electrical quantities in the network.
        _, _, _, _, pfe_converged = \
            self.transition(P_load, Q_load)

        # 5. Re-construct the state dictionary after modifying the SoC.
        self.state = self._gather_state()

        return pfe_converged

    # def get_action_space(self):
    #     """
    #     Return the range of each possible control action.
    #     """
    #     #TODO vedere se devo aggiungere un action space

    def get_state_space(self):
        """
        Returns the range of potential values for all state variables.

        These lower and upper bounds are respected at all timesteps in the
        simulator. For unbounded values, a range of :code:`(-inf, inf)` is used.

        Returns
        -------
        specs : dict of {str : dict}
            A dictionary where keys are the names of the state variables (e.g.,
            {'bus_p', 'bus_q', ...}) and the values are dictionary, indexed with
            the device/branch/bus unique ID, that store dictionaries of
            {units : (lower bound, upper bound)}.
        """

        # Bus bounds.
        bus_p, bus_q = {}, {}
        bus_v_magn, bus_v_ang = {}, {}
        bus_i_magn, bus_i_ang = {}, {}
        for bus_id, bus in self.buses.items():
            bus_p[bus_id] = {'MW': (bus.p_min * self.baseMVA, bus.p_max * self.baseMVA),
                             'pu': (bus.p_min, bus.p_max)}
            bus_q[bus_id] = {'MVAr': (bus.q_min * self.baseMVA, bus.q_max * self.baseMVA),
                             'pu': (bus.q_min, bus.q_max)}
            if bus.is_slack:
                bus_v_magn[bus_id] = {'pu': (bus.v_slack, bus.v_slack),
                                      'kV': (bus.v_slack * bus.baseKV, bus.v_slack * bus.baseKV)}
                bus_v_ang[bus_id] = {'degree': (0, 0),
                                     'rad': (0, 0)}
            else:
                bus_v_magn[bus_id] = {'pu': (- np.inf, np.inf),
                                      'kV': (- np.inf, np.inf)}
                bus_v_ang[bus_id] = {'degree': (- 180, 180),
                                     'rad': (- np.pi, np.pi)}
            bus_i_magn[bus_id] = {'pu': (- np.inf, np.inf),
                                  'kA': (- np.inf, np.inf)}
            bus_i_ang[bus_id] = {'degree': (- 180, 180),
                                 'rad': (- np.pi, np.pi)}

        # Device bounds.
        dev_p, dev_q = {}, {}
        for dev_id, dev in self.devices.items():
            dev_p[dev_id] = {'MW': (dev.p_min * self.baseMVA, dev.p_max * self.baseMVA),
                             'pu': (dev.p_min, dev.p_max)}
            dev_q[dev_id] = {'MVAr': (dev.q_min * self.baseMVA, dev.q_max * self.baseMVA),
                             'pu': (dev.q_min, dev.q_max)}

        # Branch bounds.
        branch_p, branch_q = {}, {}
        branch_s, branch_i_magn, branch_i_ang = {}, {}, {}
        for br_id, branch in self.branches.items():
            branch_p[br_id] = {'MW': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_q[br_id] = {'MVAr': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_s[br_id] = {'MVA': (- np.inf, np.inf),
                               'pu': (- np.inf, np.inf)}
            branch_i_magn[br_id] = {'pu': (- np.inf, np.inf),
                                    'kA': (- np.inf, np.inf)}
            branch_i_ang[br_id] = {'rad': (- np.pi, np.pi),
                                   'degree': (- 180, 180)}

        specs = {'bus_p': bus_p, 'bus_q': bus_q,
                 'bus_v_magn': bus_v_magn, 'bus_v_ang': bus_v_ang,
                 'bus_i_magn': bus_i_magn, 'bus_i_ang': bus_i_ang,
                 'dev_p': dev_p, 'dev_q': dev_q,
                 'branch_p': branch_p, 'branch_q': branch_q,
                 'branch_s': branch_s,
                 'branch_i_magn': branch_i_magn, 'branch_i_ang': branch_i_ang
                }

        return specs

    def transition(self, P_load, Q_load):
        """
        Simulate a transition of the system from time :math:`t` to time :math:`t+1`.

        This function simulates a transition of the system after actions were
        taken by the DSO. The results of these decisions then affect the new
        state of the system, and the associated reward is returned.

        Parameters
        ----------
        P_load : dict of {int : float}
            A dictionary with device IDs as keys and fixed real power injection :math:`P_l^{(dev)}`
            (MW) as values (load devices only).

        Returns
        -------
        state : dict of {str : numpy.ndarray}
            The new state of the system :math:`s_{t+1}`.
        reward : float
            The reward :math:`r_t` associated with the transition.
        e_loss : float
            The total energy loss :math:`\\Delta E_{t:t+1}` (MWh).
        penalty : float
            The total penalty :math:`\\lambda \\phi(s_{t+1})` due to violation of operating constraints.
        pfe_converged : bool
            True if a feasible solution was reached (within the specified
            tolerance) for the power flow equations. If False, it might indicate
            that the network has collapsed (e.g., voltage collapse).
        """

        for dev_id, dev in self.devices.items():

            # 1. Compute the (P, Q) injection point of each load.
            if isinstance(dev, Load):
                dev.map_pq(P_load[dev_id] / self.baseMVA,
                           Q_load[dev_id] / self.baseMVA)

            # 4a. Initialize the (P, Q) injection point of the slack bus device to 0.
            elif dev.is_slack:
                dev.p = 0.
                dev.q = 0.

        # 4b. Compute the total (P, Q) injection at each bus.
        self._get_bus_total_injections()

        self.branches[(self.bus_from, self.bus_to)].tap_magn = self.tc
        self.branches[(self.bus_from, self.bus_to)].tap = self.branches[(self.bus_from, self.bus_to)].tap_magn + 0j
        self.Y_bus[self.bus_from, self.bus_to] = - self.branches[self.bus_from, self.bus_to].series / np.conjugate(self.tc)
        self.Y_bus[self.bus_to, self.bus_from] = - self.branches[self.bus_from, self.bus_to].series / self.tc
        self.Y_bus[self.bus_from, self.bus_from] = self.branches[self.bus_from, self.bus_to].series / (np.abs(self.tc) ** 2)

        '''Modificare qua sotto per avere lo slack variabile'''
        # vsb = self.v_SB  # Se si vuole testare la rete con uno slack bassa con tensione fissa
        vsb = self.v_SB[0]
        self.v_SB = np.delete(self.v_SB, 0)

        # 5. Solve the network equations and compute nodal V, P, and Q vectors.
        _, self.pfe_converged = \
            solve_load_flow.solve_pfe_newton_raphson(self, vsb, xtol=1e-5)

        # 6. Construct the new state of the network.
        self.state = self._gather_state()

        # 7. Compute the reward associated with the transition.
        reward, e_loss, penalty = self._compute_reward()

        return self.state, reward, e_loss, penalty, self.pfe_converged

    def _get_bus_total_injections(self):
        """
        Compute the total (P, Q) injection point at each bus.
        """
        for bus in self.buses.values():
            bus.p = 0.
            bus.q = 0.

        for dev in self.devices.values():
            self.buses[dev.bus_id].p += dev.p
            self.buses[dev.bus_id].q += dev.q

    def _gather_state(self):
        """
        Gather all electrical quantities of the network in a single dictionary.

        All values are gathered in all supported units.
        """

        # Collect bus variables.
        bus_p, bus_q = {'pu': {}, 'MW': {}}, {'pu': {}, 'MVAr': {}}
        bus_v_magn, bus_v_ang = {'pu': {}, 'kV': {}}, {'rad': {}, 'degree': {}}
        bus_i_magn, bus_i_ang = {'pu': {}, 'kA': {}}, {'rad': {}, 'degree': {}}
        for bus_id, bus in self.buses.items():
            bus_p['pu'][bus_id] = bus.p
            bus_p['MW'][bus_id] = bus.p * self.baseMVA

            bus_q['pu'][bus_id] = bus.q
            bus_q['MVAr'][bus_id] = bus.q * self.baseMVA

            bus_v_magn['pu'][bus_id] = np.abs(bus.v)
            bus_v_magn['kV'][bus_id] = np.abs(bus.v) * bus.baseKV

            bus_v_ang['rad'][bus_id] = np.angle(bus.v)
            bus_v_ang['degree'][bus_id] = np.angle(bus.v) * 180 / np.pi

            bus_i_magn['pu'][bus_id] = np.abs(bus.i)
            bus_i_magn['kA'][bus_id] = np.abs(bus.i) * self.baseMVA / bus.baseKV

            bus_i_ang['rad'][bus_id] = np.angle(bus.i)
            bus_i_ang['degree'][bus_id] = np.angle(bus.i) * 180 / np.pi

        # Collect device variables.
        dev_p, dev_q = {'pu': {}, 'MW': {}}, {'pu': {}, 'MVAr': {}}
        for dev_id, dev in self.devices.items():
            dev_p['pu'][dev_id] = dev.p
            dev_p['MW'][dev_id] = dev.p * self.baseMVA

            dev_q['pu'][dev_id] = dev.q
            dev_q['MVAr'][dev_id] = dev.q * self.baseMVA


        # Collect branch variables.
        branch_p, branch_q = {'pu': {}, 'MW': {}}, {'pu': {}, 'MVAr': {}}
        branch_s = {'pu': {}, 'MVA': {}}
        branch_i_magn, branch_i_ang = {'pu': {}}, {'rad': {}, 'degree': {}}
        for (f, t), branch in self.branches.items():
            branch_p['pu'][(f, t)] = branch.p_from
            branch_p['MW'][(f, t)] = branch.p_from * self.baseMVA

            branch_q['pu'][(f, t)] = branch.q_from
            branch_q['MVAr'][(f, t)] = branch.q_from * self.baseMVA

            branch_s['pu'][(f, t)] = branch.s_apparent_max
            branch_s['MVA'][(f, t)] = branch.s_apparent_max * self.baseMVA

            branch_i_magn['pu'][(f, t)] = np.sign(branch.i_from).real * np.abs(branch.i_from)

            branch_i_ang['rad'][(f, t)] = np.angle(branch.i_from)
            branch_i_ang['degree'][(f, t)] = np.angle(branch.i_from) * 180 / np.pi

        state = {'bus_p': bus_p,
                 'bus_q': bus_q,
                 'bus_v_magn': bus_v_magn,
                 'bus_v_ang': bus_v_ang,
                 'bus_i_magn': bus_i_magn,
                 'bus_i_ang': bus_i_ang,
                 'dev_p': dev_p,
                 'dev_q': dev_q,
                 'branch_p': branch_p,
                 'branch_q': branch_q,
                 'branch_s': branch_s,
                 'branch_i_magn': branch_i_magn,
                 'branch_i_ang': branch_i_ang
                 }

        return state

    def _compute_reward(self):
        """
        Return the total reward associated with the current state of the system.

        The reward is computed as a negative sum of transmission
        losses, curtailment losses, (dis)charging losses, and operational
        constraints violation costs.

        Returns
        -------
        reward : float
            The total reward associated with the transition to a new system
            state.
        e_loss : float
            The total energy loss (p.u. per hour).
        penalty : float
            The total penalty due to violation of operating constraints (p.u. per
            hour).
        """

        # Compute the energy loss.
        e_loss = 0.
        for dev in self.devices.values():
            if isinstance(dev, (Generator, Load)):
                e_loss += dev.p

        e_loss *= self.delta_t * self.PUN  # gia energia

        # Compute the penalty term.
        penalty = 0.
        for bus in self.buses.values():
            v_magn = np.abs(bus.v)
            penalty += np.maximum(0, v_magn - bus.v_max) \
                       + np.maximum(0, bus.v_min - v_magn)

        penalty *= self.delta_t * self.lamb

        tc_move = self.penalty_tc if (self.current_action != 1) else 0
        # if self.current_action != 1:
            # print("Current action: {}".format(self.current_action))
        # print("Additional penalty: {}".format(tc_move))
        # TODO
        penalty += tc_move
        # Compute the total reward.
        reward = - (e_loss + penalty)

        return reward, e_loss, penalty
