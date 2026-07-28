"""Shared simulation core for the Q-LoRa study.

This module is the single source of truth for the LoRa physical layer, the
domain objects (``myBS`` / ``myNode`` / ``myPacket``), the ``airtime`` formula,
the independent-interference collision model and the ``LoRaNetwork`` environment
used for tabular multi-agent Q-learning on spreading-factor (SF) allocation.

It merges the three notebook variants that used to live in this repo:

* ``Cooperative.*`` -- shared/global reward (paper Eq. 2);
* ``Competitive.*`` -- individual per-node reward (paper Eq. 1);
* ``Q_LoRA_V1.ipynb`` -- cooperative reward **plus** an ADR baseline.

The reward is selected with ``reward_mode`` and ADR is a baseline strategy
toggled with ``adr=True`` (it manages each node's SF itself and ignores the
action). Variable/function names mirror the LoRa physical layer and the paper
(``sf``, ``bw``, ``cr``, ``Ptx``, ``pdr``, ``toa``, ``n_agents``,
``n_threshold`` ...) so the code stays readable against both.

The original notebooks depended on ``gym`` only for the ``Env`` base class,
``MultiDiscrete`` spaces and the seeded RNG. ``gym`` is unmaintained and does
not install on modern Python/NumPy, so those are replaced with a plain NumPy
``Generator``; behaviour is otherwise preserved. Everything that consumes
randomness inside an episode (node placement, transmit decision, arrival
jitter) draws from ``self.np_random``, so an episode is fully reproducible given
its seed and the action sequence.
"""

import math
import datetime as dt

import numpy as np

# --------------------------------------------------------------------------- #
# LoRa physical-layer constants (log-distance shadowing model, from LoRaSim)
# --------------------------------------------------------------------------- #
Ptx = 14          # transmit power [dBm]
gamma = 2.08      # path-loss exponent
d0 = 40.0         # reference distance [m]
var = 0           # shadowing variance (unused, kept for parity)
Lpld0 = 127.41    # path loss at d0 [dB]
GL = 0            # antenna / cable gain [dB]
noise = -110      # noise floor [dBm], used for the ADR SNR estimate

adr_margin = 6    # ADR installation margin [dB]

# Receiver sensitivity per SF for BW = 125 / 250 / 500 kHz [dBm].
sf7 = np.array([7, -123, -120, -116])
sf8 = np.array([8, -126, -123, -119])
sf9 = np.array([9, -129, -125, -122])
sf10 = np.array([10, -132, -128, -125])
sf11 = np.array([11, -134.5, -130, -128])
sf12 = np.array([12, -137, -133, -130])
sensi = np.array([sf7, sf8, sf9, sf10, sf11, sf12])

minsensi = -112.0
Lpl = Ptx - minsensi

# Required demodulation SNR per SF [dB], used by the ADR baseline.
adr_snr_req = {7: -7.5, 8: -10, 9: -12.5, 10: -15, 11: -17.5, 12: -20}

# Simulation area (a single gateway sits at the centre).
maxX = 500
maxY = 500

# --------------------------------------------------------------------------- #
# Derived maximum communication distances per SF / bandwidth.
# --------------------------------------------------------------------------- #
Lpl_matrix = Ptx - (sensi[:, 1:])
maxDist = np.dot(d0, np.exp(np.divide((Lpl_matrix - Lpld0), (10 * gamma))))
# ``minDist`` is the SF7 (highest data-rate) range at 125 kHz -- ~63 m, matching
# the paper's loc_range = 64 m. ``bestDist`` is the SF12 range at 125 kHz.
minDist = float(np.amin(maxDist[:, 0]))
bestDist = float(np.max(maxDist[:, 0]))


def default_loc_range():
    """Default node-placement radius (SF7 125 kHz range, ~63 m)."""
    return minDist


# --------------------------------------------------------------------------- #
# Domain objects
# --------------------------------------------------------------------------- #
class myBS:
    """A base station / gateway."""

    def __init__(self, id, position):
        self.id = id
        self.x, self.y = position
        self.packet = []


class myNode:
    """An end device: position, radio config and per-node counters."""

    def __init__(self, id, position=(1, 1), period=20000):
        self.id = id
        self.x, self.y = position
        self.period = period
        self.dist = []
        self.freq = 868.3
        self.packetlen = 50
        self.packet = []
        self.snrList = []       # recently observed SNRs, consumed by ADR
        self.bw = 125
        self.sf = 12
        self.cr = 1
        self.sent = 0
        self.delivered = 0
        self.lost = 0
        self.total_airtime = 0

    def getTransmissionParams(self):
        return self.freq, self.bw, self.sf, self.cr

    def updateSF(self, sf):
        self.sf = sf

    def updatePos(self, position):
        self.x, self.y = position

    def updateDist(self, bsList):
        self.dist = []
        for b in bsList:
            d = math.sqrt((self.x - b.x) ** 2 + (self.y - b.y) ** 2)
            self.dist.append(d)


class myPacket:
    """One uplink transmission: RSSI/SNR, symbol/airtime and collision flags."""

    def toString(self):
        return "Packet - Node:" + str(self.node) + " BS: " + str(self.bs) + " SF: " + str(self.sf)

    def __init__(self, node, plen, distance, freq, bs, sf, cr, bw):
        self.bs = bs
        self.node = node
        self.txpow = Ptx

        self.sf = sf
        self.cr = cr
        self.bw = bw

        # Log-distance path loss (transmit power is the global Ptx).
        if distance == 0:
            distance = 0.00001
        Lpl = Lpld0 + 10 * gamma * math.log10(distance / d0)
        Prx = Ptx - GL - Lpl

        self.transRange = 150
        self.pl = plen
        self.symTime = (2 ** self.sf) / self.bw
        self.rssi = Prx
        self.snr = self.rssi - noise
        self.freq = freq
        self.addTime = None
        self.rectime = dt.timedelta(milliseconds=airtime(self))
        self.collided = False
        self.lost = False


# --------------------------------------------------------------------------- #
# Airtime (LoRa design guide) and the independent-interference collision model
# --------------------------------------------------------------------------- #
def airtime(packet):
    """Time-on-air of a packet [ms], per LoraDesignGuide_STD.pdf."""
    H = 0        # implicit header disabled
    DE = 0       # low data-rate optimization
    Npream = 8   # preamble symbols

    if packet.bw == 125 and packet.sf in [11, 12]:
        # low data-rate optimization is mandatory for BW125 + SF11/SF12
        DE = 1

    Tsym = packet.symTime
    Tpream = (Npream + 4.25) * Tsym
    payloadSymbNB = 8 + max(
        math.ceil((8.0 * packet.pl - 4.0 * packet.sf + 28 + 16 - 20 * H)
                  / (4.0 * (packet.sf - 2 * DE))) * (packet.cr + 4), 0)
    Tpayload = payloadSymbNB * Tsym
    return Tpream + Tpayload


def frequencyCollision(p1, p2):
    """True if two packets are close enough in frequency to interfere."""
    if abs(p1.freq - p2.freq) <= 120 and (p1.bw == 500 or p2.freq == 500):
        return True
    elif abs(p1.freq - p2.freq) <= 60 and (p1.bw == 250 or p2.freq == 250):
        return True
    else:
        if abs(p1.freq - p2.freq) <= 30:
            return True
    return False


def sfCollision(p1, p2):
    """True if two packets share the same spreading factor."""
    return p1.sf == p2.sf


def powerCollision(p1, p2):
    """Return the packet(s) lost to a power-domain (capture) collision."""
    powerThreshold = 6  # dB
    if abs(p1.rssi - p2.rssi) < powerThreshold:
        return (p1, p2)
    elif p1.rssi - p2.rssi < powerThreshold:
        return (p1,)
    return (p2,)


def timingCollision(p1, p2):
    """True if p1 overlaps p2's critical (preamble) section."""
    Npream = 8
    if p1.addTime > p2.addTime:
        Tpreamb = p1.symTime * (Npream - 5)
        p2_end = p2.addTime + p2.rectime
        p1_cs = p1.addTime + dt.timedelta(milliseconds=Tpreamb)
    else:
        Tpreamb = p2.symTime * (Npream - 5)
        p2_end = p1.addTime + p1.rectime
        p1_cs = p2.addTime + dt.timedelta(milliseconds=Tpreamb)
    return p1_cs < p2_end


# --------------------------------------------------------------------------- #
# The RL environment
# --------------------------------------------------------------------------- #
class LoRaNetwork:
    """Tabular multi-agent SF-allocation environment.

    Parameters
    ----------
    max_steps : int
        Number of discrete timesteps (transmission rounds) per episode.
    n_agents : int
        Number of end devices ``N``.
    n_threshold : int
        Number of occupancy threshold levels ``T`` (state abstraction).
    period : int
        Max arrival jitter within a round [ms] (paper's ``delta_n``).
    loc_range : float
        Node-placement radius [m]. Defaults to the SF7 range (~63 m).
    tx_prob : float
        Per-node probability of transmitting in a round.
    beta : float
        Energy/PDR trade-off weight in the reward.
    reward_mode : {"cooperative", "competitive"}
        ``cooperative`` -> single network-wide scalar reward (Eq. 2).
        ``competitive`` -> per-node reward array (Eq. 1).
    adr : bool
        If True the environment ignores the action and drives each node's SF
        with a distance/SNR-based ADR controller (a baseline strategy).
    init_sf : int
        SF assigned to every node on ``reset`` (ADR overrides this to 12).
    verbose : bool
        Enable the notebooks' diagnostic prints (lost packets, ADR steps).
    """

    N_ACTIONS = 6  # SF7..SF12

    def __init__(self, max_steps=100, n_agents=100, n_threshold=6, period=12000,
                 loc_range=None, tx_prob=0.99, beta=0.25,
                 reward_mode="cooperative", adr=False, init_sf=7, verbose=False):
        if reward_mode not in ("cooperative", "competitive"):
            raise ValueError("reward_mode must be 'cooperative' or 'competitive'")
        self.n_agents = n_agents
        self.adr = adr
        self.beta = beta
        self.loc_range = default_loc_range() if loc_range is None else loc_range
        self.tx_prob = tx_prob
        self.reward_mode = reward_mode
        self.init_sf = init_sf
        self.verbose = verbose

        self.bsList = []
        self.n_bs = 1
        self.period = period
        self._max_steps = max_steps
        self._step_count = None
        self.agents = []
        self.datetime = None
        self.full_collision = True

        # State abstraction: thresholds for the per-SF occupancy histogram.
        self._n_threshold = n_threshold
        self.full_threshold = False
        n = 1
        if self._n_threshold > self.n_agents:
            # Degenerate case (T > N): use raw counts, no abstraction.
            self.full_threshold = True
            self.arr_thresholds = np.arange(stop=self._n_threshold, step=1)
        else:
            if self.n_agents >= 2 * self._n_threshold:
                n = 2.5
            self.thresholds = [round(i * self.n_agents / n / self._n_threshold)
                               for i in range(n_threshold)]
            self.arr_thresholds = np.array(self.thresholds)
            if self.n_agents / 5 / self._n_threshold < 1:
                self.arr_thresholds_high = np.arange(stop=self._n_threshold, step=1)
            else:
                self.arr_thresholds_high = np.arange(
                    stop=self.n_agents / 5,
                    step=math.ceil(self.n_agents / 5 / self._n_threshold),
                )[:self._n_threshold]

        # Airtime bounds used to normalise the energy term of the reward.
        self.max_air_time = airtime(myPacket(1, 50, 1, 868.3, 1, 12, 1, 125))
        self.min_air_time = airtime(myPacket(2, 50, 1, 868.3, 1, 7, 1, 125))

        self.np_random = np.random.default_rng()

    # -- state / action indexing ------------------------------------------- #
    def state_index(self, state):
        """Flatten a 6-element threshold state to a scalar Q-table index."""
        return sum(int(state[i]) * self._n_threshold ** i for i in range(len(state)))

    def action_index(self, action):
        """Flatten a per-agent action vector to a scalar index (base 6)."""
        return sum(int(action[i]) * 6 ** i for i in range(len(action)))

    def sample_action(self):
        """A random joint action (one SF index per agent)."""
        return self.np_random.integers(0, self.N_ACTIONS, size=self.n_agents)

    # -- observation ------------------------------------------------------- #
    def get_threshold_level(self, n, SF):
        if SF <= 3:
            return sum(l <= n for l in self.arr_thresholds) - 1
        else:
            return sum(l <= n for l in self.arr_thresholds_high) - 1

    def observation(self):
        SF_obs = np.zeros(6, dtype=np.int8)
        for agent in self.agents:
            SF_obs[agent.sf - 7] += 1
        if self.full_threshold:
            if self.verbose:
                print("Obs: ", SF_obs)
            return SF_obs
        SF_obs_t = np.zeros(6, dtype=np.int8)
        for i in range(6):
            SF_obs_t[i] = self.get_threshold_level(SF_obs[i], i)
        return SF_obs_t

    # -- episode lifecycle ------------------------------------------------- #
    def reset(self, seed=None, options=None):
        self.np_random = np.random.default_rng(seed)
        self.agents.clear()
        self.bsList.clear()
        self.datetime = dt.datetime.now()
        self.bsList.append(myBS(id=0, position=(0, 0)))
        self._step_count = 0
        # Uniform placement inside a disc of radius ``loc_range``.
        t = self.np_random.uniform(0, 1, size=self.n_agents)
        u = self.np_random.uniform(0, 1, size=self.n_agents)
        x = self.loc_range * np.sqrt(t) * np.cos(2 * np.pi * u)
        y = self.loc_range * np.sqrt(t) * np.sin(2 * np.pi * u)
        for i in range(self.n_agents):
            agent = myNode(id=i, period=self.period)
            agent.updatePos((x[i], y[i]))
            agent.updateSF(12 if self.adr else self.init_sf)
            agent.updateDist(self.bsList)
            self.agents.append(agent)
        return self.observation()

    def transmit(self, node):
        node.sent += 1
        packet = myPacket(node.id, node.packetlen, node.dist[0], node.freq, 0,
                          node.sf, node.cr, node.bw)
        node.total_airtime += packet.rectime.total_seconds() * 1000
        sensitivity = sensi[packet.sf - 7, [125, 250, 500].index(packet.bw) + 1]
        time_change = dt.timedelta(milliseconds=self.np_random.uniform(0, node.period))
        packet.addTime = self.datetime + time_change
        if packet.rssi < sensitivity:
            if self.verbose:
                print("node {}: rssi {} sf {} bw {} < sensitivity {}; packet lost"
                      .format(node.id, packet.rssi, packet.sf, packet.bw, sensitivity))
            packet.lost = True
            node.lost += 1
        else:
            packet.lost = False
            self.bsList[packet.bs].packet.append(packet)
        node.packet.append(packet)

    def calculate_reward(self, node):
        """Per-node reward (competitive / Eq. 1): PDR minus normalised ToA."""
        if node.sent == 0:
            return 0
        toa_last = airtime(node.packet[-1])
        return (node.delivered / node.sent
                - self.beta * (toa_last - self.min_air_time)
                / (self.max_air_time - self.min_air_time))

    def adr_update(self):
        """Distance/SNR-based ADR: adjust each node's SF every 100 rounds.

        For each node it looks at the strongest SNR observed since the last
        update, computes the margin over the SF's demodulation requirement
        (plus an installation margin) and moves the SF toward SF7 when there is
        spare margin or toward SF12 when the link is weak. With no successful
        reception it conservatively bumps the SF up by one.
        """
        if not self.adr or self._step_count % 100 != 0:
            return
        if self.verbose:
            print("ADR step: ", self._step_count)
        for node in self.agents:
            if node.snrList:
                maxSNR = max(node.snrList)
                snr_margin = maxSNR - adr_snr_req[node.sf] - adr_margin
                snr_var = round(snr_margin / 3)
                if snr_var > 0:
                    if node.sf > 7:
                        node.sf = max(7, node.sf - snr_var)
                else:
                    if node.sf < 12:
                        node.sf = min(12, node.sf + snr_var)
                node.snrList = []
            else:
                if node.sf < 12:
                    node.sf += 1

    def step(self, action):
        sent = 0
        delivered = 0
        total_airtime = 0
        agents_id = set()
        self._step_count += 1
        self.datetime = self.datetime + dt.timedelta(milliseconds=100000)
        done = self._step_count >= self._max_steps

        if self.adr:
            self.adr_update()

        for i, agent in enumerate(self.agents):
            if self.np_random.random() < self.tx_prob:
                agents_id.add(agent.id)
                if not self.adr:
                    agent.updateSF(int(action[i]) + 7)
                self.transmit(agent)
                sent += 1

        # Resolve collisions at each base station.
        for b in self.bsList:
            b.packet.sort(key=lambda x: x.addTime)
            for _ in range(len(b.packet)):
                packet = b.packet.pop()
                total_airtime += packet.rectime.total_seconds() * 1000
                if not packet.collided and not packet.lost:
                    if self.checkcollision(packet):
                        packet.collided = True
                    else:
                        if self.adr:
                            self.agents[packet.node].snrList.append(packet.snr)
                        delivered += 1
                        self.agents[packet.node].delivered += 1
            b.packet = []

        avg_airtime = total_airtime / sent if sent else 0.0
        del_factor = delivered / total_airtime if total_airtime else 0.0

        if self.reward_mode == "competitive":
            reward = np.zeros(self.n_agents)
            pdr = np.zeros(self.n_agents)
            for node_id in agents_id:
                reward[node_id] = self.calculate_reward(self.agents[node_id])
                node = self.agents[node_id]
                pdr[node_id] = node.delivered / node.sent
            info = {'pdr': float(pdr.mean()), 'avg_airtime': avg_airtime,
                    'del_factor': del_factor}
        else:
            pdr = delivered / sent if sent else 0.0
            reward = pdr - self.beta * (avg_airtime - self.min_air_time) \
                / (self.max_air_time - self.min_air_time)
            info = {'pdr': pdr, 'avg_airtime': avg_airtime,
                    'del_factor': del_factor}

        return self.observation(), reward, done, info

    def checkcollision(self, packet):
        """Independent-interference check for a packet against the BS buffer."""
        col = False
        if packet.lost:
            return col
        for other in self.bsList[packet.bs].packet:
            if other.node != packet.node:
                if frequencyCollision(packet, other) and sfCollision(packet, other):
                    if self.full_collision:
                        if timingCollision(packet, other):
                            col = True
                            other.collided = True
                    else:
                        other.collided = True
                        col = True
        return col
