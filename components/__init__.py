import copy
from collections import defaultdict

import numpy as np

import param

from imagen import Composite, Gaussian, Disk
from imagen.random import UniformRandom

import topo
from topo.base.simulation import EPConnectionEvent
from topo.sheet import JointNormalizingCFSheet_Continuous
from topo.base.projection import SheetMask
from topo.submodel import Model

class MultiPortSheet(JointNormalizingCFSheet_Continuous):
    """
    MultiPortSheet is a special Sheet class, which supports receiving, sending
    and combining inputs from different ports
    """

    src_ports = ['Activity', 'Subthreshold']

    def activate(self):
        """
        Collect activity from each projection, combine it to calculate
        the activity for this sheet, and send the result out.

        Subclasses may override this method to whatever it means to
        calculate activity in that subclass.
        """

        # Initialize temporary datastructures and reset activities
        self.activity *= 0.0
        tmp_dict = {}
        tmp_dict['Activity'] = {}
        port_activities = {}
        port_activities['Activity'] = self.activity.copy() * 0.0

        for proj in self.in_connections:
            if proj.activity_group != None:
                if type(proj.activity_group) == type([]):
                    activity_groups = [ag for ag in proj.activity_group]
                else:
                    activity_groups = [proj.activity_group]

                for ag in activity_groups:
                    # If it's a simple activity group, simply append the
                    # projection appropriate priority group of the
                    # 'Activity' port.
                    if len(ag) == 2:
                        if not ag[0] in tmp_dict['Activity']:
                            tmp_dict['Activity'][ag[0]] = []
                        tmp_dict['Activity'][ag[0]].append((proj, ag[1]))
                    # If a multi-port activity group is supplied
                    else:
                        # Check it's listed in the source ports
                        if ag[2] not in self.src_ports:
                            self.src_ports.append(ag[2])
                        # Make sure it has an entry in the temporary port list
                        if not ag[2] in tmp_dict:
                            tmp_dict[ag[2]] = {}
                        # Make sure the priority group exists in the port group
                        if not ag[0] in tmp_dict[ag[2]]:
                            tmp_dict[ag[2]][ag[0]] = []
                        # Reset the ports activity
                        if not ag[2] in port_activities:
                            port_activities[ag[2]] = self.activity.copy() * 0.0
                        tmp_dict[ag[2]][ag[0]].append((proj, ag[1]))

        # Iterate over the ports and priority groups and accumulate the
        # activities.
        for port in tmp_dict:
            priority_keys = tmp_dict[port].keys()
            priority_keys.sort()
            for priority in priority_keys:
                tmp_activity = self.activity.copy() * 0.0
                for proj, op in tmp_dict[port][priority]:
                    tmp_activity += proj.activity
                port_activities[port] = tmp_dict[port][priority][0][1](
                    port_activities[port], tmp_activity)

        self.activity = port_activities['Activity']

        # Send output on 'Subthreshold' port
        self.send_output(src_port='Subthreshold', data=self.activity)

        # Apply the output_fns to the activity
        if self.apply_output_fns:
            for of in self.output_fns:
                of(self.activity)
                for act in port_activities.values():
                    of(act)

        # Send output on 'Activity' port
        self.send_output(src_port='Activity', data=self.activity)

        # Send output on all other ports
        for port, data in port_activities.items():
            if port != "Activity":
                self.send_output(src_port=port, data=data)


    def send_output(self, src_port=None, data=None):
        """Send some data out to all connections on the given src_port."""

        out_conns_on_src_port = [conn for conn in self.out_connections
                                 if self._port_match(conn.src_port, [src_port])]

        for conn in out_conns_on_src_port:
            self.verbose(
                "Sending output on src_port %s via connection %s to %s" %
                (str(src_port), conn.name, conn.dest.name))
            e = EPConnectionEvent(self.simulation.convert_to_time_type(
                conn.delay) + self.simulation.time(), conn, data)
            self.simulation.enqueue_event(e)

Model.register_decorator(MultiPortSheet)

class GaussianBinaryDisk(Composite):

    gaussian_size = param.Number(default=1.0,doc="Size of the Gaussian pattern.")

    noise_scale = param.Number(default=1.0)

    aspect_ratio  = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="""
        Ratio of gaussian width to height; width is gaussian_size*aspect_ratio.""")

    offset = param.Number(default=0.0,bounds=(-1.0,1.0))


    def __call__(self, **params_to_override):
        p = param.ParamOverrides(self, params_to_override)
        gauss = Gaussian(aspect_ratio=p.aspect_ratio, size=p.gaussian_size)
        unirand = UniformRandom(scale=p.noise_scale)
        gaussrand = Composite(generators=[gauss, unirand], operator=np.add)
        p.generators = [gaussrand, Disk(smoothing=0.0, size=1.0)]
        p.operator = np.multiply
        mat = super(GaussianBinaryDisk, self).__call__(**p)
        mat = mat - mat.min()
        mat /= mat.max()

        return (mat + p.offset).round()


class GaussianAdditiveCloud(Composite):
    gaussian_size = param.Number(default=1.0,
                                 doc="Size of the Gaussian pattern.")

    aspect_ratio = param.Number(default=1.0, bounds=(0.0, None),
                                softbounds=(0.0, 2.0),
                                precedence=0.31, doc="""
        Ratio of gaussian width to height; width is gaussian_size*aspect_ratio.""")

    noise_scale = param.Number(default=1.0)


    def __call__(self, **params_to_override):
        p = param.ParamOverrides(self, params_to_override)
        gauss = Gaussian(aspect_ratio=p.aspect_ratio, size=p.gaussian_size)
        unirand = UniformRandom(scale=p.noise_scale)
        p.generators = [gauss, unirand]
        p.operator = np.add
        mat = super(GaussianAdditiveCloud, self).__call__(**p)
        mat = mat - mat.min()

        return mat/mat.max()

from topo.transferfn.misc import TransferFnWithState

class DivideWithConstant(param.Parameterized):
    """
    Divide two scalars or arrays with a constant (c) offset on the
    denominator to allow setting the gain or to avoid divide-by-zero
    issues. The result is clipped to ensure that it has only positive
    values.
    """
    c = param.Number(default=1.0)

    def __call__(self, x, y):
        return np.divide(x,np.maximum(y+self.c,0))


class MultiplyWithConstant(param.Parameterized):
    """
    Allows multiplying with a constant offset parameter.
    """

    c = param.Number(default=1.0)

    def __call__(self, x, y):
        return np.multiply(x, np.maximum(y+self.c, 0))


class SynapticScaling(TransferFnWithState):
    """
    SynapticScaling is a homeostatic mechanism to scale
    excitatory input onto inhibitory neurons to maintain
    a constant level of activity in a network with both
    excitatory and inhibitory neurons.
    """

    s_init = param.Number(default=1.0,doc="""
        Initial value of the threshold value t.""")

    randomized_init = param.Boolean(False,doc="""
        Whether to randomize the initial t parameter.""")

    seed = param.Integer(default=42, doc="""
       Random seed used to control the initial randomized scaling
       factors.""")

    target_activity = param.Number(default=0.024,doc="""
        The target average activity.""")

    learning_rate = param.Number(default=0.01,doc="""
        Learning rate for homeostatic plasticity.""")

    smoothing = param.Number(default=0.991,doc="""
        Weighting of previous activity vs. current activity when
        calculating the average activity.""")

    noise_magnitude =  param.Number(default=0.1,doc="""
        The magnitude of the additive noise to apply to the s_init
        parameter at initialization.""")

    period = param.Number(default=1.0, constant=True, doc="""
        How often the synaptic scaling factor should be adjusted.

        If the period is 0, the threshold is adjusted continuously, each
        time this TransferFn is called.

        For nonzero periods, adjustments occur only the first time
        this TransferFn is called after topo.sim.time() reaches an
        integer multiple of the period.

        For example, if period is 2.5 and the TransferFn is evaluated
        every 0.05 simulation time units, the threshold will be
        adjusted at times 2.55, 5.05, 7.55, etc.""")


    def __init__(self,**params):
        super(SynapticScaling,self).__init__(**params)
        self.first_call = True
        self.__current_state_stack=[]
        self.t=None     # To allow state_push at init
        self.y_avg=None # To allow state_push at init

        next_timestamp = topo.sim.time() + self.period
        self._next_update_timestamp = topo.sim.convert_to_time_type(next_timestamp)
        self._y_avg_prev = None
        self._x_prev = None


    def _initialize(self,x):
        self._x_prev = np.copy(x)
        self._y_avg_prev = np.ones(x.shape, x.dtype.char) * self.target_activity

        if self.randomized_init:
            self.t = np.ones(x.shape, x.dtype.char) * self.s_init + \
                (topo.pattern.random.UniformRandom( \
                    random_generator=np.random.RandomState(seed=self.seed)) \
                     (xdensity=x.shape[0],ydensity=x.shape[1]) \
                     -0.5)*self.noise_magnitude*2
        else:
            self.t = np.ones(x.shape, x.dtype.char) * self.s_init
        self.y_avg = np.ones(x.shape, x.dtype.char) * self.target_activity


    def _apply_scaling(self,x):
        """Applies the piecewise-linear thresholding operation to the activity."""
        x *= self.t

    def _update_scalefactor(self, prev_t, x, prev_avg, smoothing, learning_rate, target_activity):
        """
        Applies exponential smoothing to the given current activity and previous
        smoothed value following the equations given in the report cited above.

        If plastic is set to False, the running exponential average
        values and thresholds are not updated.
        """
        y_avg = (1.0-smoothing)*x + smoothing*prev_avg
        t = prev_t + learning_rate * (target_activity - y_avg)
        t = np.clip(t, 0, 2)
        return (y_avg, t) if self.plastic else (prev_avg, prev_t)


    def __call__(self,x):
        """Initialises on the first call and then applies homeostasis."""
        if self.first_call: self._initialize(x); self.first_call = False

        if (topo.sim.time() > self._next_update_timestamp):
            self._next_update_timestamp += self.period
            # Using activity matrix and and smoothed activity from *previous* call.
            (self.y_avg, self.t) = self._update_scalefactor(self.t, self._x_prev, self._y_avg_prev,
                                                            self.smoothing, self.learning_rate,
                                                            self.target_activity)
            self._y_avg_prev = self.y_avg   # Copy only if not in continuous mode

        self._apply_scaling(x)            # Apply the threshold only after it is updated
        self._x_prev[...,...] = x[...,...]  # Recording activity for the next periodic update


    def state_push(self):
        self.__current_state_stack.append((copy.copy(self.t),
                                           copy.copy(self.y_avg),
                                           copy.copy(self.first_call),
                                           copy.copy(self._next_update_timestamp),
                                           copy.copy(self._y_avg_prev),
                                           copy.copy(self._x_prev)))
        super(SynapticScaling, self).state_push()

    def state_pop(self):
        (self.t, self.y_avg, self.first_call, self._next_update_timestamp,
        self._y_avg_prev, self._x_prev) = self.__current_state_stack.pop()
        super(SynapticScaling, self).state_pop()
