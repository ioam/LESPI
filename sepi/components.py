import numpy as np

import param

from imagen import Composite, Gaussian, Disk
from imagen.random import UniformRandom

from topo.base.simulation import EPConnectionEvent
from topo.sheet import JointNormalizingCFSheet_Continuous

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
