import numpy as np

import param

import imagen

import topo
from topo.base.arrayutil import DivideWithConstant
from topo.base.projection import CircularMask
from topo import learningfn, projection, responsefn, sheet, transferfn, optimized
from topo.submodel.earlyvision import EarlyVisionModel
from topo.submodel.scal import EarlyVisionSCAL
from topo.submodel import Model

from . import MultiplyWithConstant

@Model.definition
class ModelSEPI(EarlyVisionSCAL):

    area = param.Number(default=2.0,bounds=(0,None),
        inclusive_bounds=(False,True),doc="""
        Linear size of cortical area to simulate.

        SCAL and other spatially calibrated variants of GCAL require
        cortical areas larger than 1.0x1.0 to avoid strong suppressive
        edge effects.""")

    circular_mask = param.Boolean(default=True, doc="""
        Whether to apply a circular mask to V1 sheets to avoid edge
        effects.""")

    lgn_density = param.Integer(default=16, doc="""
        Density of the LGN sheets""")

    cortex_density=param.Number(default=47.0,bounds=(0,None),
        inclusive_bounds=(False,True),doc="""
        The nominal_density to use for V1.""")

    homeostasis = param.Boolean(default=True, doc="""
        Whether or not the homeostatic adaption should be
        applied in V1""")

    t_init = param.Number(default=0.15, doc="""
        The initial V1 threshold value. This value is static
        in the L and GCL models and adaptive in the AL and
        SCAL models.""")

    target_activity = param.Number(default=0.012, doc="""
        The homeostatic target activity.""")

    pv_exponent = param.Number(default=1.0, doc="""
        Exponent of PV neuron output function.""")

    pv_timeconstant = param.Number(default=1.0, doc="""
        Hysteresis time constant for PV neurons.""")

    num_inputs = param.Integer(default=1, bounds=(1,None))

    #======================#
    # Projection strengths #
    #======================#

    # Afferent Inputs #

    lgn2exc_str=param.Number(default=3.0, doc="""
        Thalamocortical afferent strength""")

    lgn2pv_str=param.Number(default=2, doc="""
        Thalamocortical afferent strength""")

    # Excitatory Projections #

    locexc_strength=param.Number(default=2.5, doc="""
        Local excitatory connection strength""")

    latpv_strength=param.Number(default=2.5, doc="""
        Lateral PV excitatory projection strength""")

    # PV projections #

    pv_strength=param.Number(default=4.5, doc="""
        PV Divisive GC strength """)

    recurrent_pv_strength=param.Number(default=0.8, doc="""
        Recurrent inhibition strength in PV population""")

    #================#
    # Learning rates #
    #================#

    aff_lr=param.Number(default=0.25,bounds=(0.0,None),doc="""
        Learning rate for the afferent projection(s) to V1.""")

    locexc_lr=param.Number(default=0.0, doc="""
        Local excitatory connection strength""")

    latpv_lr=param.Number(default=0.25, doc="""
        Lateral PV excitatory projection strength""")

    pv_lr=param.Number(default=0.25, doc="""
        PV Divisive GC strength """)

    recurrent_pv_lr=param.Number(default=0.25, doc="""
        Recurrent inhibition strength in PV population""")

    #=====================#
    # Spatial Calibration #
    #=====================#

    v1aff_radius = param.Number(default=0.5, bounds=(0, None), doc="""
        Connection field radius of a unit in V1 to units in a LGN
        sheet.""")

    # Excitatory connection profiles #

    local_radius = param.Number(default=0.14, bounds=(0, None), doc="""
        Radius of the local projections within the V1Exc sheet.""")

    local_size = param.Number(default=0.06, bounds=(0, None), doc="""
        Size of the local excitatory connections within V1.""")

    local_pv_size = param.Number(default=0.06, bounds=(0, None), doc="""
        Size of the local excitatory connections within V1.""")

    # PV connection profiles #

    pv_radius = param.Number(default=0.18, bounds=(0, None), doc="""
        Radius of the lateral inhibitory bounds within V1.""")

    pv_size = param.Number(default=0.115, bounds=(0, None), doc="""
        Size of the lateral inhibitory connections within V1.""")

    #=====================#
    # Divisive inhibition #
    #=====================#

    division_constant = param.Number(default=1.0, doc="""
        The constant offset on the denominator for divisive lateral
        inhibition to avoid divide-by-zero errors:

        divide(x,maximum(y,0) + division_constant).""")

    #=========================#
    # Long-range connectivity #
    #=========================#

    laterals = param.Boolean(default=False, doc="""
        Instantiate long-range lateral connections. Expensive!""")

    latexc_strength=param.Number(default=1.0, doc="""
        Lateral excitatory connection strength""")

    latexc_lr=param.Number(default=1.0, doc="""
        Lateral excitatory connection strength""")

    # Excitatory connection profiles #

    lateral_radius = param.Number(default=1.25, bounds=(0, None), doc="""
        Radius of the lateral excitatory bounds within V1Exc.""")

    lateral_size = param.Number(default=2.5, bounds=(0, None), doc="""
        Size of the lateral excitatory connections within V1Exc.""")

    def property_setup(self, properties):
        properties = super(ModelSEPI, self).property_setup(properties)
        "Specify weight initialization, response function, and learning function"

        projection.CFProjection.cf_shape=imagen.Disk(smoothing=0.0)
        projection.CFProjection.response_fn=optimized.CFPRF_DotProduct_cython()
        projection.CFProjection.learning_fn=optimized.CFPLF_Hebbian_cython()
        projection.CFProjection.weights_output_fns=[optimized.CFPOF_DivisiveNormalize_L1_cython()]
        projection.SharedWeightCFProjection.response_fn=optimized.CFPRF_DotProduct_cython()
        sheet.SettlingCFSheet.joint_norm_fn = optimized.compute_joint_norm_totals_cython
        return properties


    def sheet_setup(self):
        sheets = super(ModelSEPI,self).sheet_setup()
        sheets['V1Exc'] = [{}]
        sheets['V1PV'] = [{}]

        return sheets


    @Model.MultiPortSheet
    def V1Exc(self, properties):
        return Model.SettlingCFSheet.params(
            precedence=0.6,
            nominal_density=self.cortex_density,
            mask=CircularMask() if self.circular_mask else SheetMask(),
            nominal_bounds=sheet.BoundingBox(radius=self.area/2.),
            joint_norm_fn=optimized.compute_joint_norm_totals_cython,
            output_fns=[transferfn.misc.HomeostaticResponse(t_init=self.t_init, target_activity=self.target_activity,
                                                            learning_rate=0.01 if self.homeostasis else 0.0)])

    @Model.MultiPortSheet
    def V1PV(self, properties):
        return Model.SettlingCFSheet.params(
            precedence=0.7,
            nominal_density=self.cortex_density,
            nominal_bounds=sheet.BoundingBox(radius=self.area/2.),
            mask=CircularMask() if self.circular_mask else SheetMask(),
            measure_maps=False,
            joint_norm_fn=optimized.compute_joint_norm_totals_cython,
            output_fns=[transferfn.HalfRectifyAndPower(e=self.pv_exponent),
                        transferfn.Hysteresis(time_constant=self.pv_timeconstant)])

    #========================#
    # Projection definitions #
    #========================#

    @Model.matchconditions('V1Exc', 'V1_afferent')
    def V1_afferent_conditions(self, properties):
        return {'level': 'LGN'}


    @Model.matchconditions('V1PV', 'V1_afferent')
    def V1PV_afferent_conditions(self, properties):
        return {'level': 'LGN'}


    @Model.CFProjection
    def V1_afferent(self, src_properties, dest_properties):
        sf_channel = src_properties['SF'] if 'SF' in src_properties else 1
        # Adjust delays so same measurement protocol can be used with and without gain control.
        LGN_V1_delay = 0.05 if self.gain_control else 0.10

        name=''
        if 'eye' in src_properties: name+=src_properties['eye']
        if 'opponent' in src_properties:
            name+=src_properties['opponent']+src_properties['surround']
        name+=('LGN'+src_properties['polarity']+'Afferent')
        if sf_channel>1: name+=('SF'+str(src_properties['SF']))

        excitatory = dest_properties['level'] == 'V1Exc'

        gaussian_size = 2.0 * self.v1aff_radius *self.sf_spacing**(sf_channel-1)
        strength=(self.lgn2exc_str if excitatory else self.lgn2pv_str)*(1.5 if self.gain_control else 1.0)
        weights_generator = imagen.random.GaussianCloud(gaussian_size=gaussian_size)

        return [Model.CFProjection.params(
                delay=LGN_V1_delay+lag,
                dest_port=('Activity','JointNormalize','Afferent'),
                name= name if lag==0 else name+('Lag'+str(lag)),
                learning_rate=self.aff_lr if excitatory else self.aff_lr/6.,
                strength=strength,
                weights_generator=weights_generator,
                nominal_bounds_template=sheet.BoundingBox(radius=
                                            self.v1aff_radius*self.sf_spacing**(sf_channel-1)))
                for lag in self['lags']]


    @Model.matchconditions('V1Exc', 'local_excitatory')
    def local_excitatory_conditions(self, properties):
        return {'level': 'V1Exc'}


    @Model.CFProjection
    def local_excitatory(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.05,
            name='LateralExcitatory_local',
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.local_size),
            strength=self.locexc_strength,
            learning_rate=self.locexc_lr,
            nominal_bounds_template=sheet.BoundingBox(radius=self.local_radius))


    @Model.matchconditions('V1PV', 'lateral_pv')
    def lateral_pv_conditions(self, properties):
        return {'level': 'V1Exc'}


    @Model.CFProjection
    def lateral_pv(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.05,
            name='LateralPV',
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.local_pv_size),
            strength=self.latpv_strength,
            learning_rate=self.latpv_lr,
            nominal_bounds_template=sheet.BoundingBox(radius=self.local_radius))


    @Model.matchconditions('V1Exc', 'pv_inhibition')
    def pv_inhibition_conditions(self, properties):
        return {'level': 'V1PV'}


    @Model.CFProjection
    def pv_inhibition(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.05,
            name='PVInhibition',
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.pv_size),
            strength=self.pv_strength,
            learning_rate=self.pv_lr,
            activity_group=(0.8,DivideWithConstant(c=1.0)),
            nominal_bounds_template=sheet.BoundingBox(radius=self.pv_radius))


    @Model.matchconditions('V1PV', 'recurrent_pv')
    def recurrent_pv_conditions(self, properties):
        return {'level': 'V1PV'}


    @Model.CFProjection
    def recurrent_pv(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.05,
            name='RecurrentPV',
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.pv_size),
            strength=self.recurrent_pv_strength,
            learning_rate=self.recurrent_pv_lr,
            activity_group=(0.8,DivideWithConstant(c=1.0)),
            nominal_bounds_template=sheet.BoundingBox(radius=self.pv_radius))


    @Model.matchconditions('V1Exc', 'lateral_excitatory')
    def lateral_excitatory_conditions(self, properties):
        return {'level': 'V1Exc'} if self.laterals else {'level': None}


    @Model.CFProjection
    def lateral_excitatory(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.1,
            name='LateralExcitatory',
            activity_group=(0.9, MultiplyWithConstant()),
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.lateral_size),
            strength=self.latexc_strength,
            learning_rate=self.latexc_lr,
            nominal_bounds_template=sheet.BoundingBox(radius=self.lateral_radius))



@Model.definition
class ModelLESPI(ModelSEPI):
    """
    Long-range excitation, Somatostatin (Sst) and Parvalbumin (Pv)
    inhibition model (LESPI). The model reduces to the simpler,
    Short-range excitation, Parvalbumin inhibition model (SEPI),
    when the lateral flag is disabled. The SEPI model accurately
    captures the anatomy of layer 4 in V1 and exhibits robust
    and stable map development in a dual population model. The
    long-range interactions in the full LESPI model enable numerous
    surround modulation effects.
    """

    sst_exponent = param.Number(default=1.5, doc="""
        Exponent of Sst neuron activation function.""")

    sst_timeconstant = param.Number(default=0.2, doc="""
        Hysteresis time constant for Sst neurons.""")

    #======================#
    # Projection strengths #
    #======================#

    # Excitatory Projections #

    loc_sst_strength=param.Number(default=1.0, doc="""
        Lateral SOM excitatory projection strength""")

    lat_sst_strength=param.Number(default=2.0, doc="""
        Lateral SOM excitatory projection strength""")

    # Sst projections #

    disinhibition_strength=param.Number(default=0.0, doc="""
        Disinhibitory SOM-PV strength.""")

    sst_inhibition_strength=param.Number(default=-0.1, doc="""
        SOM Inhibitory strength""")

    #================#
    # Learning rates #
    #================#

    loc_sst_lr=param.Number(default=0, doc="""
        Lateral SOM excitatory projection strength""")

    lat_sst_lr=param.Number(default=3.0, doc="""
        Lateral SOM excitatory projection strength""")

    disinhibition_lr=param.Number(default=0.1, doc="""
        Disinhibitory SOM-PV strength.""")

    sst_inhibition_lr=param.Number(default=0.1, doc="""
        SOM Inhibitory strength""")

    def sheet_setup(self):
        sheets = super(ModelLESPI,self).sheet_setup()
        sheets['V1Sst'] = [{}]

        return sheets

    @Model.MultiPortSheet
    def V1Sst(self, properties):
        return Model.SettlingCFSheet.params(
            precedence=0.8,
            mask=CircularMask() if self.circular_mask else SheetMask(),
            nominal_density=self.cortex_density,
            nominal_bounds=sheet.BoundingBox(radius=self.area/2.),
            joint_norm_fn=optimized.compute_joint_norm_totals_cython,
            output_fns=[transferfn.HalfRectifyAndPower(e=self.sst_exponent),
                        transferfn.Hysteresis(time_constant=self.sst_timeconstant)])


    @Model.matchconditions('V1Sst', 'local_sst')
    def local_sst_conditions(self, properties):
        return {'level': 'V1Exc'}


    @Model.CFProjection
    def local_sst(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.05,
            name='LocalSst',
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.local_size),
            strength=self.loc_sst_strength,
            learning_rate=self.loc_sst_lr,
            nominal_bounds_template=sheet.BoundingBox(radius=self.local_radius))


    @Model.matchconditions('V1Sst', 'lateral_sst')
    def lateral_sst_conditions(self, properties):
        return {'level': 'V1Exc'}


    @Model.CFProjection
    def lateral_sst(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.1,
            name='LateralSst',
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.lateral_size),
            strength=self.lat_sst_strength,
            learning_rate=self.lat_sst_lr,
            nominal_bounds_template=sheet.BoundingBox(radius=self.lateral_radius))


    @Model.matchconditions('V1Exc', 'sst_inhibition')
    def sst_inhibition_conditions(self, properties):
        return {'level': 'V1Sst'}


    @Model.CFProjection
    def sst_inhibition(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.05,
            name='SstInhibition',
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.local_size),
            strength=self.sst_inhibition_strength,
            learning_rate=self.sst_inhibition_lr,
            activity_group=(0.9, MultiplyWithConstant()),
            nominal_bounds_template=sheet.BoundingBox(radius=self.local_radius))


    @Model.matchconditions('V1PV', 'sst_pv_inhibition')
    def sst_pv_inhibition_conditions(self, properties):
        return {'level': 'V1Sst'} if self.disinhibition_strength else {'level': None}


    @Model.CFProjection
    def sst_pv_inhibition(self, src_properties, dest_properties):
        return Model.CFProjection.params(
            delay=0.05,
            name='Disinhibition',
            weights_generator=imagen.Gaussian(aspect_ratio=1.0, size=self.local_size),
            strength=self.disinhibition_strength,
            learning_rate=self.disinhibition_lr,
            nominal_bounds_template=sheet.BoundingBox(radius=self.local_radius))
