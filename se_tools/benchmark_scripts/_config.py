# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import warnings
from _common import *

LIST_VALUES = ["pyramid", "t_MW_factor", "init_T_WB",
               "occupancy_min_max", "tau_min_max", "sigma_min_max",
               "sigma_min_max_factor", "intrinsics", "T_BC"]

def list_values(obj):
    if not obj: return
    for param, value in obj.__dict__.items():
        if value is None or value == "": continue
        if isinstance(value, list):
            if param in LIST_VALUES:
                if not isinstance(value[0], list):
                    setattr(obj, param, [value])
        else:
            setattr(obj, param, [value])

def delist_value(obj):
    return obj[0]

class Config:
    def __init__(self):
        self.general     = None
        self.map         = None
        self.sensor      = None
        self.voxel_impls = None

    def setup_from_yaml(self, config_yaml):
        if 'general' in config_yaml:
            self.general = General()
            self.general.setup_from_yaml(config_yaml['general'])
        if 'map' in config_yaml:
            self.map = Map()
            self.map.setup_from_yaml(config_yaml['map'])
        if 'sensor' in config_yaml:
            if 'type' not in config_yaml['sensor']:
                raise Exception("Sensor type definition missing, e.g. "
                                "{sensor, pinholecamera, ousterlidar}")
            sensor_type = config_yaml['sensor']['type']
            if sensor_type == 'pinholecamera':
                self.sensor = PinholeCamera()
            elif sensor_type == 'ousterlidar':
                self.sensor = OusterLidar()
            elif sensor_type == 'sensor':
                self.sensor = Sensor()
            else:
                raise Exception("Invalid sensor type used for initialisation,"
                                " e.g. {sensor, pinholecamera, ousterlidar}")
            self.sensor.setup_from_yaml(config_yaml['sensor'])
        if 'voxel_impls' in config_yaml:
            self.voxel_impls = VoxelImpls()
            self.voxel_impls.setup_from_yaml(config_yaml['voxel_impls'])


    def get_yaml_keys_and_value_combs(self):
        general_keys   = []
        general_values = []
        if self.general:
            general_keys, general_values = self.general.get_yaml_keys_and_values()

        map_keys   = []
        map_values = []
        if self.map:
            map_keys, map_values     = self.map.get_yaml_keys_and_values()

        sensor_keys   = []
        sensor_values = []
        if self.sensor:
            sensor_keys, sensor_values  = self.sensor.get_yaml_keys_and_values()

        # Contains a list of yaml keys for each voxel impl used:
        # [[keys_w_voxel_impl_01], [keys_w_voxel_impl_02], [keys_w_voxel_impl_03]]
        keys_list = []

        # Contains a list value all possible value comb for each voxel impl used:
        # [[[value_comb01_w_voxel_impl_01], [value_comb02_w_voxel_impl_01], [value_comb02_w_voxel_impl_01]],
        #  [[value_comb01_w_voxel_impl_02], [value_comb02_w_voxel_impl_02]],
        #  [[value_comb01_w_voxel_impl_03], [value_comb02_w_voxel_impl_03], [value_comb02_w_voxel_impl_03]]]
        value_comb_list = []

        voxel_impl_keys_list, voxel_impl_values_list = self.voxel_impls.get_yaml_keys_and_values()
        for voxel_impl_keys, voxel_impl_values in zip(voxel_impl_keys_list, voxel_impl_values_list):
            keys_list.append(general_keys + map_keys + sensor_keys + voxel_impl_keys)
            value_comb_list.append(list(itertools.product(*(general_values + map_values + sensor_values + voxel_impl_values))))
        return keys_list, value_comb_list

    def copy_missing(self, config):
        for section, value in config.__dict__.items():
            if value:
                if not getattr(self, section):
                    setattr(self, section, value)
                else:
                    getattr(self, section).copy_missing(value)

    def print_all(self):
        self.general.print_all() if self.general else General().print_all()
        self.map.print_all()     if self.map     else Map().print_all()
        self.sensor.print_all()  if self.sensor  else Sensor().print_all()
        self.voxel_impls.print_all()  if self.voxel_impls  else VoxelImpl().print_all()

    def print_set(self):
        if self.general: self.general.print_set()
        if self.map: self.map.print_set()
        if self.voxel_impls:  self.voxel_impls.print_set()

    def print_missing(self):
        self.general.print_missing() if self.general else General().print_all()
        self.map.print_missing()     if self.map     else Map().print_all()
        self.voxel_impls.print_missing()  if self.voxel_impls  else VoxelImpl().print_all()


class General:
    def __init__(self):
        self.enable_ground_truth    = None
        self.enable_render          = None
        self.enable_meshing         = None
        self.enable_structure       = None
        self.max_frame              = None

        # Rates
        self.integration_rate       = None
        self.tracking_rate          = None
        self.rendering_rate         = None
        self.meshing_rate           = None
        self.fps                    = None

        # Other
        self.pyramid                = None
        self.icp_threshold          = None
        self.render_volume_fullsize = None
        self.drop_frames            = None



    def setup_from_yaml(self, general_config_yaml):
        for key, value in general_config_yaml.items():
            if key in ['dataset_name', 'sequences', 'enable_benchmark',
                       'benchmark_path', 'log_path',
                       'output_render_path', 'output_mesh_path', 'output_structure_path']:
                continue
            if value is not None and value != "":
                if not hasattr(self, key):
                    warnings.warn("'General' has no {} parameter".format(key))
                    continue
                setattr(self, key, value)
            list_values(self)

    def get_yaml_keys_and_values(self):
        yaml_keys   = []
        values = []
        for param, value in self.__dict__.items():
            if value:
                yaml_keys.append(['general', param])
                values.append(value)
        return yaml_keys, values

    def print_all(self):
        print("General:")
        for param, value in self.__dict__.items():
                print('     {:23} = {}'.format(param, value))

    def print_set(self):
        print("General:")
        for param, value in self.__dict__.items():
            if value is not None and value != "":
                print('     {:23} = {}'.format(param, value))

    def print_missing(self):
        print("General:")
        for param, value in self.__dict__.items():
            if value is None or value == "":
                print('     {:23} = {}'.format(param, value))

    def copy_missing(self, general):
        for param, value in general.__dict__.items():
            if getattr(self, param) is None and value is not None:
                setattr(self, param, value)
        list_values(self)

class Map:
    def __init__(self):
        self.size        = None
        self.dim         = None
        self.t_MW_factor = None

    def setup_from_yaml(self, map_config_yaml):
        for key, value in map_config_yaml.items():
            if value is not None and value != "":
                if not hasattr(self, key):
                    warnings.warn("'Map' has no {} parameter".format(key))
                    continue
                setattr(self, key, value)
            list_values(self)

    def get_yaml_keys_and_values(self):
        yaml_keys   = []
        values = []
        for param, value in self.__dict__.items():
            if value:
                yaml_keys.append(['map', param])
                values.append(value)
        return yaml_keys, values

    def print_all(self):
        print("Map:")
        for param, value in self.__dict__.items():
                print('     {:23} = {}'.format(param, value))

    def print_set(self):
        print("Map:")
        for param, value in self.__dict__.items():
            if value is not None and value != "":
                print('     {:23} = {}'.format(param, value))

    def print_missing(self):
        print("Map:")
        for param, value in self.__dict__.items():
            if value is None or value == "":
                print('     {:23} = {}'.format(param, value))

    def copy_missing(self, map):
        for param, value in map.__dict__.items():
            if getattr(self, param) is None and value is not None:
                setattr(self, param, value)
        list_values(self)

class Sensor:
    def __init__(self):
        self.type             = None
        self.intrinsics       = None
        self.left_hand_frame  = None
        self.downsampling_factor      = None
        self.bilateral_filter = None
        self.T_BC             = None
        self.init_T_WB        = None
        self.near_plane       = None
        self.far_plane        = None

    def setup_from_yaml(self, sensor_config_yaml):
        for key, value in sensor_config_yaml.items():
            if value is not None and value != "":
                if not hasattr(self, key):
                    warnings.warn("'{}' has no {} parameter".format(delist_value(self.type), key))
                    continue
                setattr(self, key, value)
            list_values(self)

    def get_yaml_keys_and_values(self):
        yaml_keys   = []
        values = []
        for param, value in self.__dict__.items():
            if value is not None and value != "":
                yaml_keys.append(['sensor', param])
                values.append(value)
        return yaml_keys, values

    def print_all(self):
        print("{}:".format(type(self).__name__))
        for param, value in self.__dict__.items():
                print('     {:23} = {}'.format(param, value))

    def print_set(self):
        print("{}:".format(type(self).__name__))
        for param, value in self.__dict__.items():
            if value is not None or value != "":
                print('     {:23} = {}'.format(param, value))

    def print_missing(self):
        print("{}:".format(type(self).__name__))
        for param, value in self.__dict__.items():
            if value is None or value == "":
                print('     {:23} = {}'.format(param, value))

    def copy_missing(self, sensor):
        if not isinstance(self, type(sensor)):
            pass
        else:
            for param, value in sensor.__dict__.items():
                if getattr(self, param) is None and value is not None:
                    setattr(self, param, value)
            list_values(self)

class PinholeCamera(Sensor):
    def __init__(self, intrinsics=None):
        Sensor.__init__(self)
        self.type       = 'pinholecamera'
        self.intrinsics = intrinsics

class OusterLidar(Sensor):
    def __init__(self, intrinsics=None):
        Sensor.__init__(self)
        self.type       = 'ousterlidar'
        self.intrinsics = intrinsics

class VoxelImpls():
    def __init__(self):
        self.list = []

    def get_voxel_impl(self, voxel_impl_type):
        if voxel_impl_type == TSDF().type:
            return TSDF()
        if voxel_impl_type == MultiresTSDF().type:
            return MultiresTSDF()
        if voxel_impl_type == OFusion().type:
            return OFusion()
        if voxel_impl_type == MultiresOFusion().type:
            return MultiresOFusion()
        raise Exception("{} is a invalid voxel impl type.".format(voxel_impl_type))

    def setup_from_yaml(self, voxel_impls_config_yaml):
        for voxel_impl_type in voxel_impls_config_yaml:
            voxel_impl = self.get_voxel_impl(voxel_impl_type)
            for key, value in voxel_impls_config_yaml[voxel_impl_type].items():
                if value is not None and value != "":
                    if not hasattr(voxel_impl, key):
                        warnings.warn("'{}' has no {} parameter".format(voxel_impl_type, key))
                        continue
                    setattr(voxel_impl, key, value)
            list_values(voxel_impl)
            self.list.append(voxel_impl)

    def get_types(self):
        voxel_impl_types = []
        for voxel_impl in self.list:
            voxel_impl_type = voxel_impl.type
            voxel_impl_types.append(voxel_impl_type)
        return voxel_impl_types

    def get_yaml_keys_and_values(self):
        yaml_keys_list  = []
        values_list     = []
        for voxel_impl in self.list:
            yaml_keys = []
            values    = []
            for param, value in voxel_impl.__dict__.items():
                if value is not None and value != "":
                    yaml_keys.append(['voxel_impl', param])
                    values.append(value)
            yaml_keys_list.append(yaml_keys)
            values_list.append(values)
        return yaml_keys_list, values_list

    def add_voxel_impls(self, voxel_impls):
        self.list += voxel_impls

    def copy_missing(self, voxel_impls):
        for voxel_impl_paste in self.list:
            for voxel_impl_copy in voxel_impls.list:
                if type(voxel_impl_paste) is type(voxel_impl_copy):
                    for param, value_copy in voxel_impl_copy.__dict__.items():
                        if getattr(voxel_impl_paste, param) is None and value_copy is not None:
                            setattr(voxel_impl_paste, param, value_copy)
                    list_values(voxel_impl_paste)

    def print_all(self):
        for voxel_impl in self.list:
            voxel_impl.print_all()

    def print_set(self):
        for voxel_impl in self.list:
            voxel_impl.print_set()

    def print_missing(self):
        for voxel_impl in self.list:
            voxel_impl.print_missing()

class VoxelImpl():
    def print_all(self):
        print("{}:".format(type(self).__name__))
        for param, value in self.__dict__.items():
                print('     {:23} = {}'.format(param, value))

    def print_set(self):
        print("{}:".format(type(self).__name__))
        for param, value in self.__dict__.items():
            if value is not None and value != "":
                print('     {:23} = {}'.format(param, value))

    def print_missing(self):
        print("{}:".format(type(self).__name__))
        for param, value in self.__dict__.items():
            if value is None or value == "":
                print('     {:23} = {}'.format(param, value))

    def copy_missing(self, voxel_impl):
        pass

class TSDF(VoxelImpl):
    def __init__(self):
        self.type                    = "tsdf"
        self.mu_factor               = None
        self.max_weight              = None

class MultiresTSDF(VoxelImpl):
    def __init__(self):
        self.type                    = 'multirestsdf'
        self.mu_factor               = None
        self.max_weight              = None

class OFusion(VoxelImpl):
    def __init__(self):
        self.type                    = "ofusion"
        self.surface_boundary        = None
        self.occupancy_min_max       = None
        self.tau                     = None
        self.sigma_min_max_factor    = None
        self.k_sigma                 = None

class MultiresOFusion(VoxelImpl):
    def __init__(self):
        self.type                    = 'multiresofusion'
        self.surface_boundary        = None
        self.occupancy_min_max       = None
        self.max_weight              = None
        self.free_space_integr_scale = None
        self.const_surface_thickness = None
        self.tau_min_max             = None
        self.k_tau                   = None
        self.uncertainty_model       = None
        self.sigma_min_max           = None
        self.k_sigma                 = None
