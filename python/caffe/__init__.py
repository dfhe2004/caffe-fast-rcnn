from .pycaffe import Net, SGDSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list, set_random_seed #, glog_init
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto
