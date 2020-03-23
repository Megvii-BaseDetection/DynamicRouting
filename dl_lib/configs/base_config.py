import os
import pprint
import re
from ast import literal_eval

from colorama import Back, Fore
from easydict import EasyDict as edict

from dl_lib.utils.config_helper import (_assert_with_logging,
                                        _check_and_coerce_cfg_value_type,
                                        diff_dict, find_key, highlight, update)

_config_dict = dict(
    MODEL=dict(
        DEVICE="cuda",
        WEIGHTS="",
        PIXEL_MEAN=[103.530, 116.280, 123.675],  # mean value from ImageNet
        # When using pre-trained models in Detectron1 or any MSRA models,
        # std has been absorbed into its conv1 weights, so the std needs to be
        # set 1. Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        PIXEL_STD=[1.0, 1.0, 1.0],
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(800, ),
                      max_size=1333,
                      sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800,
                      max_size=1333,
                      sample_style="choice")),
            ],
        ),
        CROP=dict(
            ENABLED=False,
            TYPE="relative_range",
            SIZE=[0.9, 0.9],
        ),
        FORMAT="BGR",
        MASK_FORMAT="polygon",
    ),
    DATASETS=dict(
        TRAIN=(),
        PROPOSAL_FILES_TRAIN=(),
        PRECOMPUTED_PROPOSAL_TOPK_TRAIN=2000,
        TEST=(),
        PROPOSAL_FILES_TEST=(),
        PRECOMPUTED_PROPOSAL_TOPK_TEST=1000,
    ),
    DATALOADER=dict(
        NUM_WORKERS=2,
        ASPECT_RATIO_GROUPING=True,
        SAMPLER_TRAIN="TrainingSampler",
        REPEAT_THRESHOLD=0.0,
        FILTER_EMPTY_ANNOTATIONS=True,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_ITER=40000,
            STEPS=(30000, ),
            WARMUP_FACTOR=1.0 / 1000,
            WARMUP_ITERS=1000,
            WARMUP_METHOD="linear",
            GAMMA=0.1,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.001,
            BIAS_LR_FACTOR=1.0,
            WEIGHT_DECAY=0.0001,
            WEIGHT_DECAY_NORM=0.0,
            WEIGHT_DECAY_BIAS=0.0001,
            SUBDIVISION=1,
            MOMENTUM=0.9,
        ),
        CHECKPOINT_PERIOD=5000,
        IMS_PER_BATCH=16,
    ),
    TEST=dict(
        EXPECTED_RESULTS=[],
        EVAL_PERIOD=0,
        KEYPOINT_OKS_SIGMAS=[],
        DETECTIONS_PER_IMAGE=100,
        AUG=dict(
            ENABLED=False,
            MIN_SIZES=(400, 500, 600, 700, 800, 900, 1000, 1100, 1200),
            MAX_SIZE=4000,
            FLIP=True,
        ),
        PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
    ),
    OUTPUT_DIR="./output",
    SEED=-1,
    CUDNN_BENCHMARK=False,
    VIS_PERIOD=0,
    GLOBAL=dict(HACK=1.0),
)


class BaseConfig(object):
    def __init__(self):
        self._config_dict = {}
        self._register_configuration(_config_dict)

    def _register_configuration(self, config):
        """
        Register all key and values of config as BaseConfig's attributes.

        Args:
            config (dict): custom config dict
        """
        self._config_dict = update(self._config_dict, config)
        self.check_update()
        for k, v in self._config_dict.items():
            if hasattr(self, k):
                if isinstance(v, dict):
                    setattr(self, k, update(getattr(self, k), v))
                else:
                    setattr(self, k, v)
            elif isinstance(v, dict):
                setattr(self, k, edict(v))
            else:
                setattr(self, k, v)

    def check_update(self):
        """
        Backward compatibility of old config's Augmentation pipelines and Optimizer configs;
        Convert old format into new ones.
        """
        # Old Augmentation config
        input_cfg = self._config_dict["INPUT"]
        train_input_pop_list = [("MIN_SIZE_TRAIN", "short_edge_length"),
                                ("MIN_SIZE_TRAIN_SAMPLING", "sample_style"),
                                ("MAX_SIZE_TRAIN", "max_size")]
        test_input_pop_list = [
            ("MIN_SIZE_TEST", "short_edge_length"),
            ("MAX_SIZE_TEST", "max_size"),
        ]
        train_contains = [k for k in train_input_pop_list if k[0] in input_cfg]
        test_contains = [k for k in test_input_pop_list if k[0] in input_cfg]
        train_transforms = [t[0] for t in input_cfg["AUG"]["TRAIN_PIPELINES"]]
        test_transforms = [t[0] for t in input_cfg["AUG"]["TEST_PIPELINES"]]
        for k in train_contains:
            if "ResizeShortestEdge" in train_transforms:
                idx = idx = train_transforms.index("ResizeShortestEdge")
                input_cfg["AUG"]["TRAIN_PIPELINES"][idx][1][k[1]] = input_cfg[
                    k[0]]

        for k in test_contains:
            if "ResizeShortestEdge" in test_transforms:
                idx = test_transforms.index("ResizeShortestEdge")
                input_cfg["AUG"]["TEST_PIPELINES"][idx][1][k[1]] = input_cfg[
                    k[0]]

        for elem in train_contains + test_contains:
            self._config_dict["INPUT"].pop(elem[0])

        # Old SOLVER format
        solver_cfg = self._config_dict["SOLVER"]
        candidates = list(solver_cfg.keys())
        lr_pop_list = [
            "LR_SCHEDULER_NAME", "MAX_ITER", "STEPS", "WARMUP_FACTOR",
            "WARMUP_ITERS", "WARMUP_METHOD", "GAMMA"
        ]
        optim_pop_list = [
            "BASE_LR", "BIAS_LR_FACTOR", "WEIGHT_DECAY", "WEIGHT_DECAY_NORM",
            "WEIGHT_DECAY_BIAS", "MOMENTUM"
        ]
        if any(item in candidates for item in lr_pop_list + optim_pop_list):
            if "LR_SCHEDULER" in solver_cfg and "OPTIMIZER" in solver_cfg:
                for k in candidates:
                    if k in lr_pop_list:
                        solver_cfg["LR_SCHEDULER"][k] = solver_cfg[k]
                    elif k in optim_pop_list:
                        solver_cfg["OPTIMIZER"][k] = solver_cfg[k]
            else:
                solver_cfg["LR_SCHEDULER"] = dict(
                    NAME=solver_cfg.get("LR_SCHEDULER_NAME",
                                        "WarmupMultiStepLR"),
                    MAX_ITER=solver_cfg.get("MAX_ITER", 30000),
                    STEPS=solver_cfg.get("STEPS", (20000, )),
                    WARMUP_FACTOR=solver_cfg.get("WARMUP_FACTOR", 1.0 / 1000),
                    WARMUP_ITERS=solver_cfg.get("WARMUP_ITERS", 1000),
                    WARMUP_METHOD=solver_cfg.get("WARMUP_METHOD", "linear"),
                    GAMMA=solver_cfg.get("GAMMA", 0.1),
                )
                solver_cfg["OPTIMIZER"] = dict(
                    NAME="SGD",
                    BASE_LR=solver_cfg.get("BASE_LR", 0.001),
                    BIAS_LR_FACTOR=solver_cfg.get("BIAS_LR_FACTOR", 1.0),
                    WEIGHT_DECAY=solver_cfg.get("WEIGHT_DECAY", 0.0001),
                    WEIGHT_DECAY_NORM=solver_cfg.get("WEIGHT_DECAY_NORM", 0.0),
                    WEIGHT_DECAY_BIAS=solver_cfg.get("WEIGHT_DECAY_BIAS",
                                                     0.0001),
                    MOMENTUM=solver_cfg.get("MOMENTUM", 0.9),
                )

            for k in candidates:
                if k in lr_pop_list + optim_pop_list:
                    solver_cfg.pop(k)

    def merge_from_list(self, cfg_list):
        """
        Merge config (keys, values) in a list (e.g., from command line) into
        this CfgNode.

        Args:
            cfg_list (list): cfg_list must be divided exactly.
            For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".
            format(cfg_list),
        )
        # root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(hasattr(d, subkey),
                                     "Non-existent key: {}".format(full_key))
                d = getattr(d, subkey)
            subkey = key_list[-1]
            _assert_with_logging(hasattr(d, subkey),
                                 "Non-existent key: {}".format(full_key))
            value = self._decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(value, getattr(d, subkey),
                                                     subkey, full_key)
            setattr(d, subkey, value)

    def link_log(self, link_name="log"):
        """
        create a softlink to output dir.

        Args:
            link_name(str): name of softlink
        """
        if os.path.islink(
                link_name) and os.readlink(link_name) != self.OUTPUT_DIR:
            os.system("rm " + link_name)
        if not os.path.exists(link_name):
            cmd = "ln -s {} {}".format(self.OUTPUT_DIR, link_name)
            os.system(cmd)

    @classmethod
    def _decode_cfg_value(cls, value):
        """
        Decodes a raw config value (e.g., from a yaml config files or command
        line argument) into a Python object.
        If the value is a dict, it will be interpreted as a new CfgNode.
        If the value is a str, it will be evaluated as literals.
        Otherwise it is returned as-is.

        Args:
            value (dict or str): value to be decoded
        """
        # Configs parsed from raw yaml will contain dictionary keys that need to be
        # converted to CfgNode objects
        if isinstance(value, dict):
            return cls(value)
        # All remaining processing is only applied to strings
        if not isinstance(value, str):
            return value
        # Try to interpret `value` as a:
        #   string, number, tuple, list, dict, boolean, or None
        try:
            value = literal_eval(value)
        # The following two excepts allow v to pass through when it represents a
        # string.
        #
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure, like
        # a list. In the case that v represents a string, what we got back from the
        # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
        # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
        # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
        # will raise a SyntaxError.
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value

    def _get_param_list(self) -> list:
        """
        get parameter(attribute) list of current config object

        Returns:
            list: attribute list
        """
        param_list = [
            name for name in self.__dir__()
            if not name.startswith("__") and not callable(getattr(self, name))
        ]
        return param_list

    def diff(self, cfg) -> dict:
        """
        diff given config with current config object

        Args:
            cfg(BaseConfig): given config, could be any subclass of BaseConfig

        Returns:
            dict: contains all diff pair
        """
        assert isinstance(cfg,
                          BaseConfig), "config is not a subclass of BaseConfig"
        diff_result = {}
        self_param_list = self._get_param_list()
        conf_param_list = cfg._get_param_list()
        for param in self_param_list:
            if param not in conf_param_list:
                diff_result[param] = getattr(self, param)
            else:
                self_val, conf_val = getattr(self, param), getattr(cfg, param)
                if self_val != conf_val:
                    if isinstance(self_val, dict):
                        diff_result[param] = diff_dict(self_val, conf_val)
                    else:
                        diff_result[param] = self_val
        return diff_result

    def show_diff(self, cfg):
        """
        print diff between current config object and given config object
        """
        return pprint.pformat(edict(self.diff(cfg)))

    def find(self,
             key: str,
             show=True,
             color=Fore.BLACK + Back.YELLOW) -> dict:
        """
        find a given key and its value in config

        Args:
            key (str): the string you want to find
            show (bool): if show is True, print find result; or return the find result
            color (str): color of `key`, default color is black(foreground) yellow(background)

        Returns:
            dict: if  show is False, return dict that contains all find result

        Example::

            >>> from config import config        # suppose you are in your training dir
            >>> config.find("weights")
        """
        key = key.upper()
        find_result = {}
        param_list = self._get_param_list()
        for param in param_list:
            param_value = getattr(self, param)
            if re.search(key, param):
                find_result[param] = param_value
            elif isinstance(param_value, dict):
                find_res = find_key(param_value, key)
                if find_res:
                    find_result[param] = find_res
        if not show:
            return find_result
        else:
            pformat_str = pprint.pformat(edict(find_result))
            print(highlight(key, pformat_str, color))

    def __repr__(self):
        param_dict = edict(
            {param: getattr(self, param)
             for param in self._get_param_list()})
        return pprint.pformat(param_dict)


config = BaseConfig()
