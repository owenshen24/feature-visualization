import typing

class Config(typing.NamedTuple):
    learning_rate: float
    n_iterations: int
    use_octaves: bool
    n_octaves: int
    octave_scale: float

config_list = {}
config_list["default"] = Config(
    0.25,
    10,
    True,
    3,
    1.4
)
config_list["owen"] = Config(
    0.05,
    10,
    False,
    0,
    0
)