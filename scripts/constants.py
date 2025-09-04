from collections import defaultdict

DEFAULT_MAX_STEPS_GLOBAL = int(2e6)

DEFAULT_MAX_STEPS = defaultdict(
    lambda: DEFAULT_MAX_STEPS_GLOBAL,
    {
        'h1-crawl-v0': int(1e6),
        'h1-stand-v0': int(1e6),
        'h1-walk-v0': int(1.5e6),
        'h1-pole-v0': int(1.5e6),
        'h1-reach-v0': int(2e6),
        'h1-run-v0': int(2e6),
        'dog-stand': int(1e6),
        'fish-swim': int(1.25e6),
        'dog-walk': int(1.25e6),
        'dog-trot': int(2e6),
        'humanoid-stand': int(1e6),
        'humanoid-walk': int(2e6),
    },
)

BASE_THRESHOLDS = {
    'h1-crawl-v0': 450,
    'h1-pole-v0': 300,
    'h1-stand-v0': 200,
    'humanoid-stand': 300,
    'acrobot-swingup': 200,
    'cheetah-run': 450,
    'finger-turn': 400,
    'fish-swim': 200,
    'hopper-hop': 150,
    'quadruped-run': 200,
    'walker-run': 350,
    'pendulum-swingup': 0,
    'dog-run': 100,
    'dog-stand': 100,
    'dog-trot': 100,
    'dog-walk': 200,
    'humanoid-run': 75,
    'humanoid-walk': 200,
}

THRESHOLD_FILTER = {
    'dmc': {'threshold_filter': 'model_size_desc != "xs"'},
    'dog_humanoid': {'threshold_filter': 'model_size_desc != "xs" and utd > 1'},
}
