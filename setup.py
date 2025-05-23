from setuptools import setup, find_packages

setup(
    name='model_scaling',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'ipykernel',
        'seaborn',
        'jaxopt',
        'ruff',
        'ml_collections',
        'tpu_pod_commander',
    ],
)
