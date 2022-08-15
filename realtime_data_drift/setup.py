from setuptools import setup

setup(
    name="realtime_data_drift",
    version="1.0",
    description="Used to Setup data drift in fast api and flask",
    author="Nuwan Withana",
    packages=["realtime_data_drift"],  # same as name
    install_requires=["pandas", "evidently==0.1.48.dev0", "prometheus-client==0.14.1"],
)
