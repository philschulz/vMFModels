from setuptools import setup, find_packages
import os

ROOT = os.path(__file__)

kwargs = dict(
    name="vmf_models",
    description="NLP models using von-Mises-Fisher likelihoods on word vectors.",
    url="https://github.com/philschulz/vMFModels",
    author="Philip Schulz",
    author_email="P.Schulz@uva.nl",
    license="Apache Licence 2.0",
    python_requires=">=3.5",
    packages= find_packages(),
    install_requires=["gensim=>3", "scipy", "numpy"],
    entry_points={
        "console_scripts" : [
            "vmf-aligner = vmf_alignment.vmf_aligner::main"
        ]
    }
)

setup(**kwargs)