from setuptools import setup, find_packages

setup(
    name='NeuralProcesses',
    version='0.1',
    description='Python Distribution Utilities',
    author='Jianqing Qing et al',
    author_email='j.qing@imperial.ac.uk',
    python_requires="==3.10.12",
    install_requires=[
        "jax==0.4.34",
        "flax==0.9.0",
        "einops==0.6.0", 
        "scipy==1.11.4",
        "scikit-learn", 
        'tensorflow==2.11.1', 
        'diffrax==0.4.1', 
        "ml_collections",
        'tensorflow-probability==0.19.0', 
        'matplotlib', 
        'gpjax==0.8.0', 
        'trieste==3.4.0', 
        "pymoo==0.6.1.3", 
        "pandas"
    ],
    packages=find_packages(include=['NeuralProcesses', 'NeuralProcesses.*'])  # Include only the NeuralProcesses package and its subpackages
)