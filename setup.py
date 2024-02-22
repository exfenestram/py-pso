from setuptools import setup, find_packages

setup(
    name='py_pso',
    version='0.1',
    packages=find_packages(include=['py_pso', 'py_pso.*']),
    license='MIT',
    description='The Particle Swarm Optimization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['adamantine', 'numpy'],

    url='https://github.com/exfenestram/adamantine',
    author='Ray Richardson')