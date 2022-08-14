from setuptools import find_packages, setup
setup(
    name='QuClu',
    packages=find_packages(include=['QuClu']),
    version='0.0.1',
    description='A Python library for k-quantile Clustering.',
    author='Filippo Fiocchi',
    license='MIT',
    long_description=open('README.md').read(),
    author_email='filippofiocchi1@gmail.com',
    url='https://github.com/filippofiocchi/QuClu',
    download_url = 'https://github.com/filippofiocchi/QuClu/archive/refs/tags/0.0.1.tar.gz',
    install_requires=['numpy','scipy','pandas'],
    setup_requires=['pytest-runner']
)
