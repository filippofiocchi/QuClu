from setuptools import find_packages, setup
setup(
    name='QuClu',
    packages=find_packages(include=['QuClu']),
    version='0.1.0',
    description='A Python library for k-quantile Clustering.',
    author='Filippo Fiocchi',
    license='MIT',
    long_description=open('README.md').read(),
    author_email='filippofiocchi1@gmail.com',
    url='',
    install_requires=['numpy','scipy','pandas'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests'
)
