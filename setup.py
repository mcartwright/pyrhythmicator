from distutils.core import setup

setup(
    name='pyrhythmicator',
    version='0.1.0',
    packages=['pyrhythmicator'],
    url='https://github.com/mcartwright/pyrhythmicator',
    license='MIT',
    author='Mark Cartwright',
    author_email='mark.cartwright@nyu.edu',
    description="A Python implementation of Sioros and Guedes's Rhythmicator for rhythm generation and synthesis",
    long_description=open('README.md').read(),
    install_requires=[
        "jams == 0.2.2",
        "librosa == 0.5.1",
        "pandas == 0.20.1",
        "soundfile == 0.9.0",
        "numpy == 1.12.1",
        "scipy == 0.19.0",
    ],
)
