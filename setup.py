from setuptools import setup

setup(
    name='platocon',
    version='0.1',
    description='contamination for platosim',
    packages=['contamination', 'filters'],
    author='Digonto Rahman, RagnarVdB',
    install_requires=[
        'numpy',
        'scipy'
        'pytransit'
        'astropy',
        'pyAstronomy'
    ],
    zip_safe=False
)
