from setuptools import setup

setup(
    name='platosim-contamination',
    version='0.1',
    description='contamination for platosim',
    packages=['platosim_contamination'],
    author='Digonto Rahman, RagnarVdB',
    install_requires=[
        'numpy',
        'scipy'
    ],
    zip_safe=False
)
