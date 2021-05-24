from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tprmp',
    version='0.1.0',
    description='',
    long_description=readme,
    author='An Thai Le',
    author_email='an.thai.le97@gmail.com',
    url='https://github.com/anindex/tp-rmp',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    package_data={'lgp': ['data/scenarios/*.pddl']},
)
