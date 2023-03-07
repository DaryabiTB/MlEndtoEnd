from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e."
def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return list of requirements.
    '''

    requiremnets = []
    with open(file_path) as file_obj:
        requiremnets = file_obj.readlines()
        [ req.replace("\n" , "") for req in requiremnets ]

        if HYPHEN_E_DOT in requiremnets:
            requiremnets.remove(HYPHEN_E_DOT)
    return requiremnets

setup(
    name = "MLProject",
    version="0.0.1",
    author="Mohammad Talib Daryabi",
    author_email="talibdaryabi@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)

