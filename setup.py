from setuptools import find_packages, setup
from typing import List

hyphon_E_Dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''this function will return list of requirements packages'''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace('/n', '') for req in requirements]

        if hyphon_E_Dot in requirements:
            requirements.remove(hyphon_E_Dot)
    
    return(requirements)


setup(
    name='ml_project_01',
    version='0.0.1',
    author='Rajat',
    author_email='rajatraj9896@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),   

)
