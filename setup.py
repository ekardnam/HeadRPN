
from head_rpn import VERSION
from setuptools import setup, find_packages

PACKAGE_NAME = 'head_rpn'

if __name__ == '__main__':
    desc = ''
    with open('README.md', 'r') as f:
        desc = f.read()
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        packages=find_packages(),
        install_requires=['tensorflow>=2.3.1'], # actually idk ¯\_( )_/¯
        python_requires='>=3.6.3',
        scripts=[],
        description='Make uncertainty propagation no longer a hassle',
        long_description=desc,
        long_description_content_type='text/markdown'
    )
