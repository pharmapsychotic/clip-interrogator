import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="clip-interrogator",
    version="0.3.1",
    license='MIT',
    author='pharmapsychotic',
    author_email='me@pharmapsychotic.com',
    url='https://github.com/pharmapsychotic/clip-interrogator',
    description="Generate a prompt from an image",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['blip','clip','prompt-engineering','stable-diffusion','text-to-image'],
)
