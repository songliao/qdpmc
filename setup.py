from setuptools import setup, find_packages

install_requires = ["numpy<1.21,>=1.19.2",
                    "joblib>=0.17.0",
                    "scipy>=1.7.1",
                    "numba"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # these rarely change
    name="QdpMC",
    description='A package for pricing derivatives via Monte Carlo',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='derivatives, finance',
    license='Free for non-commercial use',
    author='Yield Chain Developers',
    author_email='dev@yieldchain.com',
    url='http://www.yieldchain.com/',
    # these may change frequently
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.8',
    install_requires=install_requires,
)
