import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="lazydatasci",
    version="0.0.1",
    author="Aritifical Nouveau",
    author_email="artificalnouveau@gmail.com",
    description="Package of ML shortcut for Lazy Data Scientists",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=required,
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

#for more information on building a library, see: https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f
