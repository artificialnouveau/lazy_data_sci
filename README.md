# Introduction 
This is for the data scientists for Centre for Human Drug Research (CHDR). The code is to be used to standardize the ML practices at CHDR. The package contained within the project has the goal to provide functionality to get insights into the Method Development and Trial@Home datasets, which can be relevant to be used within projects within CHDR.

# Getting Started
In order to get started, one can build the package and try out functionality in combination with their dataset.

# Build and Test
In order to create a distribution of the MethDevDataSci package, one can run the commmand ```python setup.py sdist bdist_wheel```. This will create a built distribution in the wheel format and currently supported by pip.

# Contribute
One can contribute to the project by:
* Testing existing code
* Developing new methods
* Creating/updating documentation

# Deploy to Databricks
1. Make sure the build pipeline on Azure DevOps has run successfully; check how-to-trigger-new-build.md for more details.
2. Install new library on the selected Databricks cluster with PyPI. Use https://chdrpypi:cmfncrd6572j5hi7s3bvhrz3it3kb3rplp5h4hylrtrss6tdpp2a@pkgs.dev.azure.com/chdr/_packaging/chdrpypi/pypi/simple/ as the repository and more==$vnumber as the package name where $vnumber is the version number that matches the last build.
