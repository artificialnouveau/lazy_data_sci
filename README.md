# Introduction 
This is for the beginner lazy data scientists looking for functions to help streamline their data science projects.

# Getting Started
In order to get started, one can build the package and try out functionality in combination with their dataset.

# Package structure
### EDA - Exploratory Data Analysis
    Purpose: Used to understand your data
    Includes: Functions for checking for outliers, missing data, distribution of data, correlated features

### Dataviz - Data visualization
    Purpose: Used to visualize your data
    Includes: Correlation Plots

### Feature Engineering
    Purpose: how to prep your data for your stats modelling
    Functions: Converting categorical data to ordinal
### Feature Selection
    Purpose: how to select which features to use prior to modelling
    Functions: VIF-based feature selections and backwards elimination(using LR)

### Model Validation
    Purpose: how to perform cross validation and model validation
    Functions: custom_GridSearch_nestedCV and print ROC curves


# Build and Test
In order to create a distribution of the lazydatasci package, one can run the commmand ```python setup.py sdist bdist_wheel```. This will create a built distribution in the wheel format and currently supported by pip.

# Contribute
One can contribute to the project by:
* Testing existing code
* Developing new methods
* Creating/updating documentation
