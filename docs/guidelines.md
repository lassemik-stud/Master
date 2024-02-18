
# Best Practice
    - Use proper naming conventions for variables, functions, methods, and more.
    - Variables, functions, methods, packages, modules: this_is_a_variable
    - Classes and exceptions: CapWords
    - Protected methods and internal functions: _single_leading_underscore
    - Private methods: __double_leading_underscore
    - Constants: CAPS_WITH_UNDERSCORES
    - Use 4 spaces for indentation. For more conventions, refer to PEP8.

# create requirement file
    - pip3 install pipreqs
    - pipreqs --ignore /examples </path/to/project>

# install from requirenments file
    - pip3 install -r requirenments.txt

# use virtualenvirenments for each project
## INSTALL virtualenv
pip3 install virtualenv

## CREATE
virtualenv <PATH>

##  ACTIVATE
source <PATH>/bin/activate

## DEACTIVATE
deactivate