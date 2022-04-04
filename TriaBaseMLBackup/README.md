# TriaBaseML
 A Web 3.0 Database for Massive IoT Workloads
 
 
#### Confiquration
Configuring pybuilder for developing command-line application.

Conda as a package manager is currently not supported by pybuilder.
https://github.com/pybuilder/pybuilder/issues/581


#### Development setup
1. Create conda-env environment
    ```
    $ conda env create -n conda-env -f dev_environment.yml
    $ source activate conda-env
    ```

   If **conda-env** already present
    ```
    $ conda env update -f dev_environment.yml
    $ source activate conda-env
    ```
2. Run main method

    ```
    $ pyb run -P arg1="param1" -P arg2="param1" -P conf="./default_conf"
    ```

###### Tests
1. Run unittests

    ```
    $ conda env update -f unittest/test_environment.yml
    $ pyb run_unit_tests
    ```

###### Packaging
1. Generate pip package

    ```
    $ pyb publish
    ```

2. package location

    target/dist/pybuilder-demo-<version>/dist/pybuilder-demo-<version>.tar.gz


#### Production setup
1. Create conda-env environment
    ```
    $ conda create -n prod-conda-env python=3.6

    $ source activate prod-conda-env
    ```
    Environment name should be similar to the name present in **environment.yml** and **main/scripts/pyb-demo-setup**

2. Install pybuilder-demo package
    ```
    $ pip install <pybuilder-demo-package-path>
    ```

3. Install pybuilder-demo dependencies
    ```
    $ pyb-demo-setup
    ```

4. Run main method
    ```
    $ pyb run
    ```
    OR
    ```
    $ pyb-demo
    ```

    **--conf** has default value set to \<prod-venv-name\>/etc/configs
