#   -*- coding: utf-8 -*-
from os import path
import sys

project_path = path.dirname(path.abspath(__file__))
sys.path.append(f'{project_path}')

from pybuilder.core import use_plugin, init, Author, task, depends

use_plugin("python.core")
use_plugin("python.distutils")
use_plugin('python.pycharm')
use_plugin("python.install_dependencies")

name = "TriaBaseML"
version = "1.0.1"
summary = "Our example project  for configuring pybuilder"
description = """Our example project  for configuring pybuilder."""

authors = [Author("Panagiotis", "panagiotisdrakatos@gmail.com")]
url = "github url"
license = "GPL License"
default_task = ["clean"]


@init
def initialize(project):
    project.build_depends_on('tensorflow-plot')
    project.build_depends_on('tensorflow')
    # project.build_depends_on('tensorflow-gpu')
    project.build_depends_on('nest-asyncio')
    project.build_depends_on('tensorflow_federated')
    #project.build_depends_on('tensorflow-federated-nightly')
    project.build_depends_on('protobuf')
    project.build_depends_on('grpcio-tools')
    project.build_depends_on('matplotlib')
    project.build_depends_on('scikit-learn')
    project.build_depends_on('pandas')
    project.build_depends_on('jsonpickle')

    project.set_property('dir_source_main_python', '/')
    project.set_property('dir_source_main_scripts', '/src/main/scripts')
    project.set_property("pytest_coverage_break_build_threshold", 90)
    project.set_property("pytest_coverage_html", True)


@depends("install_runtime_dependencies")
def publish():
    pass


@init
def init():
    pass


@depends("install_build_dependencies")
@task(description="Run main")
def run(project):
    from src.main.Examples.ExampleMnist import start
    start()
