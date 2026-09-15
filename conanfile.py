from conan import ConanFile
from conan.tools.cmake import CMakeToolchain


class GaffaConan(ConanFile):
    name = "gaffa"
    version = "0.1.0"
    package_type = "application"

    settings = "os", "arch", "compiler", "build_type"

    # Host build/test dependencies are provided by environment.yml. Keeping
    # Conan dependency-free means this step only generates a toolchain file and
    # does not need to download CMake, pybind11, GoogleTest, or their sources.
    def generate(self):
        toolchain = CMakeToolchain(self)
        toolchain.generate()
