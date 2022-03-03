from setuptools import setup, find_packages

setup(
    name="sinabro",
    version="0.0.1",
    description="sinabro tools",
    url="https://github.com/ccdc-git/sinabro.git",
    author="ccdc",
    author_email="titieiti.com@gmail.com",
    license="js",
    py_modules=["sinabro"],
    packages=find_packages(),
    zip_safe=False,
    install_requires=["funcy", "sklearn", "opencv-python", "numpy", "imutils", "pycocotools"],
)
