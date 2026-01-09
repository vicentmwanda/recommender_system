from setuptools import setup, find_packages

setup(
    name="my_recommender_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "python-dotenv",
        "matplotlib",
        "Pillow",
        "numba",
        "requests"
    ],
    python_requires='>=3.8',
)
