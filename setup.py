from setuptools import setup, find_packages

setup(
    name="deepverify",
    version="1.0.0",
    author="B. Anadhyanth",
    description="Digital Image Forgery Detection Using Deep Learning",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "opencv-python",
        "tensorflow",
        "keras",
        "scikit-learn",
        "pillow",
        "joblib"
    ]
)