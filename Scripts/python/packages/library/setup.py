from setuptools import find_packages, setup

setup(
    name="bridge_data_tools",  # Choose a unique name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Dependencies (if any)
    author="Your Name",
    author_email="your_email@example.com",
    description="A tools library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_library",  # Your repository
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
