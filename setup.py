from setuptools import setup, find_packages

setup(
    name="carpet-wizard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.27.0",
        "flask>=2.3.0",
        "requests>=2.31.0",
        "huggingface-hub>=0.20.0",
        "pillow>=10.0.0",
        "python-dotenv>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered custom rug design generator",
    keywords="ai design rug generation",
)zani