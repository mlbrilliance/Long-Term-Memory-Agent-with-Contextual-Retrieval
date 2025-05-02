from setuptools import find_packages, setup

setup(
    name="ltm_agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain_anthropic",
        "dotenv",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "ltm-agent=ltm_agent.main:run_main",
        ],
    },
)
