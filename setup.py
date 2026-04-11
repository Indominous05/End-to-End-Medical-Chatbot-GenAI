from setuptools import find_packages, setup


def read_requirements(path="requirements.txt"):
    requirements = []
    with open(path, encoding="utf-8") as req_file:
        for line in req_file:
            item = line.strip()
            if not item or item.startswith("#") or item.startswith("-e"):
                continue
            requirements.append(item)
    return requirements

setup(
    name="GenAI Project",
    version="0.1.0",
    author="Parth Patil",
    author_email="parthpatil.051104@gmail.com",
    packages=find_packages(),
    install_requires=read_requirements(),
)