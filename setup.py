from setuptools import setup
with open('requirements.txt', 'r') as f:
    reqs = f.read()
#with open('LICENSE', 'r') as f:
#    legal = f.read()
with open('README.md', 'r', encoding="utf-8") as f:
    readme = f.read()
setup(
    name='reading4listeners',
    version='0.0.3.post1',
    packages=['r4l'],
    url='https://github.com/CypherousSkies/reading-for-listeners',
    project_urls={
        "Bug Tracker": "https://github.com/CypherousSkies/reading-for-listeners/issues",
    },
    #license=legal,
    license='AGPL-3',
    author='CypherousSkies',
    author_email="5472563+CypherousSkies@users.noreply.github.com",
    description='A deep-learning powered application which turns pdfs into audio files. Featuring ocr improvement and tts with inflection!',
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=reqs,
    entry_points={"console_scripts": ["r4l = r4l.bin.cli:main"], }
)
