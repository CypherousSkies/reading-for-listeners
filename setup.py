from setuptools import setup
with open('requirements.txt', 'r') as f:
    reqs = f.read()
with open('LICENSE', 'r') as f:
    legal = f.read()
with open('README.md', 'r') as f:
    readme = f.read()
setup(
    name='pdf_to_speech',
    version='0.0.1',
    packages=['pdf_to_speech'],
    url='https://github.com/CypherousSkies/pdf-to-speech',
    license=legal,
    author='CypherousSkies',
    author_email="5472563+CypherousSkies@users.noreply.github.com",
    description='A deep-learning powered application which turns pdfs into audio files. Featuring ocr improvement and tts with inflection!',
    long_description=readme,
    install_requires=reqs,
    entry_points={"console_scripts": ["p2s = pdf_to_speech.bin.cli:main"], }
)
