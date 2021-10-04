from setuptools import setup, find_packages
with open('requirements.txt', 'r') as f:
    reqs = f.read()
with open('LICENSE', 'r') as f:
    legal = f.read()
setup(
    name='pdf2speech',
    version='0.0.1',
    packages=['bin', 'util'],
    package_dir={'': 'pdf2speech'},
    url='https://github.com/CypherousSkies/pdf-to-speech',
    license=legal,
    author='CypherousSkies',
    author_email="5472563+CypherousSkies@users.noreply.github.com",
    description='A deep-learning powered application which turns pdfs into audio files. Featuring ocr improvement and tts with inflection!',
    install_requires=reqs,
    entry_points={"console_scripts": ["p2s = pdf2speech.bin.cli:main"], }
)
