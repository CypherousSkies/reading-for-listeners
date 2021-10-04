from setuptools import setup, find_packages

with open('README.md','r') as f:
    readme = f.read()
with open('LICENSE','r') as f:
    license = f.read()
with open('requirements.txt','r') as f:
    reqs = f.read()

setup(
        name='pdf-to-speech',
        version='0.0.1',
        description='A deep-learning powered accessibility application which turns pdfs into audio files. Featuring ocr improvement and tts with inflection!',
        long_description=readme,
        author='CypherousSkies',
        author_email='5472563+CypherousSkies@users.noreply.github.com',
        url='https://github.com/CypherousSkies/pdf-to-speech',
        license=license,
        packages=find_packages(exclude=('tests','docs')),
        install_requires=reqs#,
        #entry_points={"console_scripts":"pdf2speech=pdf-to-speech.bin.cli:main"}
)
