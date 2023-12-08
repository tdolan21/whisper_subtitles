from setuptools import setup, find_packages

setup(
    name='whisper_subtitles',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'librosa',
        'soundfile',
        'torch',
        'torchaudio',
        'torchvision',        
        'transformers',
        'pyannote.audio',
        'colorama',
        'optimum',
        'moviepy',
        'python-dotenv',
        'python-multipart',
        'lightning',


        
        # ... Add any other dependencies
    ],
    entry_points={
        'console_scripts': [
            'whisper-subtitles=whisper_subtitles.main:main',
        ],
    },
)


