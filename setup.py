from setuptools import setup, find_packages

setup(
    name='chromperiod',
    version='1.0.1',
    description='Detecting periodic chromatin organization from accessibility data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Wolf Henning Gebhardt',
    author_email='w.gebhardt@protonmail.com',
    url='https://github.com/WolfGebhardt/chromperiod',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'matplotlib>=3.5',
        'pandas>=1.3',
    ],
    extras_require={
        'dev': ['pytest>=7.0', 'seaborn>=0.11'],
        'full': ['seaborn>=0.11', 'pyBigWig'],
    },
    python_requires='>=3.8',
    license='chromperiod Research and Non-Commercial Use License v1.0 (see LICENSE)',
    license_files=('LICENSE', 'PATENT_NOTICE.md'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords=[
        'chromatin', 'wavelet', 'CWT', 'DNase-seq', 'ATAC-seq',
        'A/B compartments', 'Hi-C', 'periodicity', 'genomics',
        'nuclear organization', 'chromosome architecture',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/WolfGebhardt/chromperiod/issues',
        'Source': 'https://github.com/WolfGebhardt/chromperiod',
        'Manuscript': 'https://github.com/WolfGebhardt/chromperiod#citation',
    },
)
