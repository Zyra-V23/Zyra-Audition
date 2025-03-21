#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Get version from package
with open('conversation_analytics/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip("'").strip('"')
            break

# Get description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Package requirements
requirements = [
    'numpy>=1.20.0',
    'pandas>=1.3.0',
    'scikit-learn>=1.0.0',
    'nltk>=3.6.0',
    'matplotlib>=3.4.0',
    'tqdm>=4.62.0',
]

# Optional requirements
extras_require = {
    'faiss': ['faiss-cpu>=1.7.0'],
    'transformers': ['sentence-transformers>=2.2.0'],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'flake8>=3.9.0',
        'black>=21.5b0',
    ],
}

setup(
    name='conversation-analytics',
    version=version,
    description='A Python package for analyzing conversation data from messaging applications',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='ZYRAV23',
    author_email='zyrav23@example.com',
    url='https://github.com/zyra-v23/conversation-analytics',
    packages=find_packages(include=['conversation_analytics', 'conversation_analytics.*']),
    include_package_data=True,
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'conversation-analytics=conversation_analytics.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
    ],
    python_requires='>=3.7',
) 