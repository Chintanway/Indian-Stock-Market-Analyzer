from setuptools import setup, find_packages

setup(
    name="stockinsightai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'fastapi==0.68.2',
        'uvicorn[standard]==0.15.0',
        'python-multipart==0.0.20',
        'pandas==1.5.3',
        'numpy==1.24.4',
        'pandas-ta==0.3.14b0',
        'nsetools==1.1.8',
        'scikit-learn==1.2.2',
        'statsmodels==0.13.5',
        'plotly==5.15.0',
        'matplotlib==3.7.1',
        'jinja2==3.1.2',
        'requests==2.31.0',
        'beautifulsoup4==4.12.2',
        'sqlalchemy==1.4.50',
        'alembic==1.11.3',
        'yfinance==0.2.31',
        'python-dotenv==1.0.0',
        'gunicorn==20.1.0',
        'psycopg2-binary==2.9.7',
        'Cython==0.29.36'
    ],
    python_requires='>=3.9, <3.12',
)
