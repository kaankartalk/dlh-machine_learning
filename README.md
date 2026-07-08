# DLH — Machine Learning

A structured curriculum building the mathematical and data engineering foundations for machine learning.

---

## Directory Structure

```
dlh-machine_learning/
├── math/                          # Mathematical foundations
│   ├── linear_algebra/            # Matrix operations: Python lists → NumPy
│   │   ├── 0-slice_me_up.py through 14-saddle_up.py
│   │   └── README.md
│   ├── advanced_linear_algebra/   # Determinant, minor, cofactor, adjugate, inverse, definiteness
│   │   ├── 0-determinant.py through 5-definiteness.py
│   │   └── README.md
│   ├── calculus/                  # Sigma/Pi notation, derivatives, partial derivatives, integrals
│   │   ├── 0-sigma_is_for_sum through 17-integrate.py
│   │   └── README.md
│   ├── probability/               # Distributions: binomial, normal, poisson, exponential
│   │   ├── binomial.py, normal.py, poisson.py, exponential.py
│   │   └── README.md
│   ├── bayesian_prob/             # Likelihood, intersection, marginal, posterior probability
│   │   ├── 0-likelihood.py through 3-posterior.py
│   │   └── README.md
│   ├── multivariate_prob/         # Mean vector, covariance, correlation, multivariate normal
│   │   ├── 0-mean_cov.py, 1-correlation.py, multinormal.py
│   │   └── README.md
│   ├── plotting/                  # Matplotlib: line, scatter, bar, frequency, all-in-one
│   │   ├── 0-line.py through 6-bars.py
│   │   └── README.md
│   └── README.md
├── pipeline/                       # Data engineering
│   ├── databases/                  # SQL (MySQL) + NoSQL (MongoDB)
│   │   ├── 0-create_database_if_missing.sql through 104-find
│   │   └── README.md
│   ├── pandas/                     # DataFrames: from NumPy/dict/file, slicing, renaming
│   │   ├── 0-from_numpy.py through 7-high.py
│   │   └── README.md
└── README.md
```

---

## Quick Reference

| Track | Module | Topics | Tasks |
|-------|--------|--------|-------|
| **Math** | [Linear Algebra](math/linear_algebra/) | Slicing, shape, transpose, element-wise ops, concatenation, matrix multiplication, NumPy vectorization | 15 |
| **Math** | [Advanced Linear Algebra](math/advanced_linear_algebra/) | Determinant, minor, cofactor, adjugate, inverse, definiteness (manual + NumPy) | 6 |
| **Math** | [Calculus](math/calculus/) | Summation/product notation, derivatives, partial derivatives, integrals, definite/indefinite, double integrals | 18 |
| **Math** | [Probability](math/probability/) | Binomial, normal, poisson, exponential distributions | 4 |
| **Math** | [Bayesian Probability](math/bayesian_prob/) | Likelihood, intersection, marginal, posterior probability | 4 |
| **Math** | [Multivariate Probability](math/multivariate_prob/) | Mean vector, covariance, correlation, multivariate normal distribution | 3 |
| **Math** | [Plotting](math/plotting/) | Line, scatter, bar, frequency, all-in-one, change of scale | 7 |
| **Pipeline** | [Databases](pipeline/databases/) | DDL, CRUD, filtering/sorting, joins, aggregates, constraints, triggers, indexing — plus MongoDB (PyMongo CRUD & queries) | 39 (+1 schema) |
| **Pipeline** | [Pandas](pipeline/pandas/) | Building DataFrames from NumPy/dict/file, renaming, slicing, boolean indexing | 8 |

---

## Learning Progression

### Math Track
1. **Python Slicing** → 2. **Manual Matrix Ops** (nested loops) → 3. **NumPy Vectorization** → 4. **Advanced Linear Algebra** (determinant → inverse → definiteness) → 5. **Calculus** (derivatives → integrals) → 6. **Probability Distributions** (binomial → normal → poisson → exponential) → 7. **Bayesian Probability** (likelihood → posterior) → 8. **Multivariate Probability** (covariance → correlation → multivariate normal) → 9. **Visualization** (plotting)

### Pipeline Track
1. **SQL Foundations** (CREATE, CRUD) → 2. **Filtering/Sorting** → 3. **Joins & Aggregates** → 4. **Constraints & Indexing** → 5. **Triggers & Stored Logic** → 6. **MongoDB** (NoSQL CRUD & queries) → 7. **Pandas DataFrames**

---

## Setup

```bash
git clone https://github.com/kaankartalk/dlh-machine_learning.git
cd dlh-machine_learning
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib pymongo mysql-connector-python
```

---

## Resources

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [MongoDB / PyMongo Documentation](https://pymongo.readthedocs.io/)
- [Python Official Documentation](https://docs.python.org/3/)
