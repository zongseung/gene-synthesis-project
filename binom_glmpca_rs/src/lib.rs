//! Binomial(2, p) GLM-PCA fitted by alternating damped Newton (IRLS) steps.
//!
//! Same penalised objective as `src/preprocessing/binomial_glm_pca.py`:
//!   sum_obs [2*softplus(eta) - y*eta] + 0.5*(||Z||^2 + ||V||^2),  eta = b_j + z_i . v_j
//! The intercept b is not penalised. Loadings/intercept are fitted on the
//! training rows only; every row is then scored with V, b frozen.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod fit;

/// Fit Binomial(2, p) GLM-PCA. `y` is (n, j) f64 with NaN for missing calls.
#[pyfunction]
fn fit_binomial<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    train_indices: PyReadonlyArray1<'py, i64>,
    components: usize,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = y.shape();
    let (n, j) = (shape[0], shape[1]);
    let y: Vec<f64> = y.as_array().iter().copied().collect();
    let train: Vec<i64> = train_indices.as_array().iter().copied().collect();

    let out = py
        .allow_threads(|| fit::fit(&y, n, j, &train, components, max_iter))
        .map_err(PyValueError::new_err)?;

    let d = PyDict::new_bound(py);
    d.set_item("factors", PyArray1::from_vec_bound(py, out.factors).reshape([n, components])?)?;
    d.set_item("loadings", PyArray1::from_vec_bound(py, out.loadings).reshape([j, components])?)?;
    d.set_item("intercept", PyArray1::from_vec_bound(py, out.intercept))?;
    d.set_item("objective_initial", out.objective_initial)?;
    d.set_item("objective_final", out.objective_final)?;
    d.set_item("converged", out.converged)?;
    Ok(d)
}

#[pymodule]
fn binom_glmpca_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_binomial, m)?)?;
    Ok(())
}
