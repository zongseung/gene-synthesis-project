//! Pure-Rust numerics; no Python types here so it can be unit-tested with `cargo test`.
//!
//! Objective (training rows only):  sum_obs [2 sp(eta) - y eta] + 0.5 (||Z||^2 + ||V||^2),
//! eta_ij = b_j + z_i . v_j.  Fitted by alternating damped Newton (IRLS) steps: one step
//! per training row on z_i, then one step per variant on (b_j, v_j).  After each sweep the
//! gauge freedom that leaves every eta_ij unchanged is used to lower the penalty: the column
//! mean of Z moves into the intercept, and (Z, V) take the balanced SVD form of Z V^T.  Plain
//! alternating minimisation crawls along those directions at rates like 1 - 1/(j w).

#[derive(Debug)]
pub struct Fit {
    pub factors: Vec<f64>,   // n * k, row-major
    pub loadings: Vec<f64>,  // j * k, row-major
    pub intercept: Vec<f64>, // j
    pub objective_initial: f64,
    pub objective_final: f64,
    pub converged: bool,
}

const REL_TOL: f64 = 1e-10; // outer loop: stop when the objective's relative decrease is below this
const STEP_TOL: f64 = 1e-12; // block step: skip when the Newton decrement is below this (relative)
const ARMIJO: f64 = 1e-4;
const MAX_HALVINGS: u32 = 40;

/// For one entry: (2 softplus(eta) - y eta, y - 2p, 2p(1-p)) with a single exp; zeros if missing.
/// `ln(1+e)` instead of `ln_1p(e)`: twice as fast, and its absolute error (~1e-16) is below
/// the rounding noise of the O(1) terms it is summed with.
#[inline]
fn entry(eta: f64, y: f64) -> (f64, f64, f64) {
    if y.is_nan() {
        return (0.0, 0.0, 0.0);
    }
    let (p, sp) = if eta >= 0.0 {
        let e = (-eta).exp();
        (1.0 / (1.0 + e), eta + (1.0 + e).ln())
    } else {
        let e = eta.exp();
        (e / (1.0 + e), (1.0 + e).ln())
    };
    (2.0 * sp - y * eta, y - 2.0 * p, 2.0 * p * (1.0 - p))
}

#[inline]
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Deterministic pseudo-random in (-1, 1): splitmix64 on the index, no platform RNG.
fn hash_unit(seed: u64, idx: u64) -> f64 {
    let mut z = seed.wrapping_add(idx.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Solve the SPD system `a x = rhs` (dimension m, row-major) in place via Cholesky.
/// Returns false if a pivot is not positive (then `x` is unspecified).
fn cholesky_solve(a: &mut [f64], rhs: &[f64], x: &mut [f64], m: usize) -> bool {
    for c in 0..m {
        for r in c..m {
            let mut s = a[r * m + c];
            for k in 0..c {
                s -= a[r * m + k] * a[c * m + k];
            }
            if r == c {
                if s <= 0.0 || !s.is_finite() {
                    return false;
                }
                a[c * m + c] = s.sqrt();
            } else {
                a[r * m + c] = s / a[c * m + c];
            }
        }
    }
    for r in 0..m {
        let mut s = rhs[r];
        for k in 0..r {
            s -= a[r * m + k] * x[k];
        }
        x[r] = s / a[r * m + r];
    }
    for r in (0..m).rev() {
        let mut s = x[r];
        for k in r + 1..m {
            s -= a[k * m + r] * x[k];
        }
        x[r] = s / a[r * m + r];
    }
    true
}

/// Scratch for one damped Newton step on a block problem with `m` parameters and a
/// design of `m` contiguous columns of length `len`.
struct Newton {
    m: usize,
    info: Vec<f64>,
    grad: Vec<f64>,
    delta: Vec<f64>,
    eta: Vec<f64>,
    deta: Vec<f64>,
    r: Vec<f64>,
    w: Vec<f64>,
}

impl Newton {
    fn new(m: usize, len: usize) -> Self {
        Newton {
            m,
            info: vec![0.0; m * m],
            grad: vec![0.0; m],
            delta: vec![0.0; m],
            eta: vec![0.0; len],
            deta: vec![0.0; len],
            r: vec![0.0; len],
            w: vec![0.0; len],
        }
    }

    /// One damped Newton step on `theta` for the block objective
    ///   sum_obs [2 sp(eta) - y eta] + 0.5 ||theta[pen..]||^2,   eta = offset + design . theta,
    /// where `obs` (NaN = missing) and `offset` have length `len` and `design` holds m columns
    /// of length `len`. Returns (objective after the step, Newton decrement); theta is left
    /// alone and the decrement is 0 when the step was negligible or no Armijo step was accepted.
    fn step(&mut self, obs: &[f64], offset: Option<&[f64]>, design: &[f64], theta: &mut [f64], pen: usize) -> (f64, f64) {
        let (m, len) = (self.m, obs.len());
        let eta = &mut self.eta[..len];
        match offset {
            Some(o) => eta.copy_from_slice(o),
            None => eta.fill(0.0),
        }
        for c in 0..m {
            let (t, col) = (theta[c], &design[c * len..(c + 1) * len]);
            for i in 0..len {
                eta[i] += t * col[i];
            }
        }
        let mut f_old = 0.5 * dot(&theta[pen..], &theta[pen..]);
        for i in 0..len {
            let (nll, r, w) = entry(eta[i], obs[i]);
            f_old += nll;
            self.r[i] = r;
            self.w[i] = w;
        }
        for a in 0..m {
            let ca = &design[a * len..(a + 1) * len];
            self.grad[a] = dot(&self.r[..len], ca);
            for c in a..m {
                let cc = &design[c * len..(c + 1) * len];
                let s: f64 = (0..len).map(|i| self.w[i] * ca[i] * cc[i]).sum();
                self.info[a * m + c] = s;
                self.info[c * m + a] = s;
            }
        }
        // Penalty on theta[pen..] only; the unpenalised block gets a numerical floor so a
        // monomorphic variant (p -> 0 or 1, w -> 0) keeps the system SPD. It does not move
        // the optimum: the step is still H^{-1} g with g = 0 there.
        for a in 0..m {
            if a < pen {
                self.info[a * m + a] += 1e-12;
            } else {
                self.grad[a] -= theta[a];
                self.info[a * m + a] += 1.0;
            }
        }
        if !cholesky_solve(&mut self.info, &self.grad, &mut self.delta, m) {
            return (f_old, 0.0);
        }
        let slope = dot(&self.grad, &self.delta);
        if !(slope > STEP_TOL * f_old.abs().max(1.0)) {
            return (f_old, 0.0);
        }
        let deta = &mut self.deta[..len];
        deta.fill(0.0);
        for c in 0..m {
            let (d, col) = (self.delta[c], &design[c * len..(c + 1) * len]);
            for i in 0..len {
                deta[i] += d * col[i];
            }
        }
        let mut t = 1.0;
        for _ in 0..MAX_HALVINGS {
            let mut f_new = 0.0;
            for a in pen..m {
                let x = theta[a] + t * self.delta[a];
                f_new += 0.5 * x * x;
            }
            for i in 0..len {
                if !obs[i].is_nan() {
                    let e = eta[i] + t * deta[i];
                    f_new += 2.0 * softplus(e) - obs[i] * e;
                }
            }
            if f_new <= f_old - ARMIJO * t * slope {
                for a in 0..m {
                    theta[a] += t * self.delta[a];
                }
                return (f_new, slope);
            }
            t *= 0.5;
        }
        (f_old, 0.0)
    }
}

/// Score one row to convergence with V, b frozen (strictly convex in z), from the given start.
fn score_row(ws: &mut Newton, row: &[f64], z: &mut [f64], vt: &[f64], b: &[f64]) {
    for _ in 0..100 {
        let (_, decrement) = ws.step(row, Some(b), vt, z, 0);
        if decrement == 0.0 {
            break;
        }
    }
}

/// Row-major (rows, k) -> column-major (k columns of length rows), written at `out[..k*rows]`.
fn transpose(a: &[f64], rows: usize, k: usize, out: &mut [f64]) {
    for i in 0..rows {
        for c in 0..k {
            out[c * rows + i] = a[i * k + c];
        }
    }
}

/// Modified Gram-Schmidt of the row-major (rows, k) matrix `a` in place (becomes Q);
/// returns the upper-triangular R (k x k, row-major). False if a column is (numerically) zero.
fn gram_schmidt(a: &mut [f64], rows: usize, k: usize, r: &mut [f64]) -> bool {
    r.fill(0.0);
    for c in 0..k {
        for p in 0..c {
            let s: f64 = (0..rows).map(|i| a[i * k + p] * a[i * k + c]).sum();
            r[p * k + c] = s;
            for i in 0..rows {
                a[i * k + c] -= s * a[i * k + p];
            }
        }
        let norm = (0..rows).map(|i| a[i * k + c] * a[i * k + c]).sum::<f64>().sqrt();
        if !(norm > 1e-150) {
            return false;
        }
        r[c * k + c] = norm;
        for i in 0..rows {
            a[i * k + c] /= norm;
        }
    }
    true
}

/// One-sided Jacobi SVD of the k x k row-major matrix `m` = U S W^T.
/// On return `m` holds U (columns), `w` holds W, `s` the singular values.
fn jacobi_svd(m: &mut [f64], k: usize, w: &mut [f64], s: &mut [f64]) {
    w.fill(0.0);
    for c in 0..k {
        w[c * k + c] = 1.0;
    }
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..k {
            for q in p + 1..k {
                let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
                for i in 0..k {
                    a += m[i * k + p] * m[i * k + p];
                    b += m[i * k + q] * m[i * k + q];
                    c += m[i * k + p] * m[i * k + q];
                }
                if c.abs() <= 1e-15 * (a * b).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (b - a) / (2.0 * c);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = cs * t;
                for mat in [&mut *m, &mut *w] {
                    for i in 0..k {
                        let (x, y) = (mat[i * k + p], mat[i * k + q]);
                        mat[i * k + p] = cs * x - sn * y;
                        mat[i * k + q] = sn * x + cs * y;
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }
    for c in 0..k {
        s[c] = (0..k).map(|i| m[i * k + c] * m[i * k + c]).sum::<f64>().sqrt();
        if s[c] > 0.0 {
            for i in 0..k {
                m[i * k + c] /= s[c];
            }
        }
    }
}

/// Replace (Z, V) by the balanced SVD form of the same product Z V^T:
/// Z = Q_Z U S^{1/2}, V = Q_V W S^{1/2}, the representation of minimum 0.5(||Z||^2+||V||^2).
/// Leaves Z, V untouched when either is rank deficient.
fn rebalance(z: &mut [f64], nt: usize, v: &mut [f64], j: usize, k: usize) {
    let (mut qz, mut qv) = (z.to_vec(), v.to_vec());
    let (mut rz, mut rv) = (vec![0.0; k * k], vec![0.0; k * k]);
    if !gram_schmidt(&mut qz, nt, k, &mut rz) || !gram_schmidt(&mut qv, j, k, &mut rv) {
        return;
    }
    let mut m = vec![0.0; k * k]; // R_Z R_V^T
    for a in 0..k {
        for c in 0..k {
            m[a * k + c] = (0..k).map(|p| rz[a * k + p] * rv[c * k + p]).sum();
        }
    }
    let (mut w, mut s) = (vec![0.0; k * k], vec![0.0; k]);
    jacobi_svd(&mut m, k, &mut w, &mut s);
    let smax = s.iter().cloned().fold(0.0, f64::max);
    if !s.iter().all(|&x| x > 1e-12 * smax) {
        return;
    }
    let root: Vec<f64> = s.iter().map(|x| x.sqrt()).collect();
    for i in 0..nt {
        for c in 0..k {
            z[i * k + c] = (0..k).map(|p| qz[i * k + p] * m[p * k + c]).sum::<f64>() * root[c];
        }
    }
    for i in 0..j {
        for c in 0..k {
            v[i * k + c] = (0..k).map(|p| qv[i * k + p] * w[p * k + c]).sum::<f64>() * root[c];
        }
    }
}

/// Move the column mean of Z into the intercept: Z -= 1 m^T, b += V m. Same eta, smaller ||Z||^2.
fn centre(z: &mut [f64], nt: usize, v: &[f64], b: &mut [f64], j: usize, k: usize) {
    let mut m = vec![0.0; k];
    for i in 0..nt {
        for c in 0..k {
            m[c] += z[i * k + c];
        }
    }
    for x in m.iter_mut() {
        *x /= nt as f64;
    }
    for i in 0..nt {
        for c in 0..k {
            z[i * k + c] -= m[c];
        }
    }
    for jj in 0..j {
        b[jj] += dot(&v[jj * k..(jj + 1) * k], &m);
    }
}

/// Direct evaluation of the penalised objective on the training matrix (row-major).
fn total_objective(yt: &[f64], j: usize, z: &[f64], v: &[f64], b: &[f64], k: usize) -> f64 {
    let mut f = 0.5 * (dot(z, z) + dot(v, v));
    for i in 0..yt.len() / j {
        for jj in 0..j {
            let yv = yt[i * j + jj];
            if !yv.is_nan() {
                let eta = b[jj] + dot(&z[i * k..(i + 1) * k], &v[jj * k..(jj + 1) * k]);
                f += 2.0 * softplus(eta) - yv * eta;
            }
        }
    }
    f
}

pub fn fit(y: &[f64], n: usize, j: usize, train: &[i64], k: usize, max_iter: usize) -> Result<Fit, String> {
    if n == 0 || j == 0 {
        return Err("Expected a nonempty samples-by-variants matrix".into());
    }
    if y.iter().any(|&v| !v.is_nan() && v != 0.0 && v != 1.0 && v != 2.0) {
        return Err("Observed diploid calls must be 0, 1, 2; use NaN for missingness".into());
    }
    let mut seen = vec![false; n];
    for &t in train {
        if t < 0 || t as usize >= n || std::mem::replace(&mut seen[t as usize], true) {
            return Err("Training indices must be unique valid row indices".into());
        }
    }
    let nt = train.len();
    if nt == 0 {
        return Err("Training indices must be unique valid row indices".into());
    }
    if k < 1 || k >= nt.min(j) || max_iter < 1 {
        return Err("Components must be below sample/variant counts; iterations must be positive".into());
    }
    if y.iter().all(|v| v.is_nan()) {
        return Err("All calls are missing; every variant needs an observed training call".into());
    }

    // Training subset: row-major yt (nt, j) and column-major ytc (j columns of length nt).
    let mut yt = vec![0.0; nt * j];
    for (r, &t) in train.iter().enumerate() {
        let t = t as usize;
        yt[r * j..(r + 1) * j].copy_from_slice(&y[t * j..(t + 1) * j]);
    }
    let mut ytc = vec![0.0; nt * j];
    transpose(&yt, nt, j, &mut ytc);

    // intercept_j = logit((sum_i y_ij + 0.5) / (2 count_j + 1)); loadings/factors small, deterministic.
    let mut b = vec![0.0; j];
    for jj in 0..j {
        let (mut sum, mut cnt) = (0.0, 0usize);
        for &yv in &ytc[jj * nt..(jj + 1) * nt] {
            if !yv.is_nan() {
                sum += yv;
                cnt += 1;
            }
        }
        if cnt == 0 {
            return Err("Every variant needs an observed training call".into());
        }
        let p = (sum + 0.5) / (2.0 * cnt as f64 + 1.0);
        b[jj] = (p / (1.0 - p)).ln();
    }
    let mut v: Vec<f64> = (0..j * k).map(|i| 0.1 * hash_unit(0x5EED_0001, i as u64)).collect();
    let mut z: Vec<f64> = (0..nt * k).map(|i| 0.1 * hash_unit(0x5EED_0002, i as u64)).collect();
    // Column-major copies used as Newton designs: vt (k columns of length j) for the row
    // steps, zt (a ones column then k columns of length nt) for the column steps.
    let mut vt = vec![0.0; k * j];
    let mut zt = vec![1.0; (k + 1) * nt];

    let objective_initial = total_objective(&yt, j, &z, &v, &b, k);
    let mut f = objective_initial;
    let mut converged = false;
    let mut rows = Newton::new(k, j);
    let mut cols = Newton::new(k + 1, nt);
    let mut theta = vec![0.0; k + 1];

    for _ in 0..max_iter {
        // (a) factors step, one damped Newton step per training row.
        transpose(&v, j, k, &mut vt);
        for i in 0..nt {
            rows.step(&yt[i * j..(i + 1) * j], Some(&b), &vt, &mut z[i * k..(i + 1) * k], 0);
        }
        // (b) loadings + intercept step, one damped Newton step per variant column.
        // The accepted column objectives sum to the NLL plus 0.5||V||^2.
        transpose(&z, nt, k, &mut zt[nt..]);
        let mut nll = 0.0;
        for jj in 0..j {
            theta[0] = b[jj];
            theta[1..].copy_from_slice(&v[jj * k..(jj + 1) * k]);
            nll += cols.step(&ytc[jj * nt..(jj + 1) * nt], None, &zt, &mut theta, 1).0;
            b[jj] = theta[0];
            v[jj * k..(jj + 1) * k].copy_from_slice(&theta[1..]);
        }
        nll -= 0.5 * dot(&v, &v);
        // (c) gauge fixes that keep every eta_ij and lower the penalty.
        centre(&mut z, nt, &v, &mut b, j, k);
        rebalance(&mut z, nt, &mut v, j, k);
        let f_new = nll + 0.5 * (dot(&z, &z) + dot(&v, &v));
        let decrease = f - f_new;
        f = f_new;
        if decrease <= REL_TOL * f.abs().max(1.0) {
            converged = true;
            break;
        }
    }
    if !f.is_finite() || f > objective_initial {
        return Err("Binomial GLM-PCA failed to reduce its penalized likelihood".into());
    }

    // Score every row with V, b frozen: training rows warm-start from their fitted factors
    // (already at this optimum), held-out rows start from zero.
    transpose(&v, j, k, &mut vt);
    let mut factors = vec![0.0; n * k];
    for (r, &t) in train.iter().enumerate() {
        let t = t as usize;
        factors[t * k..(t + 1) * k].copy_from_slice(&z[r * k..(r + 1) * k]);
    }
    for i in 0..n {
        score_row(&mut rows, &y[i * j..(i + 1) * j], &mut factors[i * k..(i + 1) * k], &vt, &b);
    }
    Ok(Fit { factors, loadings: v, intercept: b, objective_initial, objective_final: f, converged })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small planted-signal fit: the objective drops, the fit is finite, every row is
    /// scored, the sweep bookkeeping matches a direct evaluation of the objective, scored
    /// rows are stationary, and a monomorphic column plus missing entries do not break anything.
    #[test]
    fn fit_reduces_objective_and_handles_missing_and_monomorphic() {
        let (n, j, k) = (40usize, 6usize, 2usize);
        let mut y = Vec::with_capacity(n * j);
        for i in 0..n {
            for jj in 0..j {
                let eta = -0.5 + (i as f64 / n as f64 - 0.5) * 3.0 * (jj as f64 - 2.5);
                let p = 1.0 / (1.0 + (-eta).exp());
                let u = 0.5 * (hash_unit(7, (i * j + jj) as u64) + 1.0);
                let g = if jj == 5 { 0.0 } else if u < p * p { 2.0 } else if u < p * p + 2.0 * p * (1.0 - p) { 1.0 } else { 0.0 };
                y.push(g);
            }
        }
        y[0] = f64::NAN;
        let train: Vec<i64> = (0..30).collect();
        let fit = fit(&y, n, j, &train, k, 200).unwrap();
        assert!(fit.objective_final < fit.objective_initial);
        assert!(fit.converged);
        assert!(fit.factors.iter().chain(&fit.loadings).chain(&fit.intercept).all(|x| x.is_finite()));
        assert_eq!(fit.factors.len(), n * k);
        let mut yt = Vec::new();
        for &t in &train { yt.extend_from_slice(&y[t as usize * j..(t as usize + 1) * j]); }
        let ztrain: Vec<f64> = train.iter().flat_map(|&t| fit.factors[t as usize * k..(t as usize + 1) * k].to_vec()).collect();
        let direct = total_objective(&yt, j, &ztrain, &fit.loadings, &fit.intercept, k);
        assert!((direct - fit.objective_final).abs() <= 1e-6 * direct, "{direct} vs {}", fit.objective_final);
        // A scored row is at the optimum of its own block: a further step is a no-op.
        let mut vt = vec![0.0; j * k];
        transpose(&fit.loadings, j, k, &mut vt);
        let mut ws = Newton::new(k, j);
        let mut z = fit.factors[..k].to_vec();
        assert_eq!(ws.step(&y[..j], Some(&fit.intercept), &vt, &mut z, 0).1, 0.0);
        // Errors: unobserved column and too many components.
        let mut bad = y.clone();
        for i in 0..n { bad[i * j + 3] = f64::NAN; }
        assert!(super::fit(&bad, n, j, &train, k, 10).unwrap_err().contains("observed training call"));
        assert!(super::fit(&y, n, j, &train, 6, 10).is_err());
    }
}
