/// Statistical tests for functional group enrichment analysis.
///
/// Implements Fisher's exact test, Benjamini-Hochberg FDR correction,
/// chi-squared test for independence, Welch's t-test, and effect size measures.
/// All implementations are dependency-free (pure Rust).

use std::collections::HashMap;

// ═══════════════════════════════════════════════════
// Fisher's exact test (one-sided, right tail)
// ═══════════════════════════════════════════════════

/// 2×2 contingency table for Fisher's exact test.
///
/// ```text
///                  FG present    FG absent
///   In cluster        a             b        | a+b
///   Not in cluster    c             d        | c+d
///                   -----         -----
///                    a+c           b+d         n
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ContingencyTable {
    pub a: u64, // cluster with FG
    pub b: u64, // cluster without FG
    pub c: u64, // rest with FG
    pub d: u64, // rest without FG
}

impl ContingencyTable {
    pub fn new(a: u64, b: u64, c: u64, d: u64) -> Self {
        Self { a, b, c, d }
    }

    /// Build from cluster/population counts.
    /// - `cluster_with_fg`: molecules in cluster that have the FG
    /// - `cluster_size`: total molecules in cluster
    /// - `pop_with_fg`: molecules in population that have the FG
    /// - `pop_size`: total molecules in population
    pub fn from_counts(
        cluster_with_fg: usize,
        cluster_size: usize,
        pop_with_fg: usize,
        pop_size: usize,
    ) -> Self {
        let a = cluster_with_fg as u64;
        let b = (cluster_size - cluster_with_fg) as u64;
        let c = (pop_with_fg - cluster_with_fg) as u64;
        let d = (pop_size - cluster_size - pop_with_fg + cluster_with_fg) as u64;
        Self { a, b, c, d }
    }

    pub fn n(&self) -> u64 {
        self.a + self.b + self.c + self.d
    }
}

/// Compute log of binomial coefficient ln(C(n,k)) using log-gamma.
fn ln_choose(n: u64, k: u64) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// Lanczos approximation for ln(Gamma(x+1)) = ln(x!).
/// Accurate to ~15 digits for x > 0.5.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    // Stirling-like series coefficients (Lanczos g=7)
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let t = x + G + 0.5;
    let pi2 = (2.0 * std::f64::consts::PI).sqrt();

    (pi2).ln() + (t).ln() * (x + 0.5) - t + sum.ln()
}

/// Log of the hypergeometric probability P(X = k) for Fisher's exact test.
/// P(X=k) = C(K,k) * C(N-K, n-k) / C(N, n)
/// where N = total, K = total with FG, n = cluster size, k = cluster with FG
fn log_hypergeometric_pmf(table: &ContingencyTable) -> f64 {
    let n = table.n();
    let k_total = table.a + table.c;  // total with FG
    let n_cluster = table.a + table.b; // cluster size

    ln_choose(k_total, table.a)
        + ln_choose(n - k_total, n_cluster - table.a)
        - ln_choose(n, n_cluster)
}

/// Fisher's exact test, right-tailed (tests for over-representation).
/// Returns p-value = P(X >= observed | H0: no association).
pub fn fishers_exact_test(table: &ContingencyTable) -> f64 {
    let n = table.n();
    let k_total = table.a + table.c;
    let n_cluster = table.a + table.b;

    // Upper bound for a: min(K, n)
    let a_max = k_total.min(n_cluster);

    // Sum P(X >= a_observed) in log-space for numerical stability
    let mut log_p_values: Vec<f64> = Vec::new();

    for a in table.a..=a_max {
        let b = n_cluster - a;
        let c = k_total - a;
        // Check underflow: if c > k_total this is impossible
        if a > k_total { break; }
        let d = n - n_cluster - c;
        // Impossible configuration check
        if n < n_cluster + c { continue; }

        let tab = ContingencyTable::new(a, b, c, d);
        let lp = log_hypergeometric_pmf(&tab);
        if lp.is_finite() {
            log_p_values.push(lp);
        }
    }

    if log_p_values.is_empty() { return 1.0; }

    // Log-sum-exp for numerical stability
    let max_lp = log_p_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let p = (log_p_values.iter().map(|&lp| (lp - max_lp).exp()).sum::<f64>()).ln() + max_lp;

    p.exp().min(1.0).max(0.0)
}

// ═══════════════════════════════════════════════════
// Benjamini-Hochberg FDR correction
// ═══════════════════════════════════════════════════

/// Result of an enrichment test for one functional group in one cluster.
#[derive(Debug, Clone)]
pub struct EnrichmentResult {
    pub fg_name: String,
    pub enrichment_ratio: f64,
    pub p_value: f64,
    pub p_adjusted: f64,
    pub significant: bool, // p_adjusted < alpha
    pub cluster_prev_pct: f64,
    pub pop_prev_pct: f64,
    pub cluster_count: usize,
    pub cluster_size: usize,
}

/// Apply Benjamini-Hochberg FDR correction to a vector of p-values.
/// Returns adjusted p-values (same length, same order).
pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 { return Vec::new(); }

    // Sort indices by p-value
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted = vec![0.0; m];
    let mut cummin = f64::MAX;

    // Walk from largest to smallest rank
    for rank in (0..m).rev() {
        let i = indices[rank];
        let adj = (p_values[i] * m as f64 / (rank + 1) as f64).min(1.0);
        cummin = cummin.min(adj);
        adjusted[i] = cummin;
    }

    adjusted
}

/// Run enrichment analysis with Fisher's exact test + BH-FDR for one cluster.
pub fn enrichment_with_significance(
    cluster_fg_counts: &HashMap<String, usize>, // FG name -> count of molecules with that FG in cluster
    cluster_size: usize,
    pop_fg_counts: &HashMap<String, usize>,      // FG name -> count in population
    pop_size: usize,
    alpha: f64,
) -> Vec<EnrichmentResult> {
    let mut results: Vec<EnrichmentResult> = Vec::new();
    let mut p_values: Vec<f64> = Vec::new();

    for (fg_name, &pop_count) in pop_fg_counts {
        let pop_prev_pct = pop_count as f64 / pop_size.max(1) as f64 * 100.0;
        if pop_prev_pct < 1.0 { continue; } // Skip very rare FGs

        let cluster_count = *cluster_fg_counts.get(fg_name).unwrap_or(&0);
        let cluster_prev_pct = cluster_count as f64 / cluster_size.max(1) as f64 * 100.0;
        let enrichment_ratio = if pop_prev_pct > 0.0 { cluster_prev_pct / pop_prev_pct } else { 0.0 };

        let table = ContingencyTable::from_counts(
            cluster_count,
            cluster_size,
            pop_count,
            pop_size,
        );
        let p = fishers_exact_test(&table);
        p_values.push(p);

        results.push(EnrichmentResult {
            fg_name: fg_name.clone(),
            enrichment_ratio,
            p_value: p,
            p_adjusted: 0.0, // filled in after BH correction
            significant: false,
            cluster_prev_pct,
            pop_prev_pct,
            cluster_count,
            cluster_size,
        });
    }

    // Apply BH correction
    let adjusted = benjamini_hochberg(&p_values);
    for (i, result) in results.iter_mut().enumerate() {
        result.p_adjusted = adjusted[i];
        result.significant = adjusted[i] < alpha;
    }

    // Sort by enrichment ratio descending
    results.sort_by(|a, b| b.enrichment_ratio.partial_cmp(&a.enrichment_ratio).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ═══════════════════════════════════════════════════
// Chi-squared test for independence (2×2)
// ═══════════════════════════════════════════════════

/// Chi-squared test result.
#[derive(Debug, Clone)]
pub struct ChiSquaredResult {
    pub chi2: f64,
    pub p_value: f64,
    pub df: usize,
}

/// Chi-squared test for a 2×2 contingency table with Yates' correction.
pub fn chi_squared_test(table: &ContingencyTable) -> ChiSquaredResult {
    let n = table.n() as f64;
    let a = table.a as f64;
    let b = table.b as f64;
    let c = table.c as f64;
    let d = table.d as f64;

    // Row and column totals
    let r1 = a + b;
    let r2 = c + d;
    let c1 = a + c;
    let c2 = b + d;

    // Yates' continuity correction for 2×2 tables
    let numerator = (a * d - b * c).abs() - n / 2.0;
    let numerator = numerator.max(0.0);
    let chi2 = (n * numerator * numerator) / (r1 * r2 * c1 * c2).max(1e-10);

    // P-value from chi-squared distribution with df=1
    // Using the survival function: P(X > chi2) for df=1
    let p_value = chi2_survival(chi2, 1);

    ChiSquaredResult { chi2, p_value, df: 1 }
}

/// Survival function for chi-squared distribution P(X > x) with given df.
/// Uses the regularized incomplete gamma function.
fn chi2_survival(x: f64, df: usize) -> f64 {
    if x <= 0.0 { return 1.0; }
    let a = df as f64 / 2.0;
    let z = x / 2.0;
    1.0 - regularized_gamma_p(a, z)
}

/// Regularized lower incomplete gamma function P(a, x) = gamma(a,x) / Gamma(a).
/// Uses series expansion for small x and continued fraction for large x.
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 { return 0.0; }

    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction
        1.0 - gamma_cf(a, x)
    }
}

/// Series expansion for regularized incomplete gamma.
fn gamma_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-15 { break; }
    }

    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Continued fraction expansion for upper regularized incomplete gamma.
fn gamma_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);

    let mut f = 1e-30_f64;
    let mut c = 1e-30_f64;
    let mut d = 1.0 / (x + 1.0 - a);

    f = d;

    for n in 1..200 {
        let an = -(n as f64) * (n as f64 - a);
        let bn = x + 2.0 * n as f64 + 1.0 - a;

        d = bn + an * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        d = 1.0 / d;

        c = bn + an / c;
        if c.abs() < 1e-30 { c = 1e-30; }

        let delta = d * c;
        f *= delta;

        if (delta - 1.0).abs() < 1e-15 { break; }
    }

    f * (-x + a * x.ln() - ln_gamma_a).exp()
}

// ═══════════════════════════════════════════════════
// Welch's t-test (two-sample, unequal variance)
// ═══════════════════════════════════════════════════

/// Result of Welch's t-test.
#[derive(Debug, Clone)]
pub struct WelchTTestResult {
    pub t_statistic: f64,
    pub p_value: f64,
    pub df: f64,
    pub mean_diff: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub cohens_d: f64,
}

/// Welch's t-test for two independent samples with unequal variances.
/// Returns two-tailed p-value.
pub fn welch_t_test(x: &[f64], y: &[f64]) -> WelchTTestResult {
    let n1 = x.len() as f64;
    let n2 = y.len() as f64;

    let mean1 = x.iter().sum::<f64>() / n1;
    let mean2 = y.iter().sum::<f64>() / n2;
    let var1 = x.iter().map(|v| (v - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = y.iter().map(|v| (v - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let se = (var1 / n1 + var2 / n2).sqrt();
    let t = if se > 1e-15 { (mean1 - mean2) / se } else { 0.0 };

    // Welch-Satterthwaite degrees of freedom
    let num = (var1 / n1 + var2 / n2).powi(2);
    let denom = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
    let df = if denom > 1e-15 { num / denom } else { n1 + n2 - 2.0 };

    // Two-tailed p-value from t-distribution
    let p_value = 2.0 * t_survival(t.abs(), df);

    // 95% confidence interval for the difference
    let t_crit = t_quantile(0.975, df);
    let mean_diff = mean1 - mean2;
    let ci_lower = mean_diff - t_crit * se;
    let ci_upper = mean_diff + t_crit * se;

    // Cohen's d (pooled std)
    let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
    let cohens_d = if pooled_var > 1e-15 { (mean1 - mean2) / pooled_var.sqrt() } else { 0.0 };

    WelchTTestResult { t_statistic: t, p_value, df, mean_diff, ci_lower, ci_upper, cohens_d }
}

/// Survival function for t-distribution: P(T > t) using regularized beta function.
fn t_survival(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 * regularized_beta(x, df / 2.0, 0.5)
}

/// Regularized incomplete beta function I_x(a, b) using continued fraction.
fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }

    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;

    // Use Lentz's continued fraction
    let mut f = 1.0 + beta_cf_term(a, b, x, 1);
    if f.abs() < 1e-30 { f = 1e-30; }
    let mut c = f;
    let mut d = 1.0;

    for m in 1..200 {
        // Even step
        let am = beta_cf_coeff_even(a, b, x, m);
        d = 1.0 + am * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        d = 1.0 / d;
        c = 1.0 + am / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        f *= d * c;

        // Odd step
        let am = beta_cf_coeff_odd(a, b, x, m);
        d = 1.0 + am * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        d = 1.0 / d;
        c = 1.0 + am / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        let delta = d * c;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 { break; }
    }

    // Clamp result
    let result = front * f;
    result.min(1.0).max(0.0)
}

fn beta_cf_term(_a: f64, _b: f64, _x: f64, _m: usize) -> f64 { 0.0 }

fn beta_cf_coeff_even(a: f64, b: f64, x: f64, m: usize) -> f64 {
    let m = m as f64;
    (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
}

fn beta_cf_coeff_odd(a: f64, b: f64, x: f64, m: usize) -> f64 {
    let m = m as f64;
    -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
}

/// Approximate t-distribution quantile using Cornish-Fisher expansion.
/// For large df this approaches the normal quantile.
fn t_quantile(p: f64, df: f64) -> f64 {
    // Start with normal quantile (Beasley-Springer-Moro algorithm)
    let z = normal_quantile(p);

    // Cornish-Fisher correction for t-distribution
    if df > 1000.0 { return z; }
    let g1 = (z.powi(3) + z) / (4.0 * df);
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df * df);
    z + g1 + g2
}

/// Normal quantile function (probit) using rational approximation.
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-15 { return 0.0; }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Rational approximation (Abramowitz and Stegun 26.2.23)
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -z } else { z }
}

// ═══════════════════════════════════════════════════
// Linear regression
// ═══════════════════════════════════════════════════

/// Simple linear regression result.
#[derive(Debug, Clone)]
pub struct LinearRegressionResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub slope_se: f64,
    pub p_value: f64, // for H0: slope = 0
    pub n: usize,
}

/// Ordinary least-squares simple linear regression.
pub fn linear_regression(x: &[f64], y: &[f64]) -> LinearRegressionResult {
    let n = x.len().min(y.len());
    if n < 3 {
        return LinearRegressionResult {
            slope: 0.0, intercept: 0.0, r_squared: 0.0, slope_se: 0.0, p_value: 1.0, n,
        };
    }

    let mean_x = x.iter().take(n).sum::<f64>() / n as f64;
    let mean_y = y.iter().take(n).sum::<f64>() / n as f64;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }

    let slope = if ss_xx > 1e-15 { ss_xy / ss_xx } else { 0.0 };
    let intercept = mean_y - slope * mean_x;

    let r_squared = if ss_xx > 1e-15 && ss_yy > 1e-15 {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    } else { 0.0 };

    // Residual standard error
    let ss_res: f64 = (0..n).map(|i| {
        let predicted = intercept + slope * x[i];
        (y[i] - predicted).powi(2)
    }).sum();
    let mse = ss_res / (n as f64 - 2.0).max(1.0);
    let slope_se = if ss_xx > 1e-15 { (mse / ss_xx).sqrt() } else { 0.0 };

    // t-test for slope
    let t = if slope_se > 1e-15 { slope / slope_se } else { 0.0 };
    let df = (n as f64 - 2.0).max(1.0);
    let p_value = 2.0 * t_survival(t.abs(), df);

    LinearRegressionResult { slope, intercept, r_squared, slope_se, p_value, n }
}

// ═══════════════════════════════════════════════════
// Co-occurrence chi-squared test
// ═══════════════════════════════════════════════════

/// Test whether two functional groups co-occur more than expected by independence.
/// `fg_a[i]` and `fg_b[i]` are booleans for molecule i.
pub fn cooccurrence_chi_squared(fg_a: &[bool], fg_b: &[bool]) -> ChiSquaredResult {
    let n = fg_a.len().min(fg_b.len());
    let mut both = 0u64;
    let mut a_only = 0u64;
    let mut b_only = 0u64;
    let mut neither = 0u64;

    for i in 0..n {
        match (fg_a[i], fg_b[i]) {
            (true, true) => both += 1,
            (true, false) => a_only += 1,
            (false, true) => b_only += 1,
            (false, false) => neither += 1,
        }
    }

    chi_squared_test(&ContingencyTable::new(both, a_only, b_only, neither))
}

// ═══════════════════════════════════════════════════
// Summary statistics helpers
// ═══════════════════════════════════════════════════

/// Format a p-value for display.
pub fn format_p_value(p: f64) -> String {
    if p < 1e-100 { return "< 10⁻¹⁰⁰".to_string(); }
    if p < 1e-90 { return format!("< 10⁻⁹⁰"); }
    if p < 1e-50 { return format!("< 10⁻⁵⁰"); }
    if p < 1e-20 { return format!("< 10⁻²⁰"); }
    if p < 1e-10 { return format!("< 10⁻¹⁰"); }
    if p < 1e-6 { return format!("< 10⁻⁶"); }
    if p < 0.001 { return format!("{:.2e}", p); }
    format!("{:.4}", p)
}

/// Compute confidence interval for Pearson correlation using Fisher z-transform.
pub fn correlation_ci(r: f64, n: usize, confidence: f64) -> (f64, f64) {
    if n < 4 { return (r, r); }
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let se = 1.0 / ((n as f64 - 3.0).max(1.0)).sqrt();
    let alpha = 1.0 - confidence;
    let z_crit = normal_quantile(1.0 - alpha / 2.0);
    let lower = (2.0 * (z - z_crit * se)).exp() - 1.0;
    let lower = lower / ((2.0 * (z - z_crit * se)).exp() + 1.0);
    let upper = (2.0 * (z + z_crit * se)).exp() - 1.0;
    let upper = upper / ((2.0 * (z + z_crit * se)).exp() + 1.0);
    (lower.max(-1.0).min(1.0), upper.max(-1.0).min(1.0))
}

// ═══════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_exact_no_enrichment() {
        // Equal proportions: 50/100 in cluster vs 500/1000 in population
        let table = ContingencyTable::from_counts(50, 100, 500, 1000);
        let p = fishers_exact_test(&table);
        assert!(p > 0.05, "Expected non-significant, got p={}", p);
    }

    #[test]
    fn test_fisher_exact_strong_enrichment() {
        // Strong enrichment: 90/100 in cluster vs 200/1000 in population
        let table = ContingencyTable::from_counts(90, 100, 200, 1000);
        let p = fishers_exact_test(&table);
        assert!(p < 0.001, "Expected highly significant, got p={}", p);
    }

    #[test]
    fn test_benjamini_hochberg() {
        let pvals = vec![0.001, 0.04, 0.03, 0.5, 0.01];
        let adjusted = benjamini_hochberg(&pvals);
        // Adjusted p-values should be >= original
        for (adj, &orig) in adjusted.iter().zip(pvals.iter()) {
            assert!(*adj >= orig || (*adj - orig).abs() < 1e-10,
                "Adjusted {} < original {}", adj, orig);
        }
        // Adjusted p-values should be <= 1
        for adj in &adjusted {
            assert!(*adj <= 1.0 + 1e-10, "Adjusted p > 1: {}", adj);
        }
    }

    #[test]
    fn test_chi_squared() {
        // Strong association
        let table = ContingencyTable::new(90, 10, 10, 90);
        let result = chi_squared_test(&table);
        assert!(result.chi2 > 100.0, "Expected large chi2, got {}", result.chi2);
        assert!(result.p_value < 0.001, "Expected small p, got {}", result.p_value);
    }

    #[test]
    fn test_welch_t_test() {
        let x: Vec<f64> = (0..100).map(|i| 5.0 + (i as f64) * 0.01).collect();
        let y: Vec<f64> = (0..100).map(|i| 3.0 + (i as f64) * 0.01).collect();
        let result = welch_t_test(&x, &y);
        assert!(result.p_value < 0.001, "Expected significant difference");
        assert!((result.mean_diff - 2.0).abs() < 0.1, "Expected ~2.0 difference");
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.1, 7.9, 10.1];
        let result = linear_regression(&x, &y);
        assert!((result.slope - 2.0).abs() < 0.1, "Expected slope ~2.0, got {}", result.slope);
        assert!(result.r_squared > 0.99, "Expected high R², got {}", result.r_squared);
    }

    #[test]
    fn test_ln_gamma_basic() {
        // ln(Gamma(1)) = ln(0!) = 0
        assert!((ln_gamma(1.0)).abs() < 0.01, "ln_gamma(1) should be ~0");
        // ln(Gamma(6)) = ln(5!) = ln(120) ≈ 4.787
        assert!((ln_gamma(6.0) - 4.787).abs() < 0.01);
    }

    #[test]
    fn test_correlation_ci() {
        let (lo, hi) = correlation_ci(0.5, 100, 0.95);
        assert!(lo > 0.3 && lo < 0.5, "CI lower bound {}", lo);
        assert!(hi > 0.5 && hi < 0.7, "CI upper bound {}", hi);
    }
}
