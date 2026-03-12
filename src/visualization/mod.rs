/// Visualization module: generates SVG plots for dissertation-grade reporting.
/// Uses the `plotters` crate with SVG backend for high-quality vector graphics.

use plotters::prelude::*;
use std::path::Path;
use std::collections::HashMap;
use umap_rs::{Umap as UmapAlgo, UmapConfig as UmapCfg, GraphParams as UmapGraphParams,
               OptimizationParams as UmapOptParams};
use rayon::prelude::*;

// ═══════════════════════════════════════════════════
// Color palettes
// ═══════════════════════════════════════════════════

const STRATUM_COLORS: [RGBColor; 5] = [
    RGBColor(31, 119, 180),   // blue
    RGBColor(255, 127, 14),   // orange
    RGBColor(44, 160, 44),    // green
    RGBColor(214, 39, 40),    // red
    RGBColor(148, 103, 189),  // purple
];

const PROPERTY_COLORS: [RGBColor; 3] = [
    RGBColor(31, 119, 180),   // QED - blue
    RGBColor(255, 127, 14),   // logP - orange
    RGBColor(44, 160, 44),    // SAS - green
];

fn heatmap_color(value: f64, min_val: f64, max_val: f64) -> RGBColor {
    let t = if (max_val - min_val).abs() < 1e-12 {
        0.5
    } else {
        ((value - min_val) / (max_val - min_val)).clamp(0.0, 1.0)
    };
    if t < 0.5 {
        let s = t * 2.0;
        RGBColor(
            (59.0 + s * 196.0) as u8,
            (76.0 + s * 179.0) as u8,
            (192.0 + s * 63.0) as u8,
        )
    } else {
        let s = (t - 0.5) * 2.0;
        RGBColor(
            (255.0 - s * 42.0) as u8,
            (255.0 - s * 216.0) as u8,
            (255.0 - s * 217.0) as u8,
        )
    }
}

fn sequential_color(value: f64, min_val: f64, max_val: f64) -> RGBColor {
    let t = if (max_val - min_val).abs() < 1e-12 { 0.5 }
        else { ((value - min_val) / (max_val - min_val)).clamp(0.0, 1.0) };
    // Viridis-inspired: dark purple → teal → yellow
    let r = (68.0 + t * 187.0) as u8;
    let g = (1.0 + t * 214.0) as u8;
    let b = (84.0 + (1.0 - (2.0 * t - 1.0).abs()) * 128.0) as u8;
    RGBColor(r, g, b)
}

// ═══════════════════════════════════════════════════
// 1. Property distribution histograms
// ═══════════════════════════════════════════════════

pub fn plot_property_distributions(
    qed_vals: &[f64],
    logp_vals: &[f64],
    sas_vals: &[f64],
    output_dir: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut paths = Vec::new();

    let configs: Vec<(&str, &str, &[f64], usize, RGBColor)> = vec![
        ("qed_distribution.svg", "QED (Drug-likeness) Distribution", qed_vals, 50, PROPERTY_COLORS[0]),
        ("logp_distribution.svg", "logP (Lipophilicity) Distribution", logp_vals, 50, PROPERTY_COLORS[1]),
        ("sas_distribution.svg", "SAS (Synthetic Accessibility) Distribution", sas_vals, 50, PROPERTY_COLORS[2]),
    ];

    for (filename, title, data, bins, color) in &configs {
        let path = format!("{}/figures/{}", output_dir, filename);
        let x_label = if filename.contains("qed") { "QED" }
            else if filename.contains("logp") { "logP" }
            else { "SAS" };
        plot_histogram(data, &path, title, x_label, *bins, *color)?;
        paths.push(format!("figures/{}", filename));
    }

    // Combined overlay
    let path = format!("{}/figures/property_distributions_combined.svg", output_dir);
    plot_property_overlay(qed_vals, logp_vals, sas_vals, &path)?;
    paths.push("figures/property_distributions_combined.svg".to_string());

    Ok(paths)
}

fn plot_histogram(
    data: &[f64],
    path: &str,
    title: &str,
    x_label: &str,
    num_bins: usize,
    color: RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_parent_dir(path);
    let (bins, counts) = compute_histogram(data, num_bins);
    let max_count = *counts.iter().max().unwrap_or(&1) as f64;

    let root = SVGBackend::new(path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22).into_font())
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(bins[0]..bins[bins.len() - 1], 0.0..max_count * 1.05)?;

    chart.configure_mesh()
        .x_desc(x_label)
        .y_desc("Count")
        .x_label_style(("sans-serif", 14))
        .y_label_style(("sans-serif", 14))
        .draw()?;

    let bar_width = (bins[1] - bins[0]) * 0.95;
    chart.draw_series(
        bins.windows(2).zip(counts.iter()).map(|(bin_pair, &count)| {
            let x0 = bin_pair[0];
            let x1 = x0 + bar_width;
            Rectangle::new([(x0, 0.0), (x1, count as f64)], color.mix(0.8).filled())
        })
    )?;

    // Add mean line
    let mean = data.iter().sum::<f64>() / data.len().max(1) as f64;
    chart.draw_series(LineSeries::new(
        vec![(mean, 0.0), (mean, max_count * 1.0)],
        RED.stroke_width(2),
    ))?
    .label(format!("Mean = {:.3}", mean))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 13))
        .draw()?;

    root.present()?;
    Ok(())
}

fn plot_property_overlay(
    qed_vals: &[f64],
    logp_vals: &[f64],
    sas_vals: &[f64],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_parent_dir(path);
    let root = SVGBackend::new(path, (1200, 450)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 3));
    let datasets: Vec<(&str, &[f64], RGBColor)> = vec![
        ("QED", qed_vals, PROPERTY_COLORS[0]),
        ("logP", logp_vals, PROPERTY_COLORS[1]),
        ("SAS", sas_vals, PROPERTY_COLORS[2]),
    ];

    for (area, (label, data, color)) in areas.iter().zip(datasets.iter()) {
        let (bins, counts) = compute_histogram(data, 40);
        let max_count = *counts.iter().max().unwrap_or(&1) as f64;

        let mut chart = ChartBuilder::on(area)
            .caption(*label, ("sans-serif", 20).into_font())
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(55)
            .build_cartesian_2d(bins[0]..bins[bins.len() - 1], 0.0..max_count * 1.1)?;

        chart.configure_mesh()
            .x_desc(*label)
            .y_desc("Count")
            .x_label_style(("sans-serif", 12))
            .y_label_style(("sans-serif", 12))
            .draw()?;

        let bar_width = (bins[1] - bins[0]) * 0.92;
        chart.draw_series(
            bins.windows(2).zip(counts.iter()).map(|(bp, &c)| {
                Rectangle::new([(bp[0], 0.0), (bp[0] + bar_width, c as f64)], color.mix(0.75).filled())
            })
        )?;
    }

    root.present()?;
    Ok(())
}

// ═══════════════════════════════════════════════════
// 2. Functional group prevalence bar chart
// ═══════════════════════════════════════════════════

pub fn plot_fg_prevalence(
    fg_data: &[(String, f64)],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/fg_prevalence.svg", output_dir);
    ensure_parent_dir(&path);

    let n = fg_data.len();
    let root = SVGBackend::new(&path, (900, 80 + n as u32 * 32)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_pct = fg_data.iter().map(|(_, p)| *p).fold(0.0_f64, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Functional Group Prevalence (%)", ("sans-serif", 20).into_font())
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(200)
        .build_cartesian_2d(0.0..max_pct, 0..n)?;

    chart.configure_mesh()
        .x_desc("Prevalence (%)")
        .y_label_formatter(&|idx| {
            fg_data.get(*idx).map(|(name, _)| name.clone()).unwrap_or_default()
        })
        .x_label_style(("sans-serif", 12))
        .y_label_style(("sans-serif", 11))
        .draw()?;

    chart.draw_series(
        fg_data.iter().enumerate().map(|(i, (_, pct))| {
            let color = if *pct > 50.0 {
                RGBColor(31, 119, 180)
            } else if *pct > 10.0 {
                RGBColor(44, 160, 44)
            } else {
                RGBColor(148, 103, 189)
            };
            Rectangle::new([(0.0, i), (*pct, i + 1)], color.mix(0.8).filled())
        })
    )?;

    root.present()?;
    Ok("figures/fg_prevalence.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 3. Latent space UMAP scatter plot
// ═══════════════════════════════════════════════════

pub fn plot_latent_space_umap(
    embeddings: &[Vec<f32>],
    stratum_labels: &[usize],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/latent_space_umap.svg", output_dir);
    ensure_parent_dir(&path);

    // Subsample for UMAP (max 15000 points for performance)
    let step = (embeddings.len() / 15000).max(1);
    let sampled: Vec<Vec<f32>> = embeddings.iter().step_by(step).cloned().collect();
    let sampled_strata: Vec<usize> = stratum_labels.iter().step_by(step).copied().collect();

    // Run optimized parallel UMAP via umap-rs (Rayon Hogwild! SGD)
    let n_samples = sampled.len();
    let n_features = sampled[0].len();
    let k = 15usize.min(n_samples - 1);

    // Build data matrix using ndarray 0.17 (required by umap-rs)
    let mut data_flat: Vec<f32> = Vec::with_capacity(n_samples * n_features);
    for e in &sampled {
        for &v in e { data_flat.push(v); }
    }
    let data_arr = ndarray_017::Array2::from_shape_vec((n_samples, n_features), data_flat)
        .map_err(|e| format!("UMAP data array error: {}", e))?;

    // Parallel brute-force KNN (15k × 16-dim is trivially fast with rayon)
    let knn_results: Vec<Vec<(usize, f32)>> = (0..n_samples).into_par_iter().map(|i| {
        let mut dists: Vec<(usize, f32)> = (0..n_samples)
            .filter(|&j| j != i)
            .map(|j| {
                let d: f32 = sampled[i].iter().zip(sampled[j].iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.truncate(k);
        dists
    }).collect();

    let mut knn_idx_flat: Vec<u32> = vec![0; n_samples * k];
    let mut knn_dist_flat: Vec<f32> = vec![0.0; n_samples * k];
    for (i, neighbors) in knn_results.iter().enumerate() {
        for (j, &(idx, dist)) in neighbors.iter().enumerate() {
            knn_idx_flat[i * k + j] = idx as u32;
            knn_dist_flat[i * k + j] = dist;
        }
    }
    let knn_indices = ndarray_017::Array2::from_shape_vec((n_samples, k), knn_idx_flat)
        .map_err(|e| format!("KNN index array error: {}", e))?;
    let knn_dists = ndarray_017::Array2::from_shape_vec((n_samples, k), knn_dist_flat)
        .map_err(|e| format!("KNN dist array error: {}", e))?;

    // Random 2D initialization
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut init_flat: Vec<f32> = Vec::with_capacity(n_samples * 2);
    for _ in 0..n_samples {
        init_flat.push(rng.gen::<f32>() * 20.0 - 10.0);
        init_flat.push(rng.gen::<f32>() * 20.0 - 10.0);
    }
    let init = ndarray_017::Array2::from_shape_vec((n_samples, 2), init_flat)
        .map_err(|e| format!("UMAP init array error: {}", e))?;

    let config = UmapCfg {
        n_components: 2,
        graph: UmapGraphParams {
            n_neighbors: k,
            ..Default::default()
        },
        optimization: UmapOptParams {
            n_epochs: Some(500),
            ..Default::default()
        },
        ..Default::default()
    };
    let umap = UmapAlgo::new(config);
    let fitted = umap.fit(data_arr.view(), knn_indices.view(), knn_dists.view(), init.view());
    let embedding = fitted.embedding();
    let umap_result: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| vec![embedding[[i, 0]] as f64, embedding[[i, 1]] as f64])
        .collect();

    let points: Vec<(f64, f64, usize)> = umap_result.iter()
        .zip(sampled_strata.iter())
        .map(|(coords, &s)| (coords[0], coords[1], s))
        .collect();

    let x_min = points.iter().map(|p| p.0).fold(f64::MAX, f64::min);
    let x_max = points.iter().map(|p| p.0).fold(f64::MIN, f64::max);
    let y_min = points.iter().map(|p| p.1).fold(f64::MAX, f64::min);
    let y_max = points.iter().map(|p| p.1).fold(f64::MIN, f64::max);
    let x_pad = (x_max - x_min) * 0.05;
    let y_pad = (y_max - y_min) * 0.05;

    let root = SVGBackend::new(&path, (1000, 750)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Latent Space (UMAP Projection)", ("sans-serif", 22).into_font())
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(
            (x_min - x_pad)..(x_max + x_pad),
            (y_min - y_pad)..(y_max + y_pad),
        )?;

    chart.configure_mesh()
        .x_desc("UMAP 1")
        .y_desc("UMAP 2")
        .x_label_style(("sans-serif", 14))
        .y_label_style(("sans-serif", 14))
        .draw()?;

    let qed_labels = ["QED [0, 0.4)", "QED [0.4, 0.52)", "QED [0.52, 0.69)", "QED [0.69, 0.81)", "QED [0.81, 1.0]"];

    for stratum in 0..5 {
        let stratum_points: Vec<(f64, f64)> = points.iter()
            .filter(|p| p.2 == stratum)
            .map(|p| (p.0, p.1))
            .collect();

        if stratum_points.is_empty() { continue; }

        let color = STRATUM_COLORS[stratum];
        chart.draw_series(
            stratum_points.iter().map(|&(x, y)| {
                Circle::new((x, y), 4, color.mix(0.6).filled())
            })
        )?
        .label(*qed_labels.get(stratum).unwrap_or(&""))
        .legend(move |(x, y)| Circle::new((x + 10, y), 6, color.filled()));
    }

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()?;

    root.present()?;
    Ok("figures/latent_space_umap.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 4. Cluster size distribution
// ═══════════════════════════════════════════════════

pub fn plot_cluster_size_distributions(
    strata_cluster_sizes: &[Vec<usize>],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/cluster_size_distribution.svg", output_dir);
    ensure_parent_dir(&path);

    let n_strata = strata_cluster_sizes.len();
    let root = SVGBackend::new(&path, (900, 250 * n_strata as u32 + 80)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((n_strata, 1));
    let qed_labels = ["Stratum 0: QED [0, 0.4)", "Stratum 1: QED [0.4, 0.52)", "Stratum 2: QED [0.52, 0.69)", "Stratum 3: QED [0.69, 0.81)", "Stratum 4: QED [0.81, 1.0]"];

    for (i, (area, sizes)) in areas.iter().zip(strata_cluster_sizes.iter()).enumerate() {
        let float_sizes: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
        let (bins, counts) = compute_histogram(&float_sizes, 30);
        let max_count = *counts.iter().max().unwrap_or(&1) as f64;

        let mut chart = ChartBuilder::on(area)
            .caption(qed_labels.get(i).unwrap_or(&""), ("sans-serif", 18).into_font())
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(50)
            .build_cartesian_2d(bins[0]..bins[bins.len() - 1], 0.0..max_count * 1.15)?;

        chart.configure_mesh()
            .x_desc("Cluster Size")
            .y_desc("Count")
            .x_label_style(("sans-serif", 12))
            .y_label_style(("sans-serif", 12))
            .draw()?;

        let bar_width = (bins[1] - bins[0]) * 0.92;
        let color = STRATUM_COLORS[i % 5];
        chart.draw_series(
            bins.windows(2).zip(counts.iter()).map(|(bp, &c)| {
                Rectangle::new([(bp[0], 0.0), (bp[0] + bar_width, c as f64)], color.mix(0.75).filled())
            })
        )?;
    }

    root.present()?;
    Ok("figures/cluster_size_distribution.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 5. Dimension-property correlation heatmap
// ═══════════════════════════════════════════════════

pub fn plot_dim_property_heatmap(
    dim_correlations: &[(usize, f64, f64, f64)],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/dim_property_heatmap.svg", output_dir);
    ensure_parent_dir(&path);

    let n_dims = dim_correlations.len();
    let properties = ["QED", "logP", "SAS"];
    let n_props = properties.len();

    let root = SVGBackend::new(&path, (800, 100 + n_dims as u32 * 35)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Latent Dimension ↔ Property Correlations", ("sans-serif", 20).into_font())
        .margin(15)
        .x_label_area_size(60)
        .y_label_area_size(55)
        .build_cartesian_2d(0..n_props, 0..n_dims)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_label_formatter(&|idx| properties.get(*idx).unwrap_or(&"").to_string())
        .y_label_formatter(&|idx| format!("Dim {}", dim_correlations.get(*idx).map(|d| d.0).unwrap_or(*idx)))
        .x_label_style(("sans-serif", 13))
        .y_label_style(("sans-serif", 12))
        .draw()?;

    for (row, &(_, r_qed, r_logp, r_sas)) in dim_correlations.iter().enumerate() {
        let vals = [r_qed, r_logp, r_sas];
        for (col, &val) in vals.iter().enumerate() {
            let color = heatmap_color(val, -1.0, 1.0);
            chart.draw_series(std::iter::once(
                Rectangle::new([(col, row), (col + 1, row + 1)], color.filled())
            ))?;
            let text_color = if val.abs() > 0.5 { &WHITE } else { &BLACK };
            chart.draw_series(std::iter::once(
                Text::new(
                    format!("{:+.3}", val),
                    (col, row),
                    ("sans-serif", 12).into_font().color(text_color),
                )
            ))?;
        }
    }

    root.present()?;
    Ok("figures/dim_property_heatmap.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 6. FG-property correlation heatmap
// ═══════════════════════════════════════════════════

pub fn plot_fg_property_correlations(
    fg_correlations: &[(String, f64, f64, f64)],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/fg_property_correlations.svg", output_dir);
    ensure_parent_dir(&path);

    let n_fgs = fg_correlations.len();
    let root = SVGBackend::new(&path, (750, 100 + n_fgs as u32 * 32)).into_drawing_area();
    root.fill(&WHITE)?;

    let properties = ["QED", "logP", "SAS"];

    let mut chart = ChartBuilder::on(&root)
        .caption("Functional Group ↔ Property Correlations", ("sans-serif", 20).into_font())
        .margin(15)
        .x_label_area_size(60)
        .y_label_area_size(200)
        .build_cartesian_2d(0..3usize, 0..n_fgs)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_label_formatter(&|idx| properties.get(*idx).unwrap_or(&"").to_string())
        .y_label_formatter(&|idx| fg_correlations.get(*idx).map(|(n, _, _, _)| n.clone()).unwrap_or_default())
        .x_label_style(("sans-serif", 13))
        .y_label_style(("sans-serif", 11))
        .draw()?;

    for (row, (_, r_qed, r_logp, r_sas)) in fg_correlations.iter().enumerate() {
        let vals = [*r_qed, *r_logp, *r_sas];
        for (col, &val) in vals.iter().enumerate() {
            let color = heatmap_color(val, -0.3, 0.3);
            chart.draw_series(std::iter::once(
                Rectangle::new([(col, row), (col + 1, row + 1)], color.filled())
            ))?;
            let text_color = if val.abs() > 0.20 { &WHITE } else { &BLACK };
            chart.draw_series(std::iter::once(
                Text::new(
                    format!("{:+.3}", val),
                    (col, row),
                    ("sans-serif", 11).into_font().color(text_color),
                )
            ))?;
        }
    }

    root.present()?;
    Ok("figures/fg_property_correlations.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 7. Silhouette & cluster quality comparison
// ═══════════════════════════════════════════════════

pub fn plot_cluster_quality_comparison(
    strata_metrics: &[(usize, f64, f64, f64, f64)],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/cluster_quality_comparison.svg", output_dir);
    ensure_parent_dir(&path);

    let root = SVGBackend::new(&path, (1100, 750)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 2));
    let titles = ["Silhouette Score", "Davies-Bouldin Index", "Quantization Error", "Gini Coefficient"];
    let y_labels = ["Score", "Index", "Error", "Coefficient"];
    let metrics: Vec<Vec<f64>> = vec![
        strata_metrics.iter().map(|m| m.1).collect(),
        strata_metrics.iter().map(|m| m.2).collect(),
        strata_metrics.iter().map(|m| m.3).collect(),
        strata_metrics.iter().map(|m| m.4).collect(),
    ];

    for (i, area) in areas.iter().enumerate() {
        let vals = &metrics[i];
        let min_v = vals.iter().copied().fold(f64::MAX, f64::min);
        let max_v = vals.iter().copied().fold(f64::MIN, f64::max);
        let range = (max_v - min_v).max(0.001);
        let y_lo = min_v - range * 0.15;
        let y_hi = max_v + range * 0.15;

        let mut chart = ChartBuilder::on(area)
            .caption(titles[i], ("sans-serif", 18).into_font())
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(55)
            .build_cartesian_2d(0..vals.len(), y_lo..y_hi)?;

        chart.configure_mesh()
            .x_desc("Stratum")
            .y_desc(y_labels[i])
            .x_label_style(("sans-serif", 12))
            .y_label_style(("sans-serif", 12))
            .draw()?;

        chart.draw_series(
            vals.iter().enumerate().map(|(j, &v)| {
                let color = STRATUM_COLORS[j % 5];
                let bar_bottom = if y_lo > 0.0 { y_lo } else { 0.0_f64.min(v) };
                Rectangle::new([(j, bar_bottom), (j + 1, v)], color.mix(0.8).filled())
            })
        )?;

        chart.draw_series(
            vals.iter().enumerate().map(|(j, &v)| {
                Text::new(
                    format!("{:.4}", v),
                    (j, v),
                    ("sans-serif", 11).into_font(),
                )
            })
        )?;
    }

    root.present()?;
    Ok("figures/cluster_quality_comparison.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 8. Reconstruction loss distribution
// ═══════════════════════════════════════════════════

pub fn plot_reconstruction_loss(
    losses: &[f32],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/reconstruction_loss_dist.svg", output_dir);
    ensure_parent_dir(&path);

    let float_losses: Vec<f64> = losses.iter().map(|&l| l as f64).collect();
    plot_histogram(
        &float_losses, &path,
        "VGAE Reconstruction Loss Distribution",
        "Loss", 60, RGBColor(214, 39, 40),
    )?;
    Ok("figures/reconstruction_loss_dist.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 9. Embedding dimension variance
// ═══════════════════════════════════════════════════

pub fn plot_embedding_variance(
    dim_variances: &[(usize, f64)],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/embedding_dim_variance.svg", output_dir);
    ensure_parent_dir(&path);

    let n = dim_variances.len();
    let max_var = dim_variances.iter().map(|d| d.1).fold(0.0_f64, f64::max) * 1.1;

    let root = SVGBackend::new(&path, (850, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Latent Dimension Variance", ("sans-serif", 22).into_font())
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(0..n, 0.0..max_var)?;

    chart.configure_mesh()
        .x_desc("Dimension")
        .y_desc("Variance")
        .x_label_style(("sans-serif", 14))
        .y_label_style(("sans-serif", 14))
        .draw()?;

    chart.draw_series(
        dim_variances.iter().enumerate().map(|(i, &(_, var))| {
            let intensity = (var / max_var).clamp(0.0, 1.0);
            let color = RGBColor(
                (31.0 + intensity * 183.0) as u8,
                (119.0 - intensity * 80.0) as u8,
                (180.0 - intensity * 140.0) as u8,
            );
            Rectangle::new([(i, 0.0), (i + 1, var)], color.filled())
        })
    )?;

    root.present()?;
    Ok("figures/embedding_dim_variance.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 10. U-Matrix heatmap (SOM topology)
// ═══════════════════════════════════════════════════

pub fn plot_umatrix_heatmaps(
    u_matrices: &[Vec<Vec<f64>>],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/umatrix_heatmaps.svg", output_dir);
    ensure_parent_dir(&path);

    let n_strata = u_matrices.len();
    let root = SVGBackend::new(&path, (280 * n_strata as u32 + 120, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, n_strata));
    let qed_labels = ["S0: Low QED", "S1: Med-Low", "S2: Medium", "S3: Med-High", "S4: High QED"];

    for (i, (area, u_matrix)) in areas.iter().zip(u_matrices.iter()).enumerate() {
        if u_matrix.is_empty() { continue; }
        let h = u_matrix.len();
        let w = u_matrix[0].len();

        let all_vals: Vec<f64> = u_matrix.iter().flatten().copied().collect();
        let u_min = all_vals.iter().copied().fold(f64::MAX, f64::min);
        let u_max = all_vals.iter().copied().fold(f64::MIN, f64::max);

        let mut chart = ChartBuilder::on(area)
            .caption(qed_labels.get(i).unwrap_or(&""), ("sans-serif", 16).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..w, 0..h)?;

        chart.configure_mesh()
            .disable_mesh()
            .x_desc("Grid X")
            .y_desc("Grid Y")
            .x_label_style(("sans-serif", 11))
            .y_label_style(("sans-serif", 11))
            .draw()?;

        for r in 0..h {
            for c in 0..w {
                let color = sequential_color(u_matrix[r][c], u_min, u_max);
                chart.draw_series(std::iter::once(
                    Rectangle::new([(c, r), (c + 1, r + 1)], color.filled())
                ))?;
            }
        }
    }

    root.present()?;
    Ok("figures/umatrix_heatmaps.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 11. FG enrichment heatmap (top clusters × FGs)
// ═══════════════════════════════════════════════════

pub fn plot_fg_enrichment_heatmap(
    cluster_fg_enrichments: &[(usize, Vec<(String, f64)>)],
    output_dir: &str,
    stratum_id: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let filename = format!("fg_enrichment_stratum_{}.svg", stratum_id);
    let path = format!("{}/figures/{}", output_dir, filename);
    ensure_parent_dir(&path);

    let mut fg_names: Vec<String> = Vec::new();
    let mut fg_set = std::collections::HashSet::new();
    for (_, fgs) in cluster_fg_enrichments {
        for (name, _) in fgs {
            if fg_set.insert(name.clone()) {
                fg_names.push(name.clone());
            }
        }
    }

    let n_clusters = cluster_fg_enrichments.len().min(20);
    let n_fgs = fg_names.len().min(15);
    if n_clusters == 0 || n_fgs == 0 {
        return Ok(format!("figures/{}", filename));
    }

    let root = SVGBackend::new(&path, (200 + n_fgs as u32 * 60, 120 + n_clusters as u32 * 32)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut data: Vec<Vec<f64>> = Vec::new();
    for (_, fgs) in cluster_fg_enrichments.iter().take(n_clusters) {
        let fg_map: HashMap<&str, f64> = fgs.iter().map(|(n, e)| (n.as_str(), *e)).collect();
        let row: Vec<f64> = fg_names.iter().take(n_fgs).map(|n| *fg_map.get(n.as_str()).unwrap_or(&1.0)).collect();
        data.push(row);
    }

    let all_vals: Vec<f64> = data.iter().flatten().copied().collect();
    let e_min = all_vals.iter().copied().fold(f64::MAX, f64::min).min(0.5);
    let e_max = all_vals.iter().copied().fold(f64::MIN, f64::max).max(2.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("FG Enrichment — Stratum {}", stratum_id), ("sans-serif", 18).into_font())
        .margin(15)
        .x_label_area_size(120)
        .y_label_area_size(80)
        .build_cartesian_2d(0..n_fgs, 0..n_clusters)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_label_formatter(&|idx| fg_names.get(*idx).cloned().unwrap_or_default())
        .y_label_formatter(&|idx| cluster_fg_enrichments.get(*idx).map(|(id, _)| format!("C{}", id)).unwrap_or_default())
        .x_label_style(("sans-serif", 11).into_font().transform(FontTransform::Rotate270))
        .y_label_style(("sans-serif", 11))
        .draw()?;

    for (row, row_data) in data.iter().enumerate() {
        for (col, &val) in row_data.iter().enumerate() {
            let log_val = if val > 0.0 { val.ln() } else { -2.0 };
            let color = heatmap_color(log_val, e_min.ln().min(-1.0), e_max.ln().max(1.0));
            chart.draw_series(std::iter::once(
                Rectangle::new([(col, row), (col + 1, row + 1)], color.filled())
            ))?;
        }
    }

    root.present()?;
    Ok(format!("figures/{}", filename))
}

// ═══════════════════════════════════════════════════
// 12. Inter-cluster distance matrix
// ═══════════════════════════════════════════════════

pub fn plot_distance_matrix(
    distances: &[(usize, usize, f64)],
    n_clusters: usize,
    output_dir: &str,
    stratum_id: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let filename = format!("cluster_distance_matrix_stratum_{}.svg", stratum_id);
    let path = format!("{}/figures/{}", output_dir, filename);
    ensure_parent_dir(&path);

    let n = n_clusters.min(20);
    if n < 2 { return Ok(format!("figures/{}", filename)); }

    let root = SVGBackend::new(&path, (120 + n as u32 * 35, 120 + n as u32 * 35)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut matrix = vec![vec![0.0_f64; n]; n];
    let max_dist = distances.iter().map(|d| d.2).fold(0.0_f64, f64::max);

    for &(a, b, d) in distances {
        if a < n && b < n {
            matrix[a][b] = d;
            matrix[b][a] = d;
        }
    }

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Inter-Cluster Distances — Stratum {}", stratum_id), ("sans-serif", 16).into_font())
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(0..n, 0..n)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Cluster ID")
        .y_desc("Cluster ID")
        .x_label_style(("sans-serif", 11))
        .y_label_style(("sans-serif", 11))
        .draw()?;

    for r in 0..n {
        for c in 0..n {
            let color = sequential_color(matrix[r][c], 0.0, max_dist);
            chart.draw_series(std::iter::once(
                Rectangle::new([(c, r), (c + 1, r + 1)], color.filled())
            ))?;
        }
    }

    root.present()?;
    Ok(format!("figures/{}", filename))
}

// ═══════════════════════════════════════════════════
// 13. Stratum property comparison
// ═══════════════════════════════════════════════════

pub fn plot_stratum_property_comparison(
    strata_stats: &[(usize, f64, f64, f64, f64, f64, f64)],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/stratum_property_comparison.svg", output_dir);
    ensure_parent_dir(&path);

    let root = SVGBackend::new(&path, (1200, 450)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 3));
    let property_names = ["QED (mean ± std)", "logP (mean ± std)", "SAS (mean ± std)"];
    let y_descs = ["QED", "logP", "SAS"];

    for (p_idx, area) in areas.iter().enumerate() {
        let (means, stds): (Vec<f64>, Vec<f64>) = strata_stats.iter().map(|s| {
            match p_idx {
                0 => (s.1, s.2),
                1 => (s.3, s.4),
                _ => (s.5, s.6),
            }
        }).unzip();

        let all_vals: Vec<f64> = means.iter().zip(stds.iter()).flat_map(|(&m, &s)| vec![m - s, m + s]).collect();
        let y_min = all_vals.iter().copied().fold(f64::MAX, f64::min) - 0.1;
        let y_max = all_vals.iter().copied().fold(f64::MIN, f64::max) + 0.1;

        let mut chart = ChartBuilder::on(area)
            .caption(property_names[p_idx], ("sans-serif", 18).into_font())
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(55)
            .build_cartesian_2d(0..means.len(), y_min..y_max)?;

        chart.configure_mesh()
            .x_desc("Stratum")
            .y_desc(y_descs[p_idx])
            .x_label_style(("sans-serif", 12))
            .y_label_style(("sans-serif", 12))
            .draw()?;

        chart.draw_series(
            means.iter().enumerate().map(|(i, &m)| {
                let color = STRATUM_COLORS[i % 5];
                let bar_bottom = if y_min > 0.0 { y_min } else { 0.0_f64.min(m) };
                Rectangle::new([(i, bar_bottom), (i + 1, m)], color.mix(0.7).filled())
            })
        )?;

        chart.draw_series(
            means.iter().zip(stds.iter()).enumerate().map(|(i, (&m, &s))| {
                PathElement::new(vec![(i, m - s), (i, m + s)], BLACK.stroke_width(2))
            })
        )?;
    }

    root.present()?;
    Ok("figures/stratum_property_comparison.svg".to_string())
}

// ═══════════════════════════════════════════════════
// 14. Molecule complexity scatter
// ═══════════════════════════════════════════════════

pub fn plot_molecule_complexity(
    atom_counts: &[usize],
    bond_counts: &[usize],
    qed_vals: &[f64],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/molecule_complexity.svg", output_dir);
    ensure_parent_dir(&path);

    let step = (atom_counts.len() / 8000).max(1);
    let points: Vec<(f64, f64, f64)> = atom_counts.iter()
        .zip(bond_counts.iter())
        .zip(qed_vals.iter())
        .step_by(step)
        .map(|((&a, &b), &q)| (a as f64, b as f64, q))
        .collect();

    let x_max = points.iter().map(|p| p.0).fold(0.0_f64, f64::max) + 2.0;
    let y_max = points.iter().map(|p| p.1).fold(0.0_f64, f64::max) + 2.0;

    let root = SVGBackend::new(&path, (900, 650)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Molecular Graph Complexity", ("sans-serif", 22).into_font())
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;

    chart.configure_mesh()
        .x_desc("Number of Atoms")
        .y_desc("Number of Bonds")
        .x_label_style(("sans-serif", 14))
        .y_label_style(("sans-serif", 14))
        .draw()?;

    chart.draw_series(
        points.iter().map(|&(a, b, q)| {
            let t = q.clamp(0.0, 1.0);
            let color = RGBColor(
                (255.0 * (1.0 - t)) as u8,
                (100.0 + 155.0 * t) as u8,
                (50.0 + 130.0 * t) as u8,
            );
            Circle::new((a, b), 4, color.mix(0.6).filled())
        })
    )?
    .label("Color: QED (red=low, green=high)")
    .legend(|(x, y)| Circle::new((x + 10, y), 5, RGBColor(44, 160, 44).filled()));

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()?;

    root.present()?;
    Ok("figures/molecule_complexity.svg".to_string())
}

// ═══════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════

fn compute_histogram(data: &[f64], num_bins: usize) -> (Vec<f64>, Vec<usize>) {
    if data.is_empty() {
        return (vec![0.0, 1.0], vec![0]);
    }
    let min_val = data.iter().copied().fold(f64::MAX, f64::min);
    let max_val = data.iter().copied().fold(f64::MIN, f64::max);
    let range = (max_val - min_val).max(1e-10);
    let bin_width = range / num_bins as f64;

    let mut bins = Vec::with_capacity(num_bins + 1);
    for i in 0..=num_bins {
        bins.push(min_val + i as f64 * bin_width);
    }

    let mut counts = vec![0usize; num_bins];
    for &val in data {
        let idx = ((val - min_val) / bin_width).floor() as usize;
        let idx = idx.min(num_bins - 1);
        counts[idx] += 1;
    }

    (bins, counts)
}

#[allow(dead_code)]
fn simple_pca_2d(embeddings: &[Vec<f32>]) -> (Vec<f64>, Vec<f64>) {
    let n = embeddings.len();
    if n == 0 || embeddings[0].is_empty() {
        return (vec![], vec![]);
    }
    let d = embeddings[0].len();

    let mut means = vec![0.0f64; d];
    for emb in embeddings {
        for (j, &v) in emb.iter().enumerate() {
            means[j] += v as f64;
        }
    }
    for m in &mut means { *m /= n as f64; }

    let centered: Vec<Vec<f64>> = embeddings.iter()
        .map(|e| e.iter().zip(means.iter()).map(|(&v, &m)| v as f64 - m).collect())
        .collect();

    let pc1 = power_iteration(&centered, d, None, 50);
    let pc2 = power_iteration(&centered, d, Some(&pc1), 50);

    let proj1: Vec<f64> = centered.iter().map(|row| row.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum()).collect();
    let proj2: Vec<f64> = centered.iter().map(|row| row.iter().zip(pc2.iter()).map(|(a, b)| a * b).sum()).collect();

    (proj1, proj2)
}

#[allow(dead_code)]
fn power_iteration(data: &[Vec<f64>], d: usize, deflate: Option<&[f64]>, iterations: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut v: Vec<f64> = (0..d).map(|_| rng.gen::<f64>() - 0.5).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }

    for _ in 0..iterations {
        let w: Vec<f64> = data.iter().map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()).collect();
        let mut new_v = vec![0.0; d];
        for (row, &wi) in data.iter().zip(w.iter()) {
            for (j, &val) in row.iter().enumerate() {
                new_v[j] += val * wi;
            }
        }

        if let Some(prev) = deflate {
            let dot: f64 = new_v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            for (j, p) in prev.iter().enumerate() {
                new_v[j] -= dot * p;
            }
        }

        let norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 { break; }
        v = new_v.iter().map(|x| x / norm).collect();
    }

    v
}

fn ensure_parent_dir(path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
}

// ═══════════════════════════════════════════════════
// Master generate function
// ═══════════════════════════════════════════════════

pub struct VisualizationData {
    pub qed_vals: Vec<f64>,
    pub logp_vals: Vec<f64>,
    pub sas_vals: Vec<f64>,
    pub atom_counts: Vec<usize>,
    pub bond_counts: Vec<usize>,
    pub embeddings: Vec<Vec<f32>>,
    pub stratum_labels: Vec<usize>,
    pub recon_losses: Vec<f32>,
    pub dim_correlations: Vec<(usize, f64, f64, f64)>,
    pub dim_variances: Vec<(usize, f64)>,
    pub fg_prevalence: Vec<(String, f64)>,
    pub fg_property_corr: Vec<(String, f64, f64, f64)>,
    pub strata_quality: Vec<(usize, f64, f64, f64, f64)>,
    pub strata_cluster_sizes: Vec<Vec<usize>>,
    pub strata_property_stats: Vec<(usize, f64, f64, f64, f64, f64, f64)>,
    pub u_matrices: Vec<Vec<Vec<f64>>>,
    pub strata_fg_enrichments: Vec<(usize, Vec<(usize, Vec<(String, f64)>)>)>,
    pub strata_distances: Vec<(usize, Vec<(usize, usize, f64)>, usize)>,
}

pub fn generate_all_figures(data: &VisualizationData, output_dir: &str) -> Vec<(String, String)> {
    let mut figures: Vec<(String, String)> = Vec::new();

    log::info!("Generating figures...");

    match plot_property_distributions(&data.qed_vals, &data.logp_vals, &data.sas_vals, output_dir) {
        Ok(paths) => {
            for p in &paths { figures.push((p.clone(), "Property distribution".to_string())); }
            log::info!("  ✓ Property distributions ({} plots)", paths.len());
        }
        Err(e) => log::warn!("  ✗ Property distributions: {}", e),
    }

    match plot_fg_prevalence(&data.fg_prevalence, output_dir) {
        Ok(p) => { figures.push((p, "Functional group prevalence".to_string())); log::info!("  ✓ FG prevalence"); }
        Err(e) => log::warn!("  ✗ FG prevalence: {}", e),
    }

    match plot_latent_space_umap(&data.embeddings, &data.stratum_labels, output_dir) {
        Ok(p) => { figures.push((p, "Latent space UMAP projection".to_string())); log::info!("  ✓ Latent space UMAP"); }
        Err(e) => log::warn!("  ✗ Latent space UMAP: {}", e),
    }

    match plot_cluster_size_distributions(&data.strata_cluster_sizes, output_dir) {
        Ok(p) => { figures.push((p, "Cluster size distributions".to_string())); log::info!("  ✓ Cluster sizes"); }
        Err(e) => log::warn!("  ✗ Cluster sizes: {}", e),
    }

    match plot_dim_property_heatmap(&data.dim_correlations, output_dir) {
        Ok(p) => { figures.push((p, "Dimension-property heatmap".to_string())); log::info!("  ✓ Dim-property heatmap"); }
        Err(e) => log::warn!("  ✗ Dim-property heatmap: {}", e),
    }

    match plot_fg_property_correlations(&data.fg_property_corr, output_dir) {
        Ok(p) => { figures.push((p, "FG-property correlations".to_string())); log::info!("  ✓ FG-property correlations"); }
        Err(e) => log::warn!("  ✗ FG-property correlations: {}", e),
    }

    match plot_cluster_quality_comparison(&data.strata_quality, output_dir) {
        Ok(p) => { figures.push((p, "Cluster quality comparison".to_string())); log::info!("  ✓ Cluster quality"); }
        Err(e) => log::warn!("  ✗ Cluster quality: {}", e),
    }

    match plot_reconstruction_loss(&data.recon_losses, output_dir) {
        Ok(p) => { figures.push((p, "Reconstruction loss".to_string())); log::info!("  ✓ Reconstruction loss"); }
        Err(e) => log::warn!("  ✗ Reconstruction loss: {}", e),
    }

    match plot_embedding_variance(&data.dim_variances, output_dir) {
        Ok(p) => { figures.push((p, "Embedding variance".to_string())); log::info!("  ✓ Embedding variance"); }
        Err(e) => log::warn!("  ✗ Embedding variance: {}", e),
    }

    match plot_umatrix_heatmaps(&data.u_matrices, output_dir) {
        Ok(p) => { figures.push((p, "SOM U-matrix".to_string())); log::info!("  ✓ U-matrix heatmaps"); }
        Err(e) => log::warn!("  ✗ U-matrix: {}", e),
    }

    match plot_stratum_property_comparison(&data.strata_property_stats, output_dir) {
        Ok(p) => { figures.push((p, "Stratum properties".to_string())); log::info!("  ✓ Stratum properties"); }
        Err(e) => log::warn!("  ✗ Stratum properties: {}", e),
    }

    match plot_molecule_complexity(&data.atom_counts, &data.bond_counts, &data.qed_vals, output_dir) {
        Ok(p) => { figures.push((p, "Molecule complexity".to_string())); log::info!("  ✓ Molecule complexity"); }
        Err(e) => log::warn!("  ✗ Molecule complexity: {}", e),
    }

    for (stratum_id, cluster_enrichments) in &data.strata_fg_enrichments {
        match plot_fg_enrichment_heatmap(cluster_enrichments, output_dir, *stratum_id) {
            Ok(p) => { figures.push((p, format!("FG enrichment S{}", stratum_id))); log::info!("  ✓ FG enrichment (S{})", stratum_id); }
            Err(e) => log::warn!("  ✗ FG enrichment S{}: {}", stratum_id, e),
        }
    }

    for (stratum_id, distances, n_clusters) in &data.strata_distances {
        match plot_distance_matrix(distances, *n_clusters, output_dir, *stratum_id) {
            Ok(p) => { figures.push((p, format!("Distances S{}", stratum_id))); log::info!("  ✓ Distance matrix (S{})", stratum_id); }
            Err(e) => log::warn!("  ✗ Distance matrix S{}: {}", stratum_id, e),
        }
    }

    log::info!("Generated {} figures total", figures.len());
    figures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_histogram_basic() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (bins, counts) = compute_histogram(&data, 5);
        assert_eq!(bins.len(), 6); // num_bins + 1 edges
        assert_eq!(counts.len(), 5);
        let total: usize = counts.iter().sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_compute_histogram_single_value() {
        let data = vec![5.0; 100];
        let (bins, counts) = compute_histogram(&data, 10);
        assert_eq!(counts.len(), 10);
        let total: usize = counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_compute_histogram_empty() {
        let data: Vec<f64> = vec![];
        let (bins, counts) = compute_histogram(&data, 10);
        assert_eq!(bins.len(), 2);
        assert_eq!(counts, vec![0]);
    }

    #[test]
    fn test_simple_pca_2d_dimensions() {
        let embeddings: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32 * 0.1, (i as f32).sin(), (i as f32).cos(), i as f32 * 0.05])
            .collect();
        let (pc1, pc2) = simple_pca_2d(&embeddings);
        assert_eq!(pc1.len(), 100);
        assert_eq!(pc2.len(), 100);
    }

    #[test]
    fn test_simple_pca_2d_empty() {
        let embeddings: Vec<Vec<f32>> = vec![];
        let (pc1, pc2) = simple_pca_2d(&embeddings);
        assert!(pc1.is_empty());
        assert!(pc2.is_empty());
    }

    #[test]
    fn test_power_iteration_unit_vector() {
        let data = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ];
        let v = power_iteration(&data, 2, None, 30);
        assert_eq!(v.len(), 2);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "Eigenvector should be unit length");
    }

    #[test]
    fn test_heatmap_color_extremes() {
        let low = heatmap_color(0.0, 0.0, 1.0);
        let high = heatmap_color(1.0, 0.0, 1.0);
        let mid = heatmap_color(0.5, 0.0, 1.0);
        // Low should be blueish
        assert!(low.2 > low.0, "Low value should be more blue than red");
        // High should be reddish
        assert!(high.0 > high.2, "High value should be more red than blue");
        // Mid should be near white
        assert!(mid.0 > 200 && mid.1 > 200 && mid.2 > 200, "Mid value should be near white");
    }

    #[test]
    fn test_sequential_color_range() {
        let low = sequential_color(0.0, 0.0, 1.0);
        let high = sequential_color(1.0, 0.0, 1.0);
        // Low should be near white
        assert!(low.0 > 240 && low.1 > 240 && low.2 > 240);
        // High should be dark
        assert!(high.0 < 50 && high.1 < 100);
    }

    #[test]
    fn test_heatmap_color_equal_range() {
        let c = heatmap_color(5.0, 5.0, 5.0);
        // With equal min/max, should return mid-range color
        assert!(c.0 > 100);
    }

    #[test]
    fn test_ensure_parent_dir() {
        let test_path = "/tmp/fga_test_viz_dir/sub/file.svg";
        ensure_parent_dir(test_path);
        assert!(std::path::Path::new("/tmp/fga_test_viz_dir/sub").exists());
        let _ = std::fs::remove_dir_all("/tmp/fga_test_viz_dir");
    }

    #[test]
    fn test_plot_histogram_creates_file() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let path = "/tmp/fga_test_histogram.svg";
        let result = plot_histogram(&data, path, "Test", "X", 5, RGBColor(31, 119, 180));
        assert!(result.is_ok());
        assert!(std::path::Path::new(path).exists());
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("<svg"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_plot_fg_prevalence_creates_file() {
        let fg_data = vec![
            ("Phenyl".to_string(), 83.0),
            ("Amide".to_string(), 68.0),
            ("Ether".to_string(), 37.0),
        ];
        let dir = "/tmp/fga_test_fg_prev";
        let _ = std::fs::create_dir_all(format!("{}/figures", dir));
        let result = plot_fg_prevalence(&fg_data, dir);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_plot_embedding_variance_creates_file() {
        let variances: Vec<(usize, f64)> = (0..16).map(|i| (i, 0.01 * (i + 1) as f64)).collect();
        let dir = "/tmp/fga_test_emb_var";
        let _ = std::fs::create_dir_all(format!("{}/figures", dir));
        let result = plot_embedding_variance(&variances, dir);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_plot_cluster_quality_creates_file() {
        let metrics = vec![
            (0, -0.02, 3.35, 0.028, 0.38),
            (1, -0.01, 3.35, 0.029, 0.38),
            (2, 0.00, 3.32, 0.031, 0.40),
        ];
        let dir = "/tmp/fga_test_cq";
        let _ = std::fs::create_dir_all(format!("{}/figures", dir));
        let result = plot_cluster_quality_comparison(&metrics, dir);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_plot_latent_space_umap_creates_file() {
        let embeddings: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.1; 4])
            .collect();
        let labels = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                          0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                          0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                          0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                          0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let dir = "/tmp/fga_test_umap";
        let _ = std::fs::create_dir_all(format!("{}/figures", dir));
        let result = plot_latent_space_umap(&embeddings, &labels, dir);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_plot_distance_matrix_creates_file() {
        let distances = vec![
            (0, 1, 0.5), (0, 2, 0.8), (1, 2, 0.3),
        ];
        let dir = "/tmp/fga_test_dist";
        let _ = std::fs::create_dir_all(format!("{}/figures", dir));
        let result = plot_distance_matrix(&distances, 3, dir, 0);
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }
}
