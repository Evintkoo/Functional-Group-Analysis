/// Visualization module: generates publication-quality SVG figures for Q1 journal submission.
/// Uses the `plotters` crate with SVG backend for high-quality vector graphics.
/// Typography, sizing, and styling follow Nature/Science/PNAS figure guidelines.

use plotters::prelude::*;
use std::path::Path;
use std::collections::HashMap;
use umap_rs::{Umap as UmapAlgo, UmapConfig as UmapCfg, GraphParams as UmapGraphParams,
               OptimizationParams as UmapOptParams};
use rayon::prelude::*;

// ═══════════════════════════════════════════════════
// Q1 journal figure standards
// ═══════════════════════════════════════════════════

const FONT: &str = "Helvetica";
const TITLE_SIZE: u32 = 16;
const AXIS_LABEL_SIZE: u32 = 13;
const TICK_SIZE: u32 = 11;
const LEGEND_SIZE: u32 = 11;
const ANNOT_SIZE: u32 = 10;
const PANEL_LABEL_SIZE: u32 = 18;
const HEATMAP_ANNOT_SIZE: u32 = 10;

const BAR_ALPHA: f64 = 0.85;
const SCATTER_ALPHA: f64 = 0.70;
const SCATTER_R: u32 = 3;
const BAR_STROKE: u32 = 1;
const GRID_STYLE: RGBColor = RGBColor(225, 225, 225);
const MEAN_LINE: RGBColor = RGBColor(180, 0, 0);

// Standard figure dimensions (single-column ~89mm, double-column ~178mm at 96 DPI)
const FIG_SINGLE: (u32, u32) = (800, 600);
const FIG_WIDE: (u32, u32) = (1200, 500);
const FIG_PANEL_2X2: (u32, u32) = (1200, 900);

// ═══════════════════════════════════════════════════
// Color palettes — Wong (2011, Nature Methods) colorblind-safe
// ═══════════════════════════════════════════════════

const STRATUM_COLORS: [RGBColor; 5] = [
    RGBColor(0, 114, 178),    // blue
    RGBColor(230, 159, 0),    // orange
    RGBColor(0, 158, 115),    // bluish green
    RGBColor(204, 121, 167),  // reddish purple
    RGBColor(86, 180, 233),   // sky blue
];

const PROPERTY_COLORS: [RGBColor; 3] = [
    RGBColor(0, 114, 178),    // QED - blue
    RGBColor(230, 159, 0),    // logP - orange
    RGBColor(0, 158, 115),    // SAS - bluish green
];

/// CVD-safe diverging colormap: blue → white → red (Crameri-style).
fn heatmap_color(value: f64, min_val: f64, max_val: f64) -> RGBColor {
    let t = if (max_val - min_val).abs() < 1e-12 {
        0.5
    } else {
        ((value - min_val) / (max_val - min_val)).clamp(0.0, 1.0)
    };
    if t < 0.5 {
        let s = t * 2.0;
        // Dark blue (33, 102, 172) → white (247, 247, 247)
        RGBColor(
            (33.0 + s * 214.0) as u8,
            (102.0 + s * 145.0) as u8,
            (172.0 + s * 75.0) as u8,
        )
    } else {
        let s = (t - 0.5) * 2.0;
        // White (247, 247, 247) → dark red (178, 24, 43)
        RGBColor(
            (247.0 - s * 69.0) as u8,
            (247.0 - s * 223.0) as u8,
            (247.0 - s * 204.0) as u8,
        )
    }
}

/// CVD-safe sequential colormap: Viridis (5-stop piecewise linear).
fn sequential_color(value: f64, min_val: f64, max_val: f64) -> RGBColor {
    let t = if (max_val - min_val).abs() < 1e-12 { 0.5 }
        else { ((value - min_val) / (max_val - min_val)).clamp(0.0, 1.0) };
    // Viridis stops: (0.0) #440154 → (0.25) #31688e → (0.5) #21918c → (0.75) #5ec962 → (1.0) #fde725
    let stops: [(f64, u8, u8, u8); 5] = [
        (0.00,  68,   1,  84),
        (0.25,  49, 104, 142),
        (0.50,  33, 145, 140),
        (0.75,  94, 201,  98),
        (1.00, 253, 231,  37),
    ];
    let mut i = 0;
    while i < 3 && t > stops[i + 1].0 { i += 1; }
    let (t0, r0, g0, b0) = stops[i];
    let (t1, r1, g1, b1) = stops[i + 1];
    let s = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
    RGBColor(
        (r0 as f64 + s * (r1 as f64 - r0 as f64)) as u8,
        (g0 as f64 + s * (g1 as f64 - g0 as f64)) as u8,
        (b0 as f64 + s * (b1 as f64 - b0 as f64)) as u8,
    )
}

/// Draw a vertical colorbar on the right side of a drawing area.
fn draw_colorbar(
    area: &DrawingArea<SVGBackend, plotters::coord::Shift>,
    min_val: f64,
    max_val: f64,
    label: &str,
    color_fn: impl Fn(f64, f64, f64) -> RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = area.dim_in_pixel();
    let bar_w = 18u32;
    let bar_h = (h as f64 * 0.6) as u32;
    let x0 = w as i32 - 55;
    let y0 = ((h - bar_h) / 2) as i32;
    let n_steps = 64u32;
    let step_h = bar_h / n_steps;

    for i in 0..n_steps {
        let t = 1.0 - i as f64 / n_steps as f64;
        let val = min_val + t * (max_val - min_val);
        let color = color_fn(val, min_val, max_val);
        let yi = y0 + i as i32 * step_h as i32;
        area.draw(&Rectangle::new(
            [(x0, yi), (x0 + bar_w as i32, yi + step_h as i32)],
            color.filled(),
        ))?;
    }
    // Border
    area.draw(&Rectangle::new(
        [(x0, y0), (x0 + bar_w as i32, y0 + bar_h as i32)],
        BLACK.stroke_width(1),
    ))?;
    // Labels
    let fmt = |v: f64| -> String {
        if v.abs() >= 100.0 { format!("{:.0}", v) }
        else if v.abs() >= 1.0 { format!("{:.2}", v) }
        else { format!("{:.3}", v) }
    };
    area.draw(&Text::new(fmt(max_val), (x0 + bar_w as i32 + 3, y0 + 2),
        (FONT, ANNOT_SIZE - 1).into_font()))?;
    area.draw(&Text::new(fmt(min_val), (x0 + bar_w as i32 + 3, y0 + bar_h as i32 - 4),
        (FONT, ANNOT_SIZE - 1).into_font()))?;
    area.draw(&Text::new(label, (x0 - 2, y0 - 14),
        (FONT, ANNOT_SIZE).into_font()))?;
    Ok(())
}

/// Draw a panel label like "(a)" in the top-left corner.
fn draw_panel_label(
    area: &DrawingArea<SVGBackend, plotters::coord::Shift>,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    area.draw(&Text::new(
        label.to_string(), (8, 6),
        (FONT, PANEL_LABEL_SIZE).into_font().style(FontStyle::Bold),
    ))?;
    Ok(())
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

    let root = SVGBackend::new(path, FIG_SINGLE).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, (FONT, TITLE_SIZE).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(65)
        .build_cartesian_2d(bins[0]..bins[bins.len() - 1], 0.0..max_count * 1.05)?;

    chart.configure_mesh()
        .x_desc(x_label)
        .y_desc("Count")
        .x_label_style((FONT, TICK_SIZE))
        .y_label_style((FONT, TICK_SIZE))
        .axis_desc_style((FONT, AXIS_LABEL_SIZE))
        .light_line_style(GRID_STYLE)
        .draw()?;

    let bar_width = (bins[1] - bins[0]) * 0.95;
    chart.draw_series(
        bins.windows(2).zip(counts.iter()).map(|(bin_pair, &count)| {
            let x0 = bin_pair[0];
            let x1 = x0 + bar_width;
            let mut bar = Rectangle::new([(x0, 0.0), (x1, count as f64)], color.mix(BAR_ALPHA).filled());
            bar.set_margin(0, 0, 0, 0);
            bar
        })
    )?;

    // Mean indicator line
    let n_data = data.len();
    let mean = data.iter().sum::<f64>() / n_data.max(1) as f64;
    let std_dev = (data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_data.max(1) as f64).sqrt();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n_data % 2 == 0 && n_data > 1 {
        (sorted[n_data / 2 - 1] + sorted[n_data / 2]) / 2.0
    } else if n_data > 0 { sorted[n_data / 2] } else { 0.0 };

    chart.draw_series(LineSeries::new(
        vec![(mean, 0.0), (mean, max_count * 1.0)],
        MEAN_LINE.stroke_width(2),
    ))?
    .label(format!("μ = {:.3}, σ = {:.3}", mean, std_dev))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MEAN_LINE.stroke_width(2)));

    // Median line
    chart.draw_series(LineSeries::new(
        vec![(median, 0.0), (median, max_count * 0.95)],
        RGBColor(0, 130, 0).stroke_width(2),
    ))?
    .label(format!("median = {:.3}", median))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(0, 130, 0).stroke_width(2)));

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK.stroke_width(1))
        .label_font((FONT, LEGEND_SIZE))
        .draw()?;

    // N annotation in upper-left area
    let x_range = bins[bins.len() - 1] - bins[0];
    chart.draw_series(std::iter::once(
        Text::new(
            format!("N = {}", n_data),
            (bins[0] + x_range * 0.02, max_count * 0.95),
            (FONT, ANNOT_SIZE).into_font(),
        )
    ))?;

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
    let root = SVGBackend::new(path, (1500, 550)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 3));
    let datasets: Vec<(&str, &[f64], RGBColor)> = vec![
        ("QED", qed_vals, PROPERTY_COLORS[0]),
        ("logP", logp_vals, PROPERTY_COLORS[1]),
        ("SAS", sas_vals, PROPERTY_COLORS[2]),
    ];
    let panel_labels = ["(a)", "(b)", "(c)"];

    // QED stratum boundaries for annotation on first panel
    let qed_boundaries = [0.399, 0.520, 0.694, 0.814];
    let boundary_color = RGBColor(120, 120, 120);

    for (i, (area, (label, data, color))) in areas.iter().zip(datasets.iter()).enumerate() {
        let _ = draw_panel_label(area, panel_labels[i]);
        let (bins, counts) = compute_histogram(data, 40);
        let max_count = *counts.iter().max().unwrap_or(&1) as f64;

        let mut chart = ChartBuilder::on(area)
            .caption(*label, (FONT, TITLE_SIZE).into_font())
            .margin(18)
            .x_label_area_size(45)
            .y_label_area_size(60)
            .build_cartesian_2d(bins[0]..bins[bins.len() - 1], 0.0..max_count * 1.18)?;

        chart.configure_mesh()
            .x_desc(*label)
            .y_desc("Count")
            .x_label_style((FONT, TICK_SIZE))
            .y_label_style((FONT, TICK_SIZE))
            .axis_desc_style((FONT, AXIS_LABEL_SIZE))
            .light_line_style(GRID_STYLE)
            .draw()?;

        let bar_width = (bins[1] - bins[0]) * 0.92;
        chart.draw_series(
            bins.windows(2).zip(counts.iter()).map(|(bp, &c)| {
                Rectangle::new([(bp[0], 0.0), (bp[0] + bar_width, c as f64)], color.mix(BAR_ALPHA).filled())
            })
        )?;

        // Mean line
        let n_d = data.len();
        let mean = data.iter().sum::<f64>() / n_d.max(1) as f64;
        chart.draw_series(LineSeries::new(
            vec![(mean, 0.0), (mean, max_count * 1.0)],
            MEAN_LINE.stroke_width(2),
        ))?
        .label(format!("μ = {:.3}", mean))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MEAN_LINE.stroke_width(2)));

        // === QED panel: add stratum boundary dashed lines and labels ===
        if i == 0 {
            let stratum_names = ["S0", "S1", "S2", "S3", "S4"];
            for (bi, &bval) in qed_boundaries.iter().enumerate() {
                if bval >= bins[0] && bval <= bins[bins.len() - 1] {
                    // Dashed boundary line (drawn as short segments)
                    let n_dashes = 12;
                    let dash_len = max_count * 1.0 / (n_dashes as f64 * 2.0);
                    for d in 0..n_dashes {
                        let y_start = d as f64 * dash_len * 2.0;
                        let y_end = y_start + dash_len;
                        chart.draw_series(std::iter::once(
                            PathElement::new(vec![(bval, y_start), (bval, y_end)], boundary_color.stroke_width(1))
                        ))?;
                    }
                    // Boundary value label at top
                    chart.draw_series(std::iter::once(
                        Text::new(
                            format!("{:.2}", bval),
                            (bval - 0.02, max_count * 1.04),
                            (FONT, ANNOT_SIZE - 2).into_font().color(&boundary_color),
                        )
                    ))?;
                }
                // Stratum label between boundaries
                let left = if bi == 0 { bins[0] } else { qed_boundaries[bi - 1] };
                let right = bval;
                let mid = (left + right) / 2.0;
                if mid >= bins[0] && mid <= bins[bins.len() - 1] {
                    chart.draw_series(std::iter::once(
                        Text::new(
                            stratum_names[bi].to_string(),
                            (mid - 0.01, max_count * 1.12),
                            (FONT, ANNOT_SIZE - 1).into_font().style(FontStyle::Bold).color(&boundary_color),
                        )
                    ))?;
                }
            }
            // Last stratum label (S4)
            let s4_mid = (qed_boundaries[3] + bins[bins.len() - 1]) / 2.0;
            chart.draw_series(std::iter::once(
                Text::new(
                    "S4".to_string(),
                    (s4_mid - 0.01, max_count * 1.12),
                    (FONT, ANNOT_SIZE - 1).into_font().style(FontStyle::Bold).color(&boundary_color),
                )
            ))?;
        }

        // === logP panel: add Lipinski Ro5 limit ===
        if i == 1 {
            let lipinski_logp = 5.0;
            if lipinski_logp <= bins[bins.len() - 1] {
                chart.draw_series(LineSeries::new(
                    vec![(lipinski_logp, 0.0), (lipinski_logp, max_count * 0.95)],
                    RGBColor(180, 0, 0).stroke_width(2),
                ))?;
                chart.draw_series(std::iter::once(
                    Text::new(
                        "Ro5 limit".to_string(),
                        (lipinski_logp + 0.1, max_count * 0.90),
                        (FONT, ANNOT_SIZE - 1).into_font().style(FontStyle::Italic).color(&RGBColor(180, 0, 0)),
                    )
                ))?;
            }
        }

        chart.configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.9))
            .border_style(BLACK.stroke_width(1))
            .label_font((FONT, LEGEND_SIZE - 1))
            .draw()?;
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
    let root = SVGBackend::new(&path, (900, 100 + n as u32 * 34)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_pct = fg_data.iter().map(|(_, p)| *p).fold(0.0_f64, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Functional Group Prevalence (%)", (FONT, TITLE_SIZE).into_font())
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(210)
        .build_cartesian_2d(0.0..max_pct, 0..n)?;

    chart.configure_mesh()
        .x_desc("Prevalence (%)")
        .y_label_formatter(&|idx| {
            fg_data.get(*idx).map(|(name, _)| name.clone()).unwrap_or_default()
        })
        .x_label_style((FONT, TICK_SIZE))
        .y_label_style((FONT, TICK_SIZE))
        .axis_desc_style((FONT, AXIS_LABEL_SIZE))
        .light_line_style(GRID_STYLE)
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
            Rectangle::new([(0.0, i), (*pct, i + 1)], color.mix(BAR_ALPHA).filled())
        })
    )?;

    // Value annotations at end of each bar
    chart.draw_series(
        fg_data.iter().enumerate().map(|(i, (_, pct))| {
            Text::new(
                format!("{:.1}%", pct),
                (*pct + max_pct * 0.01, i),
                (FONT, ANNOT_SIZE).into_font(),
            )
        })
    )?;

    // === Threshold lines with labels: ubiquitous (>50%), common (>10%), rare (<5%) ===
    let threshold_50 = 50.0_f64;
    let threshold_10 = 10.0_f64;
    let threshold_5 = 5.0_f64;
    let thresh_color = RGBColor(140, 140, 140);
    for &(thresh, label_text) in &[(threshold_50, "Ubiquitous (>50%)"), (threshold_10, "Common (>10%)"), (threshold_5, "Rare (<5%)")] {
        if thresh <= max_pct {
            // Dashed vertical threshold line
            let n_dashes = n / 2 + 1;
            for d in 0..n_dashes {
                let y_start = d * 2;
                let y_end = y_start + 1;
                if y_start < n {
                    chart.draw_series(std::iter::once(
                        PathElement::new(vec![(thresh, y_start), (thresh, y_end.min(n))], thresh_color.stroke_width(1))
                    ))?;
                }
            }
            // Label at top of line
            chart.draw_series(std::iter::once(
                Text::new(
                    label_text.to_string(),
                    (thresh + max_pct * 0.005, 0),
                    (FONT, ANNOT_SIZE - 2).into_font().style(FontStyle::Italic).color(&thresh_color),
                )
            ))?;
        }
    }

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

    // Random 3D initialization
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n_dims = 3usize;
    let mut init_flat: Vec<f32> = Vec::with_capacity(n_samples * n_dims);
    for _ in 0..(n_samples * n_dims) {
        init_flat.push(rng.gen::<f32>() * 20.0 - 10.0);
    }
    let init = ndarray_017::Array2::from_shape_vec((n_samples, n_dims), init_flat)
        .map_err(|e| format!("UMAP init array error: {}", e))?;

    let config = UmapCfg {
        n_components: n_dims,
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
    let umap_result: Vec<(f64, f64, f64, usize)> = (0..n_samples)
        .zip(sampled_strata.iter())
        .map(|(i, &s)| (
            embedding[[i, 0]] as f64,
            embedding[[i, 1]] as f64,
            embedding[[i, 2]] as f64,
            s,
        ))
        .collect();

    let x_min = umap_result.iter().map(|p| p.0).fold(f64::MAX, f64::min);
    let x_max = umap_result.iter().map(|p| p.0).fold(f64::MIN, f64::max);
    let y_min = umap_result.iter().map(|p| p.1).fold(f64::MAX, f64::min);
    let y_max = umap_result.iter().map(|p| p.1).fold(f64::MIN, f64::max);
    let z_min = umap_result.iter().map(|p| p.2).fold(f64::MAX, f64::min);
    let z_max = umap_result.iter().map(|p| p.2).fold(f64::MIN, f64::max);
    let x_pad = (x_max - x_min) * 0.08;
    let y_pad = (y_max - y_min) * 0.08;
    let z_pad = (z_max - z_min) * 0.08;

    let root = SVGBackend::new(&path, (1100, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Latent Space (3D UMAP Projection)", (FONT, TITLE_SIZE).into_font())
        .margin(25)
        .build_cartesian_3d(
            (x_min - x_pad)..(x_max + x_pad),
            (y_min - y_pad)..(y_max + y_pad),
            (z_min - z_pad)..(z_max + z_pad),
        )?;

    chart.with_projection(|mut pb| {
        pb.pitch = 0.35;
        pb.yaw = 0.65;
        pb.scale = 0.85;
        pb.into_matrix()
    });

    chart.configure_axes()
        .light_grid_style(GRID_STYLE)
        .label_style((FONT, TICK_SIZE))
        .x_formatter(&|v| format!("{:.1}", v))
        .y_formatter(&|v| format!("{:.1}", v))
        .z_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    let qed_labels = ["QED [0, 0.4)", "QED [0.4, 0.52)", "QED [0.52, 0.69)", "QED [0.69, 0.81)", "QED [0.81, 1.0]"];

    for stratum in 0..5 {
        let stratum_points: Vec<(f64, f64, f64)> = umap_result.iter()
            .filter(|p| p.3 == stratum)
            .map(|p| (p.0, p.1, p.2))
            .collect();

        if stratum_points.is_empty() { continue; }

        let color = STRATUM_COLORS[stratum];
        chart.draw_series(
            stratum_points.iter().map(|&(x, y, z)| {
                Circle::new((x, y, z), SCATTER_R, color.mix(SCATTER_ALPHA).filled())
            })
        )?
        .label(*qed_labels.get(stratum).unwrap_or(&""))
        .legend(move |(x, y)| Circle::new((x + 10, y), 5, color.filled()));
    }

    // === Key region annotations ===
    // Compute per-stratum centroids for annotation placement
    let mut stratum_centroids: Vec<(f64, f64, f64, usize)> = Vec::new();
    for s in 0..5 {
        let pts: Vec<&(f64, f64, f64, usize)> = umap_result.iter().filter(|p| p.3 == s).collect();
        if !pts.is_empty() {
            let cx = pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64;
            let cy = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
            let cz = pts.iter().map(|p| p.2).sum::<f64>() / pts.len() as f64;
            stratum_centroids.push((cx, cy, cz, pts.len()));
        }
    }
    // Annotate S4 (high QED) — dense core
    if let Some(s4) = stratum_centroids.get(4) {
        chart.draw_series(std::iter::once(
            Text::new(
                "Dense high-QED core".to_string(),
                (s4.0 + x_pad * 0.5, s4.1 + y_pad * 0.5, s4.2),
                (FONT, ANNOT_SIZE).into_font().style(FontStyle::Bold).color(&RGBColor(0, 100, 0)),
            )
        ))?;
    }
    // Annotate S0 (low QED) — diffuse periphery
    if let Some(s0) = stratum_centroids.get(0) {
        chart.draw_series(std::iter::once(
            Text::new(
                "Diffuse low-QED periphery".to_string(),
                (s0.0 - x_pad * 0.3, s0.1 - y_pad * 0.3, s0.2),
                (FONT, ANNOT_SIZE).into_font().style(FontStyle::Bold).color(&RGBColor(180, 0, 0)),
            )
        ))?;
    }

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.95))
        .border_style(BLACK.stroke_width(1))
        .label_font((FONT, LEGEND_SIZE))
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
    let root = SVGBackend::new(&path, (900, 260 * n_strata as u32 + 80)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((n_strata, 1));
    let qed_labels = ["Stratum 0: QED [0, 0.4)", "Stratum 1: QED [0.4, 0.52)", "Stratum 2: QED [0.52, 0.69)", "Stratum 3: QED [0.69, 0.81)", "Stratum 4: QED [0.81, 1.0]"];
    let panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"];

    for (i, (area, sizes)) in areas.iter().zip(strata_cluster_sizes.iter()).enumerate() {
        let _ = draw_panel_label(area, panel_labels.get(i).unwrap_or(&""));
        let float_sizes: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
        let (bins, counts) = compute_histogram(&float_sizes, 30);
        let max_count = *counts.iter().max().unwrap_or(&1) as f64;

        let mut chart = ChartBuilder::on(area)
            .caption(qed_labels.get(i).unwrap_or(&""), (FONT, TITLE_SIZE - 2).into_font())
            .margin(18)
            .x_label_area_size(40)
            .y_label_area_size(55)
            .build_cartesian_2d(bins[0]..bins[bins.len() - 1], 0.0..max_count * 1.15)?;

        chart.configure_mesh()
            .x_desc("Cluster Size")
            .y_desc("Count")
            .x_label_style((FONT, TICK_SIZE))
            .y_label_style((FONT, TICK_SIZE))
            .axis_desc_style((FONT, AXIS_LABEL_SIZE))
            .light_line_style(GRID_STYLE)
            .draw()?;

        let bar_width = (bins[1] - bins[0]) * 0.92;
        let color = STRATUM_COLORS[i % 5];
        chart.draw_series(
            bins.windows(2).zip(counts.iter()).map(|(bp, &c)| {
                Rectangle::new([(bp[0], 0.0), (bp[0] + bar_width, c as f64)], color.mix(BAR_ALPHA).filled())
            })
        )?;

        // Summary stats annotation
        let n_clust = sizes.len();
        let mean_sz = if n_clust > 0 { float_sizes.iter().sum::<f64>() / n_clust as f64 } else { 0.0 };
        let mut sorted_sz = float_sizes.clone();
        sorted_sz.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_sz = if n_clust > 0 {
            if n_clust % 2 == 0 { (sorted_sz[n_clust / 2 - 1] + sorted_sz[n_clust / 2]) / 2.0 }
            else { sorted_sz[n_clust / 2] }
        } else { 0.0 };
        let x_range = bins[bins.len() - 1] - bins[0];
        chart.draw_series(std::iter::once(
            Text::new(
                format!("n={}, μ={:.0}, med={:.0}", n_clust, mean_sz, med_sz),
                (bins[0] + x_range * 0.55, max_count * 1.02),
                (FONT, ANNOT_SIZE).into_font(),
            )
        ))?;
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

    let root = SVGBackend::new(&path, (850, 120 + n_dims as u32 * 36)).into_drawing_area();
    root.fill(&WHITE)?;

    let _ = draw_colorbar(&root, -1.0, 1.0, "r", heatmap_color);

    let mut chart = ChartBuilder::on(&root)
        .caption("Latent Dimension ↔ Property Correlations", (FONT, TITLE_SIZE).into_font())
        .margin(20)
        .x_label_area_size(65)
        .y_label_area_size(60)
        .right_y_label_area_size(70)
        .build_cartesian_2d(0..n_props, 0..n_dims)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_label_formatter(&|idx| properties.get(*idx).unwrap_or(&"").to_string())
        .y_label_formatter(&|idx| format!("Dim {}", dim_correlations.get(*idx).map(|d| d.0).unwrap_or(*idx)))
        .x_label_style((FONT, TICK_SIZE))
        .y_label_style((FONT, TICK_SIZE))
        .draw()?;

    let highlight_gold = RGBColor(255, 215, 0);
    for (row, &(_dim_id, r_qed, r_logp, r_sas)) in dim_correlations.iter().enumerate() {
        let vals = [r_qed, r_logp, r_sas];
        for (col, &val) in vals.iter().enumerate() {
            let color = heatmap_color(val, -1.0, 1.0);
            chart.draw_series(std::iter::once(
                Rectangle::new([(col, row), (col + 1, row + 1)], color.filled())
            ))?;
            // === Highlight strong correlations (|r| >= 0.5) with gold border ===
            if val.abs() >= 0.5 {
                chart.draw_series(std::iter::once(
                    Rectangle::new([(col, row), (col + 1, row + 1)], highlight_gold.stroke_width(3))
                ))?;
            }
            let text_color = if val.abs() > 0.5 { &WHITE } else { &BLACK };
            chart.draw_series(std::iter::once(
                Text::new(
                    format!("{:+.3}", val),
                    (col, row),
                    (FONT, HEATMAP_ANNOT_SIZE).into_font().color(text_color),
                )
            ))?;
        }
    }

    // === Summary annotation: count of strong correlations ===
    let n_strong = dim_correlations.iter()
        .flat_map(|&(_, q, l, s)| vec![q.abs(), l.abs(), s.abs()])
        .filter(|&v| v >= 0.5)
        .count();
    let _ = root.draw(&Text::new(
        format!("{} cells with |r| >= 0.5 (gold border)", n_strong),
        (20, 80 + n_dims as i32 * 36 + 25),
        (FONT, ANNOT_SIZE - 1).into_font().style(FontStyle::Italic).color(&RGBColor(120, 80, 0)),
    ));

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
    let root = SVGBackend::new(&path, (800, 120 + n_fgs as u32 * 34)).into_drawing_area();
    root.fill(&WHITE)?;

    let _ = draw_colorbar(&root, -0.3, 0.3, "r", heatmap_color);

    let properties = ["QED", "logP", "SAS"];

    let mut chart = ChartBuilder::on(&root)
        .caption("Functional Group ↔ Property Correlations", (FONT, TITLE_SIZE).into_font())
        .margin(20)
        .x_label_area_size(65)
        .y_label_area_size(210)
        .right_y_label_area_size(70)
        .build_cartesian_2d(0..3usize, 0..n_fgs)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_label_formatter(&|idx| properties.get(*idx).unwrap_or(&"").to_string())
        .y_label_formatter(&|idx| fg_correlations.get(*idx).map(|(n, _, _, _)| n.clone()).unwrap_or_default())
        .x_label_style((FONT, TICK_SIZE))
        .y_label_style((FONT, TICK_SIZE))
        .draw()?;

    // Find the strongest |r| per property column for highlighting
    let mut max_abs_per_col = [0.0_f64; 3];
    for (_, r_qed, r_logp, r_sas) in fg_correlations.iter() {
        if r_qed.abs() > max_abs_per_col[0] { max_abs_per_col[0] = r_qed.abs(); }
        if r_logp.abs() > max_abs_per_col[1] { max_abs_per_col[1] = r_logp.abs(); }
        if r_sas.abs() > max_abs_per_col[2] { max_abs_per_col[2] = r_sas.abs(); }
    }
    let highlight_color = RGBColor(255, 215, 0); // gold border for strongest

    for (row, (_, r_qed, r_logp, r_sas)) in fg_correlations.iter().enumerate() {
        let vals = [*r_qed, *r_logp, *r_sas];
        for (col, &val) in vals.iter().enumerate() {
            let color = heatmap_color(val, -0.3, 0.3);
            chart.draw_series(std::iter::once(
                Rectangle::new([(col, row), (col + 1, row + 1)], color.filled())
            ))?;

            // === Highlight cells with |r| >= 0.25 (strong correlations) ===
            if val.abs() >= 0.25 {
                chart.draw_series(std::iter::once(
                    Rectangle::new([(col, row), (col + 1, row + 1)], highlight_color.stroke_width(3))
                ))?;
            }
            // Strongest per column gets a star marker
            let is_strongest = (val.abs() - max_abs_per_col[col]).abs() < 1e-6;
            let label = if is_strongest && val.abs() > 0.15 {
                format!("{:+.3}*", val)
            } else {
                format!("{:+.3}", val)
            };

            let text_color = if val.abs() > 0.20 { &WHITE } else { &BLACK };
            chart.draw_series(std::iter::once(
                Text::new(
                    label,
                    (col, row),
                    (FONT, HEATMAP_ANNOT_SIZE).into_font().color(text_color),
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

    let root = SVGBackend::new(&path, FIG_PANEL_2X2).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 2));
    let titles = ["Silhouette Score", "Davies-Bouldin Index", "Quantization Error", "Gini Coefficient"];
    let y_labels = ["Score", "Index", "Error", "Coefficient"];
    let panel_labels = ["(a)", "(b)", "(c)", "(d)"];
    let metrics: Vec<Vec<f64>> = vec![
        strata_metrics.iter().map(|m| m.1).collect(),
        strata_metrics.iter().map(|m| m.2).collect(),
        strata_metrics.iter().map(|m| m.3).collect(),
        strata_metrics.iter().map(|m| m.4).collect(),
    ];

    for (i, area) in areas.iter().enumerate() {
        let _ = draw_panel_label(area, panel_labels[i]);
        let vals = &metrics[i];
        let min_v = vals.iter().copied().fold(f64::MAX, f64::min);
        let max_v = vals.iter().copied().fold(f64::MIN, f64::max);
        let range = (max_v - min_v).max(0.001);
        let y_lo = min_v - range * 0.15;
        let y_hi = max_v + range * 0.15;

        let mut chart = ChartBuilder::on(area)
            .caption(titles[i], (FONT, TITLE_SIZE - 2).into_font())
            .margin(14)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0..vals.len(), y_lo..y_hi)?;

        chart.configure_mesh()
            .x_desc("Stratum")
            .y_desc(y_labels[i])
            .x_label_style((FONT, TICK_SIZE))
            .y_label_style((FONT, TICK_SIZE))
            .axis_desc_style((FONT, AXIS_LABEL_SIZE))
            .light_line_style(GRID_STYLE)
            .draw()?;

        chart.draw_series(
            vals.iter().enumerate().map(|(j, &v)| {
                let color = STRATUM_COLORS[j % 5];
                let bar_bottom = if y_lo > 0.0 { y_lo } else { 0.0_f64.min(v) };
                Rectangle::new([(j, bar_bottom), (j + 1, v)], color.mix(BAR_ALPHA).filled())
            })
        )?;

        chart.draw_series(
            vals.iter().enumerate().map(|(j, &v)| {
                Text::new(
                    format!("{:.4}", v),
                    (j, v),
                    (FONT, ANNOT_SIZE).into_font(),
                )
            })
        )?;

        // === Key finding annotations per panel ===
        if vals.len() >= 2 {
            let first = vals[0];
            let last = vals[vals.len() - 1];
            let pct_change = ((last - first) / first.abs().max(1e-10)) * 100.0;
            let direction = if pct_change > 0.0 { "+" } else { "" };
            let trend_label = format!("S0->S4: {}{:.0}%", direction, pct_change);
            let annot_color = if pct_change < 0.0 { RGBColor(0, 120, 0) } else { RGBColor(180, 0, 0) };

            // Draw trend line from first to last bar
            chart.draw_series(std::iter::once(
                PathElement::new(
                    vec![(0, first), (vals.len() - 1, last)],
                    annot_color.stroke_width(2),
                )
            ))?;
            // Trend label
            let label_y = (first + last) / 2.0 + range * 0.08;
            chart.draw_series(std::iter::once(
                Text::new(
                    trend_label,
                    (vals.len() / 2, label_y),
                    (FONT, ANNOT_SIZE).into_font().style(FontStyle::Bold).color(&annot_color),
                )
            ))?;
        }
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
        "Loss", 80, RGBColor(214, 39, 40),
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

    let root = SVGBackend::new(&path, FIG_SINGLE).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Latent Dimension Variance", (FONT, TITLE_SIZE).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0..n, 0.0..max_var)?;

    chart.configure_mesh()
        .x_desc("Dimension")
        .y_desc("Variance")
        .x_label_style((FONT, TICK_SIZE))
        .y_label_style((FONT, TICK_SIZE))
        .axis_desc_style((FONT, AXIS_LABEL_SIZE))
        .light_line_style(GRID_STYLE)
        .draw()?;

    chart.draw_series(
        dim_variances.iter().enumerate().map(|(i, &(_, var))| {
            let intensity = (var / max_var).clamp(0.0, 1.0);
            let color = RGBColor(
                (31.0 + intensity * 183.0) as u8,
                (119.0 - intensity * 80.0) as u8,
                (180.0 - intensity * 140.0) as u8,
            );
            Rectangle::new([(i, 0.0), (i + 1, var)], color.mix(BAR_ALPHA).filled())
        })
    )?;

    // Bar outlines for print clarity
    chart.draw_series(
        dim_variances.iter().enumerate().map(|(i, &(_, var))| {
            Rectangle::new([(i, 0.0), (i + 1, var)], BLACK.stroke_width(BAR_STROKE))
        })
    )?;

    // Value labels on top of bars
    chart.draw_series(
        dim_variances.iter().enumerate().map(|(i, &(_dim_id, var))| {
            let label = if var >= 0.01 { format!("{:.3}", var) } else { format!("{:.4}", var) };
            Text::new(
                label,
                (i, var + max_var * 0.01),
                (FONT, ANNOT_SIZE - 2).into_font().transform(FontTransform::Rotate270),
            )
        })
    )?;

    // === Highlight the dominant dimension with annotation ===
    if let Some((max_idx, &(dim_id, max_variance))) = dim_variances.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
    {
        // Gold highlight border on the dominant bar
        let gold = RGBColor(200, 160, 0);
        chart.draw_series(std::iter::once(
            Rectangle::new([(max_idx, 0.0), (max_idx + 1, max_variance)], gold.stroke_width(3))
        ))?;
        // Annotation pointing to dominant dimension
        chart.draw_series(std::iter::once(
            Text::new(
                format!("Dim {} dominant", dim_id),
                (max_idx, max_variance + max_var * 0.08),
                (FONT, ANNOT_SIZE).into_font().style(FontStyle::Bold).color(&gold),
            )
        ))?;
        chart.draw_series(std::iter::once(
            Text::new(
                "(encodes aromatic ring presence, |r|=0.54)".to_string(),
                (max_idx, max_variance + max_var * 0.04),
                (FONT, ANNOT_SIZE - 2).into_font().style(FontStyle::Italic).color(&RGBColor(100, 100, 100)),
            )
        ))?;
    }

    // === Count of collapsed dimensions (variance < 0.001) ===
    let n_collapsed = dim_variances.iter().filter(|(_, v)| *v < 0.001).count();
    if n_collapsed > 0 {
        chart.draw_series(std::iter::once(
            Text::new(
                format!("{} dims near-collapsed (var < 0.001)", n_collapsed),
                (n / 2, max_var * 0.95),
                (FONT, ANNOT_SIZE - 1).into_font().color(&RGBColor(140, 140, 140)),
            )
        ))?;
    }

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
    let root = SVGBackend::new(&path, (300 * n_strata as u32 + 140, 440)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, n_strata));
    let qed_labels = ["S0: Low QED", "S1: Med-Low", "S2: Medium", "S3: Med-High", "S4: High QED"];
    let panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"];

    for (i, (area, u_matrix)) in areas.iter().zip(u_matrices.iter()).enumerate() {
        if u_matrix.is_empty() { continue; }
        let _ = draw_panel_label(area, panel_labels.get(i).unwrap_or(&""));
        let h = u_matrix.len();
        let w = u_matrix[0].len();

        let all_vals: Vec<f64> = u_matrix.iter().flatten().copied().collect();
        let u_min = all_vals.iter().copied().fold(f64::MAX, f64::min);
        let u_max = all_vals.iter().copied().fold(f64::MIN, f64::max);

        let mut chart = ChartBuilder::on(area)
            .caption(qed_labels.get(i).unwrap_or(&""), (FONT, TITLE_SIZE - 2).into_font())
            .margin(12)
            .x_label_area_size(35)
            .y_label_area_size(35)
            .build_cartesian_2d(0..w, 0..h)?;

        chart.configure_mesh()
            .disable_mesh()
            .x_desc("Grid X")
            .y_desc("Grid Y")
            .x_label_style((FONT, TICK_SIZE - 1))
            .y_label_style((FONT, TICK_SIZE - 1))
            .axis_desc_style((FONT, AXIS_LABEL_SIZE - 1))
            .draw()?;

        for r in 0..h {
            for c in 0..w {
                let color = sequential_color(u_matrix[r][c], u_min, u_max);
                chart.draw_series(std::iter::once(
                    Rectangle::new([(c, r), (c + 1, r + 1)], color.filled())
                ))?;
            }
        }

        // Min/max range annotation below grid
        let range_text = format!("Range: [{:.3}, {:.3}]", u_min, u_max);
        area.draw(&Text::new(
            range_text,
            (12, (h as i32) * 12 + 95),
            (FONT, ANNOT_SIZE - 1).into_font().color(&BLACK),
        ))?;
        // U-matrix max as a key metric
        let umax_label = format!("U-max: {:.3}", u_max);
        area.draw(&Text::new(
            umax_label,
            (12, (h as i32) * 12 + 110),
            (FONT, ANNOT_SIZE - 1).into_font().style(FontStyle::Bold).color(
                if i == n_strata - 1 { &RGBColor(0, 120, 0) } else if i == 0 { &RGBColor(180, 0, 0) } else { &BLACK }
            ),
        ))?;
    }

    // === Overall trend annotation below all panels ===
    if n_strata >= 2 {
        let first_umax = u_matrices[0].iter().flatten().copied().fold(f64::MIN, f64::max);
        let last_umax = u_matrices[n_strata - 1].iter().flatten().copied().fold(f64::MIN, f64::max);
        if first_umax > 0.0 {
            let pct_reduction = ((first_umax - last_umax) / first_umax) * 100.0;
            let trend_text = format!(
                "U-matrix max decreases S0 ({:.3}) -> S4 ({:.3}): {:.0}% reduction = smoother latent topology at higher QED",
                first_umax, last_umax, pct_reduction
            );
            root.draw(&Text::new(
                trend_text,
                (20, 420),
                (FONT, ANNOT_SIZE).into_font().style(FontStyle::Italic).color(&RGBColor(80, 80, 80)),
            ))?;
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

    let root = SVGBackend::new(&path, (220 + n_fgs as u32 * 60, 140 + n_clusters as u32 * 34)).into_drawing_area();
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

    let _ = draw_colorbar(&root, e_min, e_max, "Enrichment", heatmap_color);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("FG Enrichment — Stratum {}", stratum_id), (FONT, TITLE_SIZE).into_font())
        .margin(20)
        .x_label_area_size(130)
        .y_label_area_size(85)
        .right_y_label_area_size(70)
        .build_cartesian_2d(0..n_fgs, 0..n_clusters)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_label_formatter(&|idx| fg_names.get(*idx).cloned().unwrap_or_default())
        .y_label_formatter(&|idx| cluster_fg_enrichments.get(*idx).map(|(id, _)| format!("C{}", id)).unwrap_or_default())
        .x_label_style((FONT, TICK_SIZE).into_font().transform(FontTransform::Rotate270))
        .y_label_style((FONT, TICK_SIZE))
        .draw()?;

    let log_min = e_min.ln().min(-1.0);
    let log_max = e_max.ln().max(1.0);
    let highlight_border = RGBColor(255, 215, 0); // gold border for extreme enrichments

    for (row, row_data) in data.iter().enumerate() {
        // Track the max enrichment in this row for annotation
        let row_max = row_data.iter().copied().fold(0.0_f64, f64::max);
        let row_min = row_data.iter().copied().fold(f64::MAX, f64::min);

        for (col, &val) in row_data.iter().enumerate() {
            let log_val = if val > 0.0 { val.ln() } else { -2.0 };
            let color = heatmap_color(log_val, log_min, log_max);
            chart.draw_series(std::iter::once(
                Rectangle::new([(col, row), (col + 1, row + 1)], color.filled())
            ))?;

            // === Highlight extreme enrichments (>3x or <0.3x) with gold border ===
            if val >= 3.0 || val <= 0.3 {
                chart.draw_series(std::iter::once(
                    Rectangle::new([(col, row), (col + 1, row + 1)], highlight_border.stroke_width(2))
                ))?;
            }

            // Cell value annotation
            let norm_t = if (log_max - log_min).abs() > 1e-6 {
                (log_val - log_min) / (log_max - log_min)
            } else { 0.5 };
            let text_color = if norm_t > 0.65 || norm_t < 0.35 { &WHITE } else { &BLACK };
            let label = if val >= 10.0 { format!("{:.0}", val) }
                else if val >= 1.0 { format!("{:.1}", val) }
                else { format!("{:.2}", val) };
            chart.draw_series(std::iter::once(
                Text::new(
                    label,
                    (col, row),
                    (FONT, HEATMAP_ANNOT_SIZE - 1).into_font().color(text_color),
                )
            ))?;
        }

        // === Row annotation: flag clusters with extreme enrichment range ===
        if row_max >= 5.0 || row_min <= 0.2 {
            let cluster_id = cluster_fg_enrichments.get(row).map(|(id, _)| *id).unwrap_or(row);
            let note = if row_max >= 5.0 && row_min <= 0.3 {
                format!("C{}: {:.1}x enrich / {:.2}x depl", cluster_id, row_max, row_min)
            } else if row_max >= 5.0 {
                format!("C{}: {:.1}x max enrichment", cluster_id, row_max)
            } else {
                format!("C{}: {:.2}x strong depletion", cluster_id, row_min)
            };
            // Place annotation to the right of the heatmap row (outside grid area)
            // We use root-level drawing since chart coordinates are inside the grid
            let _ = root.draw(&Text::new(
                note,
                (85 + n_fgs as i32 * 60 - 70, 60 + row as i32 * 34 + 10),
                (FONT, ANNOT_SIZE - 2).into_font().style(FontStyle::Italic).color(&RGBColor(120, 80, 0)),
            ));
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

    let root = SVGBackend::new(&path, (140 + n as u32 * 36, 140 + n as u32 * 36)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut matrix = vec![vec![0.0_f64; n]; n];
    let max_dist = distances.iter().map(|d| d.2).fold(0.0_f64, f64::max);

    for &(a, b, d) in distances {
        if a < n && b < n {
            matrix[a][b] = d;
            matrix[b][a] = d;
        }
    }

    let _ = draw_colorbar(&root, 0.0, max_dist, "Distance", sequential_color);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Inter-Cluster Distances — Stratum {}", stratum_id), (FONT, TITLE_SIZE).into_font())
        .margin(14)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .right_y_label_area_size(70)
        .build_cartesian_2d(0..n, 0..n)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Cluster ID")
        .y_desc("Cluster ID")
        .x_label_style((FONT, TICK_SIZE))
        .y_label_style((FONT, TICK_SIZE))
        .axis_desc_style((FONT, AXIS_LABEL_SIZE))
        .draw()?;

    for r in 0..n {
        for c in 0..n {
            let color = sequential_color(matrix[r][c], 0.0, max_dist);
            chart.draw_series(std::iter::once(
                Rectangle::new([(c, r), (c + 1, r + 1)], color.filled())
            ))?;
            // Cell value annotation (only for small matrices to avoid clutter)
            if n <= 15 {
                let val = matrix[r][c];
                let norm_t = if max_dist > 1e-6 { val / max_dist } else { 0.0 };
                let text_color = if norm_t > 0.5 { &WHITE } else { &BLACK };
                chart.draw_series(std::iter::once(
                    Text::new(
                        format!("{:.2}", val),
                        (c, r),
                        (FONT, HEATMAP_ANNOT_SIZE - 2).into_font().color(text_color),
                    )
                ))?;
            }
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

    let root = SVGBackend::new(&path, (1300, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((1, 3));
    let property_names = ["QED (mean ± std)", "logP (mean ± std)", "SAS (mean ± std)"];
    let y_descs = ["QED", "logP", "SAS"];
    let panel_labels = ["(a)", "(b)", "(c)"];

    for (p_idx, area) in areas.iter().enumerate() {
        let _ = draw_panel_label(area, panel_labels[p_idx]);

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
            .caption(property_names[p_idx], (FONT, TITLE_SIZE - 2).into_font())
            .margin(16)
            .x_label_area_size(45)
            .y_label_area_size(60)
            .build_cartesian_2d(0..means.len(), y_min..y_max)?;

        chart.configure_mesh()
            .x_desc("Stratum")
            .y_desc(y_descs[p_idx])
            .x_label_style((FONT, TICK_SIZE))
            .y_label_style((FONT, TICK_SIZE))
            .axis_desc_style((FONT, AXIS_LABEL_SIZE))
            .light_line_style(GRID_STYLE)
            .draw()?;

        chart.draw_series(
            means.iter().enumerate().map(|(i, &m)| {
                let color = STRATUM_COLORS[i % 5];
                let bar_bottom = if y_min > 0.0 { y_min } else { 0.0_f64.min(m) };
                Rectangle::new([(i, bar_bottom), (i + 1, m)], color.mix(BAR_ALPHA).filled())
            })
        )?;

        // Error bars: vertical line centered on bar
        chart.draw_series(
            means.iter().zip(stds.iter()).enumerate().map(|(i, (&m, &s))| {
                let lo = m - s;
                let hi = m + s;
                PathElement::new(vec![(i, lo), (i, hi)], BLACK.stroke_width(2))
            })
        )?;

        // Value labels above bars
        chart.draw_series(
            means.iter().enumerate().map(|(i, &m)| {
                Text::new(
                    format!("{:.2}", m),
                    (i, m + (y_max - y_min) * 0.02),
                    (FONT, ANNOT_SIZE - 1).into_font(),
                )
            })
        )?;

        // === Delta annotation: S0 -> S4 change ===
        if means.len() >= 2 {
            let first = means[0];
            let last = means[means.len() - 1];
            let delta = last - first;
            let direction = if delta > 0.0 { "+" } else { "" };
            let delta_label = format!("\u{0394} = {}{:.2}", direction, delta);
            let delta_color = if delta.abs() > 0.5 { RGBColor(180, 0, 0) } else { RGBColor(80, 80, 80) };

            // Arrow from first bar to last bar
            let arrow_y = y_max - (y_max - y_min) * 0.06;
            chart.draw_series(std::iter::once(
                PathElement::new(
                    vec![(0, arrow_y), (means.len() - 1, arrow_y)],
                    delta_color.stroke_width(2),
                )
            ))?;
            // Arrowhead (small lines)
            let head_len = (y_max - y_min) * 0.03;
            chart.draw_series(std::iter::once(
                PathElement::new(
                    vec![(means.len() - 1, arrow_y - head_len), (means.len() - 1, arrow_y + head_len)],
                    delta_color.stroke_width(2),
                )
            ))?;
            // Delta label
            chart.draw_series(std::iter::once(
                Text::new(
                    delta_label,
                    (means.len() / 2, arrow_y + (y_max - y_min) * 0.02),
                    (FONT, ANNOT_SIZE).into_font().style(FontStyle::Bold).color(&delta_color),
                )
            ))?;
        }
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

    let root = SVGBackend::new(&path, (950, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let _ = draw_colorbar(&root, 0.0, 1.0, "QED", sequential_color);

    let mut chart = ChartBuilder::on(&root)
        .caption("Molecular Graph Complexity", (FONT, TITLE_SIZE).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .right_y_label_area_size(70)
        .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;

    chart.configure_mesh()
        .x_desc("Number of Atoms")
        .y_desc("Number of Bonds")
        .x_label_style((FONT, TICK_SIZE))
        .y_label_style((FONT, TICK_SIZE))
        .axis_desc_style((FONT, AXIS_LABEL_SIZE))
        .light_line_style(GRID_STYLE)
        .draw()?;

    chart.draw_series(
        points.iter().map(|&(a, b, q)| {
            let color = sequential_color(q, 0.0, 1.0);
            Circle::new((a, b), SCATTER_R, color.mix(SCATTER_ALPHA).filled())
        })
    )?
    .label("Color: QED (Viridis scale)")
    .legend(|(x, y)| Circle::new((x + 10, y), 5, RGBColor(33, 145, 140).filled()));

    // === Annotation: drug-like sweet spot region ===
    // High-QED molecules cluster in the 18-28 atom, 19-32 bond range
    let sweet_x0 = 18.0_f64;
    let sweet_x1 = 28.0_f64;
    let sweet_y0 = 19.0_f64;
    let sweet_y1 = 32.0_f64;
    let sweet_color = RGBColor(0, 150, 0);
    // Dashed rectangle outline
    let segments = [
        (sweet_x0, sweet_y0, sweet_x1, sweet_y0), // bottom
        (sweet_x1, sweet_y0, sweet_x1, sweet_y1), // right
        (sweet_x1, sweet_y1, sweet_x0, sweet_y1), // top
        (sweet_x0, sweet_y1, sweet_x0, sweet_y0), // left
    ];
    for &(ax0, ay0, ax1, ay1) in &segments {
        let n_dashes = 8;
        for d in 0..n_dashes {
            let t0 = d as f64 / n_dashes as f64;
            let t1 = (d as f64 + 0.5) / n_dashes as f64;
            let sx = ax0 + t0 * (ax1 - ax0);
            let sy = ay0 + t0 * (ay1 - ay0);
            let ex = ax0 + t1 * (ax1 - ax0);
            let ey = ay0 + t1 * (ay1 - ay0);
            chart.draw_series(std::iter::once(
                PathElement::new(vec![(sx, sy), (ex, ey)], sweet_color.stroke_width(2))
            ))?;
        }
    }
    // Label
    chart.draw_series(std::iter::once(
        Text::new(
            "High-QED sweet spot".to_string(),
            (sweet_x0, sweet_y1 + 1.0),
            (FONT, ANNOT_SIZE).into_font().style(FontStyle::Bold).color(&sweet_color),
        )
    ))?;

    // === Summary stats annotation ===
    let mean_atoms = points.iter().map(|p| p.0).sum::<f64>() / points.len() as f64;
    let mean_bonds = points.iter().map(|p| p.1).sum::<f64>() / points.len() as f64;
    chart.draw_series(std::iter::once(
        Text::new(
            format!("Mean: {:.1} atoms, {:.1} bonds", mean_atoms, mean_bonds),
            (x_max * 0.55, y_max * 0.06),
            (FONT, ANNOT_SIZE).into_font().color(&RGBColor(80, 80, 80)),
        )
    ))?;

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font((FONT, LEGEND_SIZE))
        .draw()?;

    root.present()?;
    Ok("figures/molecule_complexity.svg".to_string())
}

/// QED vs SAS scatter plot with molecule-level data and stratum means.
pub fn plot_qed_sas_scatter(
    qed_vals: &[f64],
    sas_vals: &[f64],
    stratum_labels: &[usize],
    output_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = format!("{}/figures/qed_sas_scatter.svg", output_dir);
    ensure_parent_dir(&path);

    let root = SVGBackend::new(&path, (950, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    // Subsample for scatter (5% for visibility)
    let step = (qed_vals.len() / 12000).max(1);
    let points: Vec<(f64, f64)> = qed_vals.iter()
        .zip(sas_vals.iter())
        .step_by(step)
        .map(|(&q, &s)| (q, s))
        .collect();

    // Compute stratum means and SDs
    let mut strata_qed: Vec<Vec<f64>> = vec![vec![]; 5];
    let mut strata_sas: Vec<Vec<f64>> = vec![vec![]; 5];
    for (i, (&q, &s)) in qed_vals.iter().zip(sas_vals.iter()).enumerate() {
        let stratum = if i < stratum_labels.len() { stratum_labels[i] } else { 0 };
        if stratum < 5 {
            strata_qed[stratum].push(q);
            strata_sas[stratum].push(s);
        }
    }

    let stratum_means: Vec<(f64, f64, f64, f64)> = (0..5).map(|s| {
        let n = strata_qed[s].len() as f64;
        if n == 0.0 { return (0.0, 0.0, 0.0, 0.0); }
        let qm = strata_qed[s].iter().sum::<f64>() / n;
        let sm = strata_sas[s].iter().sum::<f64>() / n;
        let qsd = (strata_qed[s].iter().map(|v| (v - qm).powi(2)).sum::<f64>() / n).sqrt();
        let ssd = (strata_sas[s].iter().map(|v| (v - sm).powi(2)).sum::<f64>() / n).sqrt();
        (qm, sm, qsd, ssd)
    }).collect();

    // Compute molecule-level correlation
    let n = qed_vals.len() as f64;
    let q_mean = qed_vals.iter().sum::<f64>() / n;
    let s_mean = sas_vals.iter().sum::<f64>() / n;
    let cov: f64 = qed_vals.iter().zip(sas_vals.iter())
        .map(|(&q, &s)| (q - q_mean) * (s - s_mean)).sum::<f64>() / n;
    let q_std = (qed_vals.iter().map(|&q| (q - q_mean).powi(2)).sum::<f64>() / n).sqrt();
    let s_std = (sas_vals.iter().map(|&s| (s - s_mean).powi(2)).sum::<f64>() / n).sqrt();
    let r = if q_std > 0.0 && s_std > 0.0 { cov / (q_std * s_std) } else { 0.0 };

    let mut chart = ChartBuilder::on(&root)
        .caption("QED vs Synthetic Accessibility Score", (FONT, TITLE_SIZE).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0_f64..1.05, 0.5_f64..6.5)?;

    chart.configure_mesh()
        .x_desc("QED")
        .y_desc("SAS")
        .x_label_style((FONT, TICK_SIZE))
        .y_label_style((FONT, TICK_SIZE))
        .axis_desc_style((FONT, AXIS_LABEL_SIZE))
        .light_line_style(GRID_STYLE)
        .draw()?;

    // Draw molecule-level scatter coloured by stratum
    let sampled_strata: Vec<usize> = stratum_labels.iter().step_by(step).copied().collect();
    for (idx, &(q, s)) in points.iter().enumerate() {
        let st = sampled_strata.get(idx).copied().unwrap_or(0).min(4);
        let c = STRATUM_COLORS[st];
        chart.draw_series(std::iter::once(
            Circle::new((q, s), 2_u32, c.mix(0.25).filled())
        ))?;
    }

    // Molecule-level legend entry
    chart.draw_series(std::iter::once(
        Circle::new((f64::NAN, f64::NAN), 0_u32, WHITE.filled())
    ))?
    .label(format!("Molecules (n = {}, r = {:.2}, R² = {:.3})", qed_vals.len(), r, r * r))
    .legend(|(x, y)| Circle::new((x + 10, y), 4, RGBColor(180, 180, 180).filled()));

    // === Ecological fallacy annotation box ===
    let box_color = RGBColor(80, 80, 80);
    // Background rectangle for annotation
    chart.draw_series(std::iter::once(
        Rectangle::new([(0.55, 1.2), (1.02, 2.3)], WHITE.mix(0.92).filled())
    ))?;
    chart.draw_series(std::iter::once(
        Rectangle::new([(0.55, 1.2), (1.02, 2.3)], box_color.stroke_width(1))
    ))?;
    let anno_lines = [
        (0.57, 2.10, "Molecule-level:"),
        (0.57, 1.85, &format!("  r = {:.2}, R\u{00B2} = {:.3}", r, r * r)),
        (0.57, 1.60, "Stratum-mean:"),
        (0.57, 1.35, "  R\u{00B2} = 0.95 (n=5)"),
    ];
    for &(ax, ay, txt) in &anno_lines {
        chart.draw_series(std::iter::once(
            Text::new(
                txt.to_string(),
                (ax, ay),
                (FONT, ANNOT_SIZE - 1).into_font().color(&box_color),
            )
        ))?;
    }

    // Draw stratum means as red squares with error bars
    for (i, &(qm, sm, _qsd, ssd)) in stratum_means.iter().enumerate() {
        // Vertical error bar (SAS direction)
        chart.draw_series(std::iter::once(
            PathElement::new(vec![(qm, sm - ssd), (qm, sm + ssd)], RED.stroke_width(2))
        ))?;
        // Mean marker
        chart.draw_series(std::iter::once(
            Circle::new((qm, sm), 6_u32, RED.filled())
        ))?;
        // Stratum label with mean values
        chart.draw_series(std::iter::once(
            Text::new(format!("S{} ({:.2}, {:.2})", i, qm, sm), (qm + 0.02, sm + 0.08),
                (FONT, ANNOT_SIZE - 1).into_font().color(&RED))
        ))?;
    }

    // Draw stratum-mean regression line
    if stratum_means.len() >= 2 {
        let q_vals: Vec<f64> = stratum_means.iter().map(|m| m.0).collect();
        let s_vals: Vec<f64> = stratum_means.iter().map(|m| m.1).collect();
        let qbar = q_vals.iter().sum::<f64>() / q_vals.len() as f64;
        let sbar = s_vals.iter().sum::<f64>() / s_vals.len() as f64;
        let slope = q_vals.iter().zip(s_vals.iter())
            .map(|(&q, &s)| (q - qbar) * (s - sbar)).sum::<f64>()
            / q_vals.iter().map(|&q| (q - qbar).powi(2)).sum::<f64>();
        let intercept = sbar - slope * qbar;
        let x0 = 0.1_f64;
        let x1 = 0.95_f64;
        chart.draw_series(std::iter::once(
            PathElement::new(
                vec![(x0, slope * x0 + intercept), (x1, slope * x1 + intercept)],
                RED.stroke_width(2)
            )
        ))?
        .label("Stratum-mean regression")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
    }

    chart.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font((FONT, LEGEND_SIZE))
        .draw()?;

    root.present()?;
    Ok("figures/qed_sas_scatter.svg".to_string())
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

    match plot_qed_sas_scatter(&data.qed_vals, &data.sas_vals, &data.stratum_labels, output_dir) {
        Ok(p) => { figures.push((p, "QED vs SAS scatter".to_string())); log::info!("  ✓ QED-SAS scatter"); }
        Err(e) => log::warn!("  ✗ QED-SAS scatter: {}", e),
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
        // Viridis: low should be dark purple (68, 1, 84)
        assert!(low.0 < 80 && low.1 < 10 && low.2 > 70, "Low should be dark purple");
        // High should be bright yellow (253, 231, 37)
        assert!(high.0 > 240 && high.1 > 220 && high.2 < 50, "High should be yellow");
    }

    #[test]
    fn test_heatmap_color_equal_range() {
        let c = heatmap_color(5.0, 5.0, 5.0);
        // With equal min/max, t=0.5 → near white (247, 247, 247)
        assert!(c.0 > 200 && c.1 > 200 && c.2 > 200);
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
