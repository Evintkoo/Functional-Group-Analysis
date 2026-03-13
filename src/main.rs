#[allow(dead_code)]
mod smiles;
#[allow(dead_code)]
mod features;
#[allow(dead_code)]
mod gnn;
#[allow(dead_code)]
mod autoencoder;
#[allow(dead_code)]
mod som;
#[allow(dead_code)]
mod functional_groups;
#[allow(dead_code)]
mod stats;
mod pipeline;
#[allow(dead_code)]
mod io;
#[allow(dead_code)]
mod visualization;

use std::env;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| {
            use std::io::Write;
            let ts = buf.timestamp_seconds();
            writeln!(buf, "[{} {:5} {}] {}",
                ts, record.level(),
                record.module_path().unwrap_or(""),
                record.args())
        })
        .target(env_logger::Target::Stderr)
        .init();

    let csv_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "250k_rndm_zinc_drugs_clean_3.csv".to_string());

    let output_dir = env::args()
        .nth(2)
        .unwrap_or_else(|| "results".to_string());

    log::info!("Functional Group Analysis — Graph-Based Pipeline (Rust)");
    log::info!("Input: {}", csv_path);
    log::info!("Output: {}", output_dir);

    match pipeline::run_pipeline(&csv_path, &output_dir) {
        Ok(results) => {
            let report = results.to_markdown();

            // Write RESULTS.md into output directory
            let results_path = format!("{}/RESULTS.md", output_dir);
            if let Err(e) = std::fs::write(&results_path, &report) {
                log::error!("Failed to write results: {}", e);
            } else {
                log::info!("Results written to {}", results_path);
            }

            // Write RESULTS.md at repo root with adjusted figure paths
            let root_report = report.replace("](figures/", "](results/figures/");
            if let Err(e) = std::fs::write("RESULTS.md", &root_report) {
                log::error!("Failed to write root RESULTS.md: {}", e);
            } else {
                log::info!("Root RESULTS.md written");
            }

            println!("\n{}", report);
        }
        Err(e) => {
            log::error!("Pipeline failed: {}", e);
            std::process::exit(1);
        }
    }
}
