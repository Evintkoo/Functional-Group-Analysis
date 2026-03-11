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
mod pipeline;
#[allow(dead_code)]
mod io;

use std::env;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
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

            // Write RESULTS.md
            let results_path = format!("{}/RESULTS.md", output_dir);
            if let Err(e) = std::fs::write(&results_path, &report) {
                log::error!("Failed to write results: {}", e);
            } else {
                log::info!("Results written to {}", results_path);
            }

            println!("\n{}", report);
        }
        Err(e) => {
            log::error!("Pipeline failed: {}", e);
            std::process::exit(1);
        }
    }
}
