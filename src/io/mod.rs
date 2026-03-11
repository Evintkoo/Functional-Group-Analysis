/// CSV data loading and result output for the ZINC molecular dataset.

use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct MoleculeRecord {
    pub smiles: String,
    #[serde(rename = "logP")]
    pub log_p: f64,
    pub qed: f64,
    #[serde(rename = "SAS")]
    pub sas: f64,
}

/// Load molecule records from the ZINC CSV dataset.
pub fn load_zinc_csv(path: &str) -> Result<Vec<MoleculeRecord>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_path(path)?;

    let mut records = Vec::new();
    for result in reader.deserialize() {
        let record: MoleculeRecord = match result {
            Ok(r) => r,
            Err(e) => {
                log::warn!("Skipping malformed row: {}", e);
                continue;
            }
        };
        // Clean SMILES string (remove trailing whitespace/newlines)
        let mut rec = record;
        rec.smiles = rec.smiles.trim().to_string();
        if !rec.smiles.is_empty() {
            records.push(rec);
        }
    }

    log::info!("Loaded {} molecule records from {}", records.len(), path);
    Ok(records)
}

/// Stratify molecules by QED score into bins.
pub fn stratify_by_qed(records: &[MoleculeRecord], edges: &[f64]) -> Vec<Vec<usize>> {
    let n_bins = edges.len() + 1;
    let mut strata: Vec<Vec<usize>> = vec![Vec::new(); n_bins];

    for (i, rec) in records.iter().enumerate() {
        let bin = edges.iter().position(|&e| rec.qed < e).unwrap_or(n_bins - 1);
        strata[bin].push(i);
    }

    for (i, stratum) in strata.iter().enumerate() {
        log::info!("QED stratum {}: {} molecules", i, stratum.len());
    }

    strata
}

/// Save clustering results to CSV.
pub fn save_cluster_results(
    output_dir: &str,
    group_id: usize,
    records: &[&MoleculeRecord],
    labels: &[usize],
    embeddings: &[Vec<f32>],
) -> Result<(), Box<dyn Error>> {
    let dir = format!("{}/group_{}", output_dir, group_id);
    fs::create_dir_all(&dir)?;

    // Labeled data
    let path = format!("{}/labeled_data.csv", dir);
    let mut file = fs::File::create(&path)?;
    writeln!(file, "smiles,logP,qed,SAS,cluster")?;
    for (i, rec) in records.iter().enumerate() {
        writeln!(
            file, "{},{},{},{},{}",
            rec.smiles, rec.log_p, rec.qed, rec.sas, labels[i]
        )?;
    }

    // Embeddings
    let emb_path = format!("{}/embeddings.csv", dir);
    let mut emb_file = fs::File::create(&emb_path)?;
    let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
    let header: Vec<String> = (0..dim).map(|i| format!("dim_{}", i)).collect();
    writeln!(emb_file, "{}", header.join(","))?;
    for emb in embeddings {
        let vals: Vec<String> = emb.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(emb_file, "{}", vals.join(","))?;
    }

    log::info!("Saved group {} results to {}", group_id, dir);
    Ok(())
}

/// Save cluster center decoded features.
pub fn save_cluster_centers(
    output_dir: &str,
    group_id: usize,
    centers: &[Vec<f32>],
    feature_names: &[&str],
) -> Result<(), Box<dyn Error>> {
    let dir = format!("{}/group_{}", output_dir, group_id);
    fs::create_dir_all(&dir)?;

    let path = format!("{}/cluster_centers.csv", dir);
    let mut file = fs::File::create(&path)?;
    writeln!(file, "{}", feature_names.join(","))?;
    for center in centers {
        let vals: Vec<String> = center.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(file, "{}", vals.join(","))?;
    }

    Ok(())
}

/// Save training loss history.
pub fn save_training_losses(
    output_dir: &str,
    train_losses: &[f32],
    val_losses: &[f32],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    let path = format!("{}/training_losses.csv", output_dir);
    let mut file = fs::File::create(Path::new(&path))?;
    writeln!(file, "epoch,train_loss,val_loss")?;
    for (i, (tl, vl)) in train_losses.iter().zip(val_losses.iter()).enumerate() {
        writeln!(file, "{},{:.6},{:.6}", i + 1, tl, vl)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratification() {
        let records = vec![
            MoleculeRecord { smiles: "C".into(), log_p: 0.0, qed: 0.3, sas: 1.0 },
            MoleculeRecord { smiles: "CC".into(), log_p: 0.0, qed: 0.5, sas: 1.0 },
            MoleculeRecord { smiles: "CCC".into(), log_p: 0.0, qed: 0.9, sas: 1.0 },
        ];
        let strata = stratify_by_qed(&records, &[0.4, 0.7]);
        assert_eq!(strata.len(), 3);
        assert_eq!(strata[0], vec![0]);
        assert_eq!(strata[1], vec![1]);
        assert_eq!(strata[2], vec![2]);
    }
}
