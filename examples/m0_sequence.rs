//! Runs the paired, multi-seed DSVLM-M0 sequence-learning study.

use dsvlm_rust::experiment::{M0ExperimentConfig, M0StudyConfig, run_m0_study};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = M0ExperimentConfig::default();
    let study_config = M0StudyConfig::default();
    let study = run_m0_study(&config, &study_config)?;

    println!("DSVLM-M0 paired seeds: {:?}", study_config.seeds);
    println!("seed  group  score  hits  false  stable  frozen replay");
    for comparison in &study.comparisons {
        for result in &comparison.groups {
            println!(
                "{:>4}  {:<5}  {:>5}  {:>4}  {:>5}  {:>6}  {:>13}",
                result.seed,
                result.group.label(),
                result.metrics.sequence_score(),
                result.metrics.transition_hits,
                result.metrics.false_transitions,
                result.metrics.stable_after_learning,
                result.metrics.frozen_probe_replay_identical,
            );
        }
    }

    println!(
        "G1−G2 mean={:.3}, p={:.6}; G1−G3 mean={:.3}, p={:.6}; M0 passed={}",
        study.success.mean_advantage_over_fixed,
        study.success.sign_test_p_over_fixed,
        study.success.mean_advantage_over_random,
        study.success.sign_test_p_over_random,
        study.success.passed,
    );
    Ok(())
}
