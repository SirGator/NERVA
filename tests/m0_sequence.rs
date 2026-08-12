use dsvlm_rust::experiment::{
    M0Experiment, M0ExperimentConfig, M0Group, M0StudyConfig, run_m0_comparison, run_m0_study,
};

#[test]
fn ordered_learning_beats_fixed_and_randomized_controls_without_runaway_activity() {
    let config = M0ExperimentConfig::default();
    let comparison = run_m0_comparison(&config).expect("valid M0 comparison");
    let g1 = comparison
        .group(M0Group::OrderedLearning)
        .expect("comparison contains G1");
    let g2 = comparison
        .group(M0Group::OrderedFixed)
        .expect("comparison contains G2");
    let g3 = comparison
        .group(M0Group::RandomLearning)
        .expect("comparison contains G3");

    assert_eq!(comparison.groups.len(), 4);
    assert!(comparison.ordered_learning_outperforms_controls());
    assert!(g1.metrics.sequence_score() > g2.metrics.sequence_score());
    assert!(g1.metrics.sequence_score() > g3.metrics.sequence_score());
    assert!(g1.metrics.transition_hits > 0);
    assert_eq!(g1.metrics.false_transitions, 0);
    assert!(g1.metrics.probe_spikes > 0);
    assert!(g1.metrics.stable_after_learning);
    assert!(g1.metrics.frozen_probe_replay_identical);
    assert!(g1.metrics.frozen_weights_unchanged);
    assert_eq!(g1.metrics.transition_latency_errors_us.len(), 3);
    assert!(g1.metrics.probe_mean_rate_hz.is_finite());
    assert_eq!(g1.metrics.probe_neuron_rates_hz.len(), 9);
    assert!(!g1.event_log.is_empty());
    assert!(
        comparison
            .groups
            .iter()
            .all(|result| result.seed == config.dsvlm.network.seed
                && result.metrics.frozen_probe_replay_identical
                && result.metrics.frozen_weights_unchanged)
    );
}

#[test]
fn symmetric_topology_and_frozen_probe_leave_weights_unchanged() {
    let config = M0ExperimentConfig::default();
    let fixed = M0Experiment::new(config.clone(), M0Group::OrderedFixed)
        .expect("valid fixed group")
        .run()
        .expect("fixed group succeeds");

    // The canonical M0 topology contains four fixed sensory connections, all
    // twelve directed non-self pattern transitions as equal plastic
    // candidates, and four fixed drive/feedback pairs for one interneuron.
    assert_eq!(fixed.initial_weights.len(), 24);
    assert_eq!(fixed.final_weights.len(), 24);
    assert!(
        fixed.final_weights[..4]
            .iter()
            .all(|&weight| weight == config.sensory_weight)
    );
    assert!(
        fixed.final_weights[4..16]
            .iter()
            .all(|&weight| (weight - config.recurrent_weight).abs()
                <= config.recurrent_weight_jitter + f32::EPSILON)
    );
    assert!(fixed.final_weights[16..].chunks_exact(2).all(|pair| {
        pair == [
            config.inhibitory_drive_weight,
            config.inhibitory_feedback_weight,
        ]
    }));
    assert_eq!(fixed.initial_weights, fixed.final_weights);
    assert!(fixed.metrics.frozen_probe_replay_identical);
    assert!(fixed.metrics.frozen_weights_unchanged);

    // A fresh run with the same seed must reproduce the complete immutable
    // observation log, not just its digest or final weight vector.
    let replay = M0Experiment::new(config, M0Group::OrderedFixed)
        .expect("valid replay")
        .run()
        .expect("replay succeeds");

    assert_eq!(fixed.event_log, replay.event_log);
    assert_eq!(fixed.event_log_digest, replay.event_log_digest);
    assert_eq!(fixed.final_weights, replay.final_weights);
}

#[test]
fn multi_seed_study_preserves_paired_seed_control_and_reports_aggregate_evidence() {
    let config = M0ExperimentConfig::default();
    let study_config = M0StudyConfig { seeds: vec![7, 19] };
    let study = run_m0_study(&config, &study_config).expect("valid paired M0 study");

    assert_eq!(study.comparisons.len(), study_config.seeds.len());
    assert_eq!(study.base_config, config);
    assert_eq!(study.study_config, study_config);
    assert_eq!(study.success.seeds, study_config.seeds.len());
    assert!(study.success.wins_over_fixed <= study.success.seeds);
    assert!(study.success.wins_over_random <= study.success.seeds);
    assert!((0.0..=1.0).contains(&study.success.sign_test_p_over_fixed));
    assert!((0.0..=1.0).contains(&study.success.sign_test_p_over_random));

    for (comparison, &seed) in study.comparisons.iter().zip(&study_config.seeds) {
        assert_eq!(comparison.groups.len(), 4);
        assert!(comparison.groups.iter().all(|result| result.seed == seed));
        assert!(comparison.groups.iter().all(|result| {
            result.metrics.frozen_probe_replay_identical && result.metrics.frozen_weights_unchanged
        }));
    }
}

#[test]
fn default_multi_seed_study_meets_the_declared_m0_success_criterion() {
    let study = run_m0_study(&M0ExperimentConfig::default(), &M0StudyConfig::default())
        .expect("default paired M0 study succeeds");

    assert_eq!(study.success.seeds, 6);
    assert_eq!(study.success.wins_over_fixed, 6);
    assert_eq!(study.success.wins_over_random, 6);
    assert_eq!(study.success.sign_test_p_over_fixed, 0.015625);
    assert_eq!(study.success.sign_test_p_over_random, 0.015625);
    assert!(study.success.all_runs_stable);
    assert!(study.success.all_replays_identical);
    assert!(study.success.passed);
}
