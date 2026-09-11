use nerva::primitives::{
    Activity, ActuatorId, Concentration, Distance, EnergyCost, ModulatorId, NeuronId, Position3,
    Position3D, Potential, RegionId, SensorId, SignalStrength, SimTime, SynapseId, SystemId,
    Threshold, Weight,
};

#[test]
fn every_supported_primitive_is_publicly_available() {
    let neuron = NeuronId(1);
    let synapse = SynapseId(2);
    let region = RegionId(3);
    let system = SystemId(4);
    let sensor = SensorId(5);
    let actuator = ActuatorId(6);
    let modulator = ModulatorId(7);

    assert_eq!(neuron.get(), 1);
    assert_eq!(synapse.get(), 2);
    assert_eq!(region.get(), 3);
    assert_eq!(system.get(), 4);
    assert_eq!(sensor.get(), 5);
    assert_eq!(actuator.get(), 6);
    assert_eq!(modulator.get(), 7);

    assert_eq!(Potential(-65.0).get(), -65.0);
    assert_eq!(Threshold(-50.0).get(), -50.0);
    assert_eq!(Weight::new(0.4).unwrap().get(), 0.4);
    assert!(matches!(
        nerva::primitives::Weight::new(-0.1),
        Err(nerva::primitives::WeightError::Negative(-0.1))
    ));
    assert!(matches!(
        nerva::primitives::Weight::new(f32::NAN),
        Err(nerva::primitives::WeightError::NonFinite(_))
    ));
    assert_eq!(SignalStrength::new(-0.5).get(), -0.5);
    assert_eq!(Concentration(0.6).get(), 0.6);
    assert_eq!(Activity(0.7).get(), 0.7);
    assert_eq!(Distance(0.8).get(), 0.8);
    assert_eq!(EnergyCost(0.9).get(), 0.9);

    assert_eq!(Position3::new(1.0, 2.0, 3.0).z, 3.0);
    assert_eq!(SimTime(10).checked_add_us(2), Some(SimTime(12)));
}

#[test]
fn compatibility_paths_use_the_canonical_primitive_types() {
    let primitive_neuron_id = NeuronId(42);
    let primitive_synapse_id = SynapseId(84);
    let core_neuron_id: nerva::core::NeuronId = primitive_neuron_id;
    let core_synapse_id: nerva::core::SynapseId = primitive_synapse_id;
    let primitive_time = SimTime(125);
    let core_time: nerva::core::SimTime = primitive_time;
    let primitive_position = Position3::new(1.0, 2.0, 3.0);
    let primitive_position_3d = Position3D {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    };
    let math_position: nerva::math::Position3D = primitive_position;

    assert_eq!(core_neuron_id, primitive_neuron_id);
    assert_eq!(core_synapse_id, primitive_synapse_id);
    assert_eq!(core_time, primitive_time);
    assert_eq!(primitive_position_3d, primitive_position);
    assert_eq!(math_position, primitive_position);
}
