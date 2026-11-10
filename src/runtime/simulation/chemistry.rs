//! Chemical event scheduling and local delayed-learning integration.
//! Kept inside simulation so it shares the authoritative queue and neural state.

use super::{Simulation, SimulationError};
use crate::runtime::learning_audit::{LearningAuditEvent, LearningPhase, TraceState};
use crate::{
    core::{Event, EventKind, NeuronId, SimTime, Spike},
    learning::PlasticityRule,
    neuromodulation::ChemicalSpecies,
    primitives::ModulatorId,
    runtime::ScheduledEvent,
};

impl<R: PlasticityRule> Simulation<R> {
    pub(super) fn advance_delayed_learning(
        &mut self,
        time: SimTime,
    ) -> Result<(), SimulationError> {
        if !self.plasticity_rule.needs_time_updates() {
            return Ok(());
        }
        let ids: Vec<_> = self.network.synapse_ids().collect();
        for id in ids {
            let edge = self.network.synapse(id).unwrap();
            let post = self
                .network
                .neuron(edge.post())
                .ok_or(SimulationError::UnknownNeuron(edge.post()))?
                .clone();
            let polarity = self
                .network
                .neuron(edge.pre())
                .ok_or(SimulationError::UnknownNeuron(edge.pre()))?
                .polarity();
            let enabled = self.learning_enabled && polarity == crate::core::Polarity::Excitatory;
            let diagnostics = self.learning_audit.as_ref().and_then(|_| {
                self.plasticity_rule
                    .delayed_update_diagnostics(edge, &post, time, enabled)
            });
            let before = self.learning_audit.as_ref().map(|_| TraceState::read(edge));
            let edge = self.network.synapse_mut(id).unwrap();
            let old = edge.weight();
            self.plasticity_rule
                .advance_time(edge, &post, time, enabled);
            edge.validate()
                .map_err(SimulationError::InvalidNetworkMutation)?;
            let new = edge.weight();
            if let (Some(before), Some(audit)) = (before, &mut self.learning_audit)
                && let Some(event) = before.finish(
                    edge,
                    time,
                    LearningPhase::Delayed,
                    post.modulation().learning_signal,
                    diagnostics,
                )
            {
                audit.record(event);
            }
            self.log_weight_change(time, id, old, new);
        }
        Ok(())
    }

    fn ensure_chemical_update(&mut self, time: SimTime) -> Result<(), SimulationError> {
        let Some(field) = &self.chemical_field else {
            return Ok(());
        };
        let Some(active_until) = field.active_until() else {
            return Ok(());
        };
        let next = time
            .checked_add_us(field.config().update_interval_us)
            .unwrap_or(active_until)
            .min(active_until);
        if self.chemical_booking.is_some_and(|(at, _)| at <= next) {
            return Ok(());
        }
        if let Some((_, sequence)) = self.chemical_booking.take() {
            self.scheduler.cancel(sequence);
        }
        let sequence = self.schedule_event(Event::new(next, EventKind::ChemicalUpdate))?;
        self.chemical_booking = Some((next, sequence));
        Ok(())
    }

    pub(super) fn process_chemical_events(
        &mut self,
        time: SimTime,
        events: &[ScheduledEvent<EventKind>],
    ) -> Result<Vec<NeuronId>, SimulationError> {
        let mut refresh = false;
        for event in events {
            match event.payload {
                EventKind::ChemicalRelease {
                    source,
                    species,
                    amount,
                } => {
                    let position = self
                        .network
                        .neuron(source)
                        .ok_or(SimulationError::UnknownNeuron(source))?
                        .position();
                    self.chemical_field
                        .as_mut()
                        .unwrap()
                        .release(ModulatorId(source.0), species, position, amount, time)
                        .map_err(SimulationError::Chemical)?;
                    refresh = true;
                }
                EventKind::ChemicalUpdate => {
                    if self.chemical_booking.is_some_and(|(at, _)| at == time) {
                        self.chemical_booking = None;
                    }
                    refresh = true;
                }
                _ => {}
            }
        }
        if !refresh {
            return Ok(Vec::new());
        }
        let field = self.chemical_field.as_mut().unwrap();
        field.advance_to(time).map_err(SimulationError::Chemical)?;
        let mut states = Vec::new();
        for cell in self.network.neurons() {
            let receptive = cell.receptor_profile() != crate::core::ReceptorProfile::default();
            let concentration = if receptive || self.learning_audit.is_some() {
                field
                    .sample(cell.position(), time)
                    .map_err(SimulationError::Chemical)?
            } else {
                crate::core::Concentrations::ZERO
            };
            let response = if receptive {
                cell.receptor_profile()
                    .response(concentration)
                    .map_err(SimulationError::Chemical)?
            } else {
                crate::core::ModulationState::default()
            };
            if let Some(audit) = &mut self.learning_audit {
                audit.record(LearningAuditEvent::ChemicalSample {
                    time,
                    neuron: cell.id(),
                    concentrations: concentration,
                    response,
                });
            }
            if response != cell.modulation() {
                states.push((cell.id(), response));
            }
        }
        let mut changed = Vec::with_capacity(states.len());
        for (id, response) in states {
            let cell = self.network.neuron_mut(id).unwrap();
            let from = cell.last_update();
            let potential_before = cell.membrane_potential();
            cell.set_modulation(time, response)?;
            if self.dynamics_auditing {
                self.event_log
                    .push(super::ObservationEvent::NeuronAdvanced {
                        time,
                        target: id,
                        from,
                        potential_before,
                        potential_after: cell.membrane_potential(),
                    });
            }
            changed.push(id);
        }
        self.ensure_chemical_update(time)?;
        Ok(changed)
    }

    pub(super) fn release_spike_chemistry(
        &mut self,
        spikes: &[Spike],
    ) -> Result<(), SimulationError> {
        if self.chemical_field.is_none() {
            return Ok(());
        }
        let mut released_at = None;
        for spike in spikes {
            let cell = self
                .network
                .neuron(spike.neuron_id)
                .ok_or(SimulationError::UnknownNeuron(spike.neuron_id))?;
            let release = cell.release_profile();
            let position = cell.position();
            for species in ChemicalSpecies::ALL {
                let amount = release.amount(species);
                if amount > 0. {
                    released_at = Some(spike.time);
                    self.chemical_field
                        .as_mut()
                        .unwrap()
                        .release(
                            ModulatorId(spike.neuron_id.0),
                            species,
                            position,
                            amount,
                            spike.time,
                        )
                        .map_err(SimulationError::Chemical)?;
                }
            }
        }
        if let Some(time) = released_at {
            self.ensure_chemical_update(time)?;
        }
        Ok(())
    }
}
