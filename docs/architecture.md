# Aktive Architektur und Konsolidierungsentscheidungen

**Stand:** 20. August 2026

Dieses Dokument beschreibt die tatsächlich über `src/lib.rs` kompilierte
Architektur. Die ausführliche M0-Logik und die experimentellen Kriterien stehen
weiterhin in `DSVLM_SLICE_ARCHITECTURE.md`.

## Aktiver Datenpfad

```text
environment
    ↕
roots + transduction + nerves
    ↓
runtime
    ├── executes core network state
    └── invokes local learning hooks
```

`core` bleibt für M0 der stabile, produktive Zwischenstand. Eine spätere
Aufteilung in `neural/neuron`, `neural/synapse` und `neural/network` erfolgt nur
als eigener, API-kompatibel geplanter Umbau und nicht zusammen mit neuen
Funktionen.

## Ebenen

| Ebene | Aktuelle Module | Verantwortung |
|---|---|---|
| 1. Grundlagen | `primitives`, `math`, `config` | Identitäten, Zeit, Werte, Geometrie, Validierung und reine Mathematik |
| 2. Neuronale Bausteine | `core` | Neuronen, Synapsen, Spikes und deterministischer Netzwerkgraph |
| 3. Lokale Veränderung | `learning`, optional `development` | lokale Plastizität beziehungsweise spätere Strukturentwicklung |
| 4. Ausführung | `runtime` | deterministische Ereignisordnung, Batches, Propagation und Simulation |
| 5. Schnittstellen | `roots`, `nerves`, `transduction` | feste Anschlussstellen, Transport und wertneutrale Übersetzung |
| 6. Versuche und Beobachtung | `environment`, `experiment`, optional `metrics`, `debug`, `visualization` | Stimulation, Versuchssteuerung und ausschließlich lesende Inspektion |

`learning` bezeichnet im aktuellen öffentlichen API-Pfad ausschließlich lokale
Plastizität. Eine spätere Umbenennung in `plasticity` benötigt einen separaten
Kompatibilitätsplan.

## Kanonische Grundtypen

`primitives` ist die einzige Definitionsquelle für Grundtypen:

- `NeuronId` und `SynapseId` werden von `core` nur weiter-exportiert.
- `SimTime` ist die exakte, mikrosekundengenaue Zeit der aktiven Runtime und
  wird von `core` kompatibel weiter-exportiert.
- `Position3` ist die kanonische räumliche Position. `Position3D` ist ein
  kompatibler Alias; `math` stellt ihn zusammen mit reinen Geometriefunktionen
  weiter bereit.

`Potential`, `Threshold`, `Weight`, `SignalStrength`, `Concentration`,
`Activity`, `Distance` und `EnergyCost` sind eine vorbereitete Migration. Der
aktive M0-Code verwendet für Membranpotential, Schwelle, Synapsengewicht und
Eingangsamplitude noch teilweise `f32`. Die Umstellung erfolgt datenpfadweise.
Als erster sinnvoller Pfad ist vorgesehen:

```text
Weight
→ core::Synapse
→ learning::PlasticityRule
→ runtime::ObservationEvent::WeightChanged
```

So existiert während einer Migration pro Invariante weiterhin genau eine
verbindliche Darstellung.

## Getrennte Ereignisrollen

| Typ | Rolle |
|---|---|
| `core::EventKind` | konkret ausführbarer Zustandsübergang der M0-Simulation |
| `core::Event` | Zeitstempel plus `EventKind`, noch ohne Queue-Reihenfolge |
| `runtime::ScheduledEvent` | technischer Queue-Umschlag mit Einfügereihenfolge |
| `runtime::EventBatch` | atomare Menge aller Ereignisse eines Zeitstempels |
| `runtime::ObservationEvent` | unveränderliche Beobachtung eines ausgeführten Übergangs |

Ein zusätzliches allgemeines Domain-Event wird erst eingeführt, wenn ein
konkreter, getesteter Übersetzungspfad zur Runtime existiert. Modulations- und
Entwicklungsereignisse gehören dann ihren jeweiligen Fachmodulen.

Die neutralen Grenztypen `Pattern`, `Observation` und `Action` werden von
`transduction` definiert. `environment` verwendet und re-exportiert sie für die
bisherige öffentliche API; dadurch hängt die Schnittstellenebene nicht mehr von
der darüberliegenden Umgebungsebene ab.

## Kontinuierliche intrinsische Neuronendynamik

`config::IntrinsicDynamicsConfig` beschreibt pro Zelle einen kontinuierlichen
Fähigkeitsvektor statt diskreter Neuronentypen:

```text
g_i = (intrinsic_drive, burst_gain, adaptation_gain,
       threshold_adaptation_gain, rebound_gain)
```

Alle Gains sind standardmäßig null, sodass bestehende LIF-Experimente denselben
`f32`-Rechenpfad und dieselben Ereigniszeiten behalten. Bei aktiven Fähigkeiten
hält `core::Neuron` die lokalen Zustände `B`, `A`, `R` und `T`; ein lesender
`core::IntrinsicState` macht sie für Diagnostik sichtbar. Der momentane Drive
wirkt als `D + B + R - A`, während `T` auf die Basisschwelle addiert wird.

`core::intrinsic` enthält die analytische Faltung exponentiell zerfallender
Ströme und die Intervallsuche mit analytischen Schranken für autonome
Schwellenüberschreitungen.
Der Core kennt dabei weiterhin keinen Scheduler. Die Runtime übernimmt nur den
vom Neuron prognostizierten lokalen Zeitpunkt, storniert veraltete Prognosen und
aggregiert inhibitorische Eingangsanteile separat, damit simultane Erregung den
Rebound-Zustand nicht verdeckt.

`DsvlmConfig::neuron` beschreibt wie bisher eine gemeinsame Zellklasse für die
Standardexperimente. Heterogene Fähigkeitsvektoren können durch einzeln
konstruierte `NeuronConfig`-Werte in einem manuell aufgebauten `Network` und
`Simulation::new` verwendet werden.

Sensorische und motorische Neuronen sind äußere Populationen desselben
`core::Network`. Ihre Rollen sind nur Metadaten; ihre internen und zum Kern
führenden Synapsen dürfen dieselben lokalen Lernregeln verwenden. Fest bleiben
die physische Root-/Nerven-Zuordnung und die wertneutrale Transduktion. Die
spätere lokale Bildung und das Pruning der Synapsen dieser äußeren Subnetze
gehören in `development`.

## Bereinigte Altstruktur

Die nicht aus `lib.rs` eingebundenen Verzeichnisse `area`, `common`, `event`,
`network`, `neuron`, `presets` und `synapse` stammten aus der Architektur vor
M0. Sie wurden entfernt, weil ihre Implementierungen entweder unvollständig
oder durch den aktiven `core`, `runtime` und `learning` funktional ersetzt
waren. Die historischen Dateien bleiben über Git-Commit `c4e80dd`
wiederherstellbar.

Künftige Fähigkeiten wie Modulation, Eligibility Traces, lernbare Änderungen
des Fähigkeitsvektors, Strukturwachstum oder größere neuronale Systeme werden
jeweils als getestete Erweiterung der aktiven Architektur implementiert und
nicht durch Reaktivieren
der Altmodule.
