# Aktive Architektur und Konsolidierungsentscheidungen

**Stand:** 3. Oktober 2026

Dieses Dokument beschreibt die tatsächlich über `src/lib.rs` kompilierte
Architektur. Die ausführliche M0-Logik und die experimentellen Kriterien stehen
weiterhin in `NERVA_SLICE_ARCHITECTURE.md`.

## Bibliothek als Projektziel

NERVA wird als wiederverwendbare Rust-Bibliothek entwickelt. Eine einbindende
Anwendung erstellt das Netzwerk, wählt die lokalen Lernregeln, liefert
zeitgestempelte Eingänge und steuert die Ausführung über `Simulation`.
Beobachtungsdaten und Netzwerkzustand stehen ihr zur Auswertung zur Verfügung.

Die öffentliche API bildet diese Aufgaben unabhängig von einem bestimmten
Versuch ab. M0 und BitWorld dienen als Referenzanwendungen und zur Prüfung der
Bibliothek. Ausführbare Einstiege liegen unter `examples/`; die Bibliothek wird
über `src/lib.rs` eingebunden. Das Beispiel `minimal_network` zeigt den Einstieg
mit zwei verbundenen Neuronen.

## Aktiver Datenpfad

```text
Außenwelt
    ↓ Receptor / ReceptorSignal
SensoryTransducer
    ↓ ChannelSpike
SensoryRoot → Nerve / Fiber → sensorisches Neuron
    ↓
runtime → core::Network + lokale learning-Hooks
    ↓
motorisches Neuron → Nerve / Fiber → MotorRoot
    ↓ MotorOutput
MotorTransducer
    ↓ EffectorSignal / Effector
Außenwelt
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
| 5. Schnittstellen | `io`, `roots`, `nerves`, `transduction` | geräteneutrale Werte, feste Anschlussstellen, Transport und Übersetzung |
| 6. Versuche und Beobachtung | `experiment` mit `experiment::m0`, optional `metrics`, `debug`, `visualization` | Stimulation, Versuchssteuerung und ausschließlich lesende Inspektion |

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
`Activity`, `Distance` und `EnergyCost` sind die Typdomänen der schrittweisen
Migration. Der aktive M0-Code verwendet für Membranpotential, Schwelle und
Eingangszusammenfassung weiterhin `f32`.

`Weight` kapselt seinen skalaren Wert privat: Jeder Konstruktor validiert
Endlichkeit und Nichtnegativität, saturierende Arithmetik kann in keiner
Richtung nach unendlich oder negativ entgleiten. Der vorzeichenbehaftete
Beitrag einer Verbindung ist eine eigene Domäne: `Polarity::apply_to_weight`
erzeugt aus `Weight` ein `SignalStrength`, `Synapse::signed_amplitude` und
`Synapse::effective_amplitude` geben nie ein `Weight` zurück. Der Typ kann
damit nicht mehr gleichzeitig Stärke und Amplitude bedeuten.

Der erste dokumentierte Pfad ist abgeschlossen migriert:

```text
Weight
→ core::Synapse
→ learning::WeightBounds / PairStdp
→ runtime::ObservationEvent::WeightChanged
→ experiment::M0GroupResult Gewichtsvektoren
→ debug::SynapseSnapshot
```

Die Amplitude eines Spike-Ausbreitungsereignisses wird bei der Emission
einmalig aus dem typisierten Gewicht als `SignalStrength` abgeleitet; in der
Ereignis-Schnittstelle von `core::EventKind` ist sie noch `f32`, bis der
Ereignispfad eigene Invarianten erhält. Als nächster sinnvoller Pfad ist
vorgesehen:

```text
SignalStrength
→ core::EventKind::ExternalInput / SynapticArrival
→ runtime::ObservationEvent
→ core::Neuron::integrate_input
```

So existiert während einer Migration pro Invariante weiterhin genau eine
verbindliche Darstellung.

## Kontrollierte Laufzeitmutation und Poisoning

Ein laufender `Simulation`-Zustand wird nie über freies `network_mut`
verändert: Die Runtime verwaltet Scheduler, neuron-lokale Homeostase-Uhren
und intrinsische Spike-Vorhersagen gemeinsam. Die beiden Buchungsregister
liegen in `runtime`, nicht im `Neuron`; Core-Zellzustand bleibt damit frei von
Scheduler-Zeitstempeln und Einfüge-Sequenzen. Dafür stellt sie kontrollierte,
transaktionale Methoden bereit:

- `Simulation::add_neuron` fügt die Zelle in Netz, Uhr und Vorhersage
  konsistent ein; ein Fehler stellt Netz, Scheduler und Buchungen zurück.
- `Simulation::remove_neuron` entfernt nur eine isolierte Zelle, hebt alle
  noch wartenden zielbezogenen Events auf und lässt bei abgelehnter Topologie
  jede Buchung unverändert.
- `Simulation::add_synapse` fügt eine validierte Verbindung zwischen bereits
  vorhandenen Endpunkten hinzu.
- `Simulation::remove_synapse` entfernt die Verbindung und storniert alle
  bereits fliegenden `SynapticArrival`-Events dieser Synapse, bevor sie eine
  spätere Batch-Validierung erreichen können.
- `Simulation::update_neuron` wendet eine lokal begrenzte Mutation an und
  erneuert die autonome Vorhersage der Zelle.
- `Simulation::update_synapse` ändert nur lokale Synapsenwerte.

Ein fehlschlagender Batch ist nicht transaktional: Der Scheduler entfernt ihn,
bevor die Verarbeitung vollständig ist. Ein fataler Fehler kann daher
teilweise angewendete Zustände hinterlassen. Solch eine Simulation wird
gepoisoned (`is_poisoned`, `poison_reason`); jede weitere Ausführung,
Planung oder Mutation wird mit `SimulationError::SimulationPoisoned`
abgelehnt. Sicherheitsstopps vor der ersten Mutation, etwa das
Batch-Limit des Schedulers, poisonen nicht.

## Pair-STDP: Nearest-Neighbor-Variante

`learning::PairStdp` koppelt jeden Spike ausschließlich mit dem zuletzt
gespeicherten gegenüberliegenden Spike innerhalb des endlichen
Paarfensters (nearest-neighbor pairing). Die gepflegten Prä- und
Post-Traces erzwingen die Fensterzugehörigkeit exakt, bestimmen die
Gewichtsänderung aber nicht all-to-all über die gesamte Historie. Eine
all-to-all-Trace-Formulierung ist eine bewusst offene Erweiterung.

## Einordnung des M0-Ergebnisses

M0 ist ein Engineering-Kriterium: Es zeigt, dass der konkrete Aufbau unter
sechs festen, gepaarten Seeds deterministisch das deklarierte Verhalten
reproduziert. Es ist kein wissenschaftlicher Beweis für Robustheit über
größere Parameterbereiche, andere Sequenzen, andere Netzgrößen oder nicht
vorher festgelegte Seeds.

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

Die versuchsspezifischen Grenztypen `Pattern`, `Observation` und `Action` sowie
deren Encoder, Decoder und Umgebungen liegen in `experiment::m0`; `experiment`
stellt sie für die Referenzversuche weiter bereit. Die allgemeinen Module
`io`, `roots`, `nerves` und `transduction` kennen keine M0-Typen. Ein rekursiver
Architekturtest schützt diese Grenze.

## Allgemeine I/O- und Transduktionsgrenze

`io::Receptor` liefert zeitgestempelte `ReceptorSignal`-Werte; `io::Effector`
nimmt `EffectorSignal`-Werte entgegen. Beide kennen nur `ChannelId`, Werte und
`SimTime`, keine Neuron-IDs. `SensoryRoot` und `MotorRoot` kapseln die generische
Root-Metadatenstruktur mit ihrer jeweiligen Richtung. Die feste physische
Verbindung zwischen Kanal und Neuron gehört ausschließlich in `nerves::Mapping`.

`SensoryTransducer` und `MotorTransducer` haben jeweils zwei Operationen:

- `push` nimmt einen zeitgestempelten Eingang entgegen und puffert ihn.
- `advance_until(until, &mut output)` verarbeitet Eingänge und verstrichene
  Zeit bis einschließlich `until` und hängt Ausgaben an den übergebenen Puffer.

Beide geben `Result<(), TransductionError>` zurück. Alle Eingänge bis zu einer
Grenze müssen vor dem Advance-Aufruf vorliegen. Nach einem erfolgreichen
Advance ist dieses inklusive Zeitintervall abgeschlossen: Eingänge bei oder
vor der letzten Grenze werden abgelehnt. Zukünftige Eingänge dürfen in
beliebiger Reihenfolge eintreffen; die Verarbeitung erfolgt chronologisch,
bei gleichem Zeitstempel in Einfügereihenfolge. Rückwärts laufende Grenzen
werden ohne Zustandsänderung abgelehnt. Ein wiederholter Advance auf dieselbe
Grenze erzeugt keine weitere Ausgabe. Ausgabezeitpunkte liegen nie nach der
angegebenen Grenze; spätere gepufferte Eingänge bleiben erhalten.

Ein kontinuierlicher sensorischer Wert gilt ab seinem Zeitstempel bis zur
nächsten Änderung. Auch ohne weitere Eingänge kann `advance_until` daraus
zeitliche Aktivität erzeugen. Sensorische Spike-Zeitpunkte müssen bei gleicher
Eingangshistorie unabhängig von der Aufteilung der Horizonte bleiben; für
seed-basierte Encoder gilt das auch für den Zufallszustand. Motor-Decoder
können bei neuen Horizonten ihren aktuellen, analytisch zerfallenden Wert
ausgeben. Die Zwischenmessungen hängen dann von den gewählten Horizonten ab,
der Zustand an gemeinsamen Grenzen muss bis auf numerische Genauigkeit
übereinstimmen. Der Host steuert diese Grenzen ohne globalen Tick.

`DirectSensoryTransducer` übersetzt einen endlichen, positiven Rezeptorwert
in einen Spike gleicher Amplitude. `DirectMotorTransducer` übersetzt einen
endlichen, positiven Motoroutput in einen Puls gleichen Werts. Kanal und
Zeitstempel bleiben erhalten; ungültige Amplituden erzeugen einen Fehler und
ändern den Zustand nicht. Die direkten Implementierungen erzeugen während
Stille keine zusätzlichen Ereignisse. `tests/io_contract.rs` prüft diese
beiden Implementierungen über Trait-Objekte im vollständigen neuronalen Pfad;
`tests/transduction_time.rs` prüft Zeitgrenzen und demonstriert kontinuierliche
Rate sowie Motorzerfall mit ausschließlich lokalen Test-Implementierungen.

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

`NervaConfig::neuron` beschreibt wie bisher eine gemeinsame Zellklasse für die
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
