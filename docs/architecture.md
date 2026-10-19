# Aktive Architektur und Konsolidierungsentscheidungen

**Stand:** 13. Oktober 2026

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
| 4. Numerik und Ausführung | `compute`, `runtime` | getrennte numerische Zellverarbeitung und deterministische Orchestrierung, Batches, Propagation und Simulation |
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

## Grenze zwischen Runtime und Compute

`Simulation` verwendet standardmäßig `CpuCompute` hinter dem objektfähigen
`compute::ComputeBackend`. Bestehende Konstruktoren und `Simulation<R>` bleiben
erhalten; `with_compute_backend` wählt eine andere Implementierung. Der Backend-
Zustand ist keine zweite Wahrheit über das Netzwerk.

Die Runtime verarbeitet einen vollständigen Zeitstempel in diesen Phasen:

1. vollständige Ereignisvalidierung;
2. betroffene Zellen, Wartungszellen und fällige intrinsische Checks sammeln;
3. betroffene Zellen durch Compute analytisch auf den Zeitpunkt fortschreiben;
4. Ankünfte protokollieren, Transmission und Utility-Eligibility erfassen,
   Prä-STDP ausführen und Eingangsamplituden sammeln;
5. fällige Vorhersagebuchungen freigeben, dann durch Compute Eingänge kanonisch
   summieren und jeden Feuerkandidaten genau einmal integrieren;
6. Spikes protokollieren;
7. Post-STDP und anschließend Post-Spike-Utility ausführen;
8. lokale Homeostase und ihre Folgetermine bearbeiten;
9. Propagation mit den nun gültigen Gewichten planen;
10. intrinsische Vorhersagen der betroffenen Zellen erneuern.

Die getrennten Backend-Methoden `advance_neurons` und `integrate_neurons` sind
notwendig: Prä-STDP muss den bereits fortgeschriebenen Zellzustand sehen, bevor
simultane Eingänge integriert werden. Diese Hooks werden daher nicht über eine
einzige undurchsichtige `process_batch`-Backend-Methode verschoben.

### Dichte Neuronenrepräsentation und flache Eingänge

Das Backend erhält `DenseNeuronBatch` mit drei zusammenhängenden Listen:
stabilen `NeuronId`s, unveränderlichen `ComputeNeuronParams` und veränderlichen
`ComputeNeuronState`-Datensätzen. Dieselbe Zeile bezeichnet in allen Listen
dieselbe Zelle. Nur die Zustandsliste ist für das Backend beschreibbar;
`&mut Neuron`, `Network`, Synapsen, Scheduler, I/O und Lernregeln werden nicht
übergeben. Arbeitsmengen enthalten eindeutige, aufsteigend geordnete IDs.
Wartungszellen werden fortgeschrieben, erhalten jedoch ohne Input oder
intrinsischen Termin keinen Feuercheck.

Die pointerfreien C-Layout-Datensätze sind in `core::dynamics` als
`NeuronDynamicsState` und `NeuronDynamicsParams` definiert und werden von
`compute` unter ihren Compute-Namen weiter-exportiert. Dadurch importiert
`core` kein Backend. Zustände enthalten Potential, veränderliche Basisschwelle,
Spuren, Refraktärzeit, alle intrinsischen Ströme, Spikezahl und lokale
Homeostasewerte einschließlich Zeitanker. Optionale Spikezeiten werden durch
`u64` plus `u32`-Gültigkeitskennzeichen dargestellt, sodass null und `u64::MAX`
keine Sentinel-Bedeutung erhalten. Parameter sind vollständig abgeflacht,
einschließlich der intrinsischen Fähigkeiten. Die konfigurierte Schwelle
bleibt getrennt von der veränderlichen Basisschwelle.

`Neuron::numerical_state` und `numerical_params` exportieren diese Daten;
`apply_numerical_state` validiert importierten Zustand, ohne Identität,
Geometrie, Polarität, Rolle oder Konfiguration zu ändern. `Neuron` und
`CpuCompute` verwenden dieselben analytischen Kernel in `core::dynamics`.
Die vorhandene Neuronendarstellung und ihre Debug-Ausgabe bleiben erhalten,
ebenso die CPU-Reihenfolge und die bisherigen mathematischen Ausdrücke.

`FlatNeuronInputs` enthält IDs, validierte `u64`-Offsets mit End-Sentinel und
ein einziges Amplitudenarray. Leere Zeilen stehen für intrinsische Feuerchecks.
IDs und Zeilen stimmen exakt mit dem Integrationsbatch überein. Das CPU-Backend
sortiert jede Zeile weiterhin kanonisch, summiert in `f64` und erhält signierte
Summe, Gesamtbetrag und inhibitorischen Betrag getrennt. Clamping und
`f32`-Konvertierung bleiben unverändert. Die Summierung erfolgt in Zellreihenfolge
während der Integration: Ein Überlauf einer späteren Zeile darf bereits
integrierte frühere Zellen nicht rückwirkend unverändert machen.

### Veröffentlichung und Fehler

Vor jedem synchronen Backend-Aufruf exportiert die Runtime nur die benötigten
Zellen neu. Danach schreibt sie die Ergebnisse zurück, bevor Lern-Hooks
weiterlaufen. Die Hostsynchronisierung prüft alle Ziele, unveränderte Parameter
und den unveränderten Quellzustand einschließlich der Fließkomma-Bitmuster.
Bei Erfolg werden alle Ergebniszustände vor der ersten Veröffentlichung
validiert. Veraltete Exporte oder ungültige Ergebnisse überschreiben keinen
autoritativen Zellzustand. Ein zusätzlicher Host-Snapshot der exportierten
Zellen ermöglicht diese Prüfung; er ist kein persistenter Compute-Cache.

Bei einem numerischen Fehler werden die entstandenen Teilzustände synchron
übernommen, bevor die Runtime poisoned wird. Das erhält den bisherigen
Fehlervertrag, einschließlich erfolgreicher früherer Zeilen und eventueller
Teiländerungen der fehlschlagenden Zelle. Bestehende numerische Fehler werden
auf ihre bisherigen `SimulationError`-Varianten abgebildet. Neue Layout- und
Synchronisationsfehler erscheinen als `SimulationError::Compute`.

Die Backend-Phasen bleiben `advance_neurons` und `integrate_neurons`; ihre
Parameter wurden von der bisherigen Objektansicht `NeuronStates` und
`NeuronInputs` auf `DenseNeuronBatch` und `FlatNeuronInputs` umgestellt.
Backend-Implementierungen müssen entsprechend angepasst werden. Konstruktoren
von `Simulation`, `Simulation<R>`, `with_compute_backend` und I/O bleiben
unverändert.

`tests/compute_regression.rs` vergleicht Standard- und explizites CPU-Backend
gegen vor der Extraktion aufgezeichnete vollständige Zell-/Synapsenzustände,
Ereignislogs, Post-Traces, Reports und ausstehende Uhr-/Vorhersagetermine.
Die Fälle enthalten Lernen, eingefrorenes Lernen, aktive intrinsische Dynamik,
Homeostase sowie Live-Synapsenänderungen. Das Fixture verwendet die aktuelle
Debug-Darstellung als Testformat, nicht als öffentliche Serialisierung.
`tests/compute_backend.rs` prüft Backend-Auswahl, geordnete Arbeitsmengen,
Live-Zelländerungen, Fehlerübersetzung und die eigenständige Kompilierbarkeit
von `compute` ausschließlich mit den unteren Schichten.

`tests/dense_compute.rs` prüft vollständige Zustands-Roundtrips mit aktiven
intrinsischen Fähigkeiten, exakte große Zeitwerte, optionale Spikezeiten,
Input-Layouts, veraltete Exporte, ungültige Backend-Ergebnisse und die
Teilzustände bei fehlgeschlagenem Fortschreiben beziehungsweise Integrieren.

### Opt-in Profiling und CUDA-Prototyp

`Simulation::with_profiling` beziehungsweise `enable_profiling` aktiviert
`RuntimeProfile`. Ohne Aktivierung gibt es keine Uhrabfragen. Das Profil hält
konstant viele Histogrammklassen und Phasensummen statt einzelner Event-Samples:
Advance-/Integrationsgröße, Eingänge pro Zelle, Synapsenankünfte pro Zeitstempel,
Spikes pro abgeschlossenem Zeitstempel sowie ausgehende Kandidaten pro Spike.
Null wird getrennt von 1–8, 9–32, 33–128, 129–512 und 513+ gezählt.

Hostzeiten trennen Export, Compute und Veröffentlichung. Weitere Phasen messen
Ankünfte, Prä-/Post-Lern-Hooks, Utility, Wartung, Propagation und Vorhersagen.
`Batch` umfasst die Verarbeitung; `PreStdp` ist innerhalb von `Arrivals`
enthalten. `PropagationPlan` misst Graphzugriffe und Amplitudenberechnung,
`PropagationEnqueue` das Einreihen pro Spike; beide sind bereits in
`Propagation` enthalten und dürfen beim Summieren nicht doppelt gezählt werden. Messungen
enthalten Instrumentierungsaufwand und sind keine Device-Kernelzeiten.
Fehlgeschlagene Phasen bleiben sichtbar; fehlgeschlagene Batches liefern keine
Probe in das Histogramm erfolgreich abgeschlossener Spikezahlen. Ein
Scheduler-Sicherheitsstopp vor dem Entfernen eines Batches zählt nicht als
gestarteter Batch. `take_profile` gibt die Daten zurück und deaktiviert die
Messung. Weder Wall-Zeiten noch Histogramme steuern den simulierten Ablauf.

`M0Experiment::run_profiled` gibt das unveränderte deterministische Ergebnis
und ein separates Profil zurück. Es aggregiert Training und beide gefrorenen
Probes. `examples/compute_profile.rs` misst alle sechs M0-Seeds und vier Gruppen
sowie ausdrücklich synthetische synchronisierte Lasten. Regressionstests
vergleichen Profiling an/aus gegen dieselben eingefrorenen Zustandsreferenzen.

Das optionale Feature `cuda` ergänzt `CudaCompute`, ohne die beiden
Backend-Methoden oder I/O-Grenzen zu ändern. Die Bindings kommen von der
optionalen Abhängigkeit `cudarc`; der Default-Build bleibt ohne externe Crates.
CUDA-Treiber und NVRTC werden dynamisch geladen. Fehlende Bibliotheken,
unerreichbare Geräte sowie Transfer-/Kernel-Fehler werden als
`ComputeError::Backend` beziehungsweise `SimulationError::Compute` gemeldet.
Initialisierung und NVRTC-Kompilation passieren im Konstruktor, außerhalb der
gemessenen Simulationsphasen.

Der Prototyp übersetzt Host-AoS feldweise in eigene SoA-Arrays aus `f32` und
`u64`. Keine Rust-Struktur wird mit ihrem Padding als Device-ABI übertragen.
Der Kernel verwendet einen Thread pro Zelle, exakte ganzzahlige Zeitdifferenzen,
die beiden LIF-/intrinsischen Rechenpfade, lokale Spuren und Spike-Flags.
FMA-Kontraktion und Fast Math sind deaktiviert; Blockgröße 64 berücksichtigt
den Registerbedarf der intrinsischen Dynamik. Neue Device-Puffer werden pro
Aufruf allokiert, hochgeladen und nach dem Kernel wieder heruntergeladen.
Die Eingänge werden vorab auf dem Host mit der unveränderten kanonischen
Reihenfolge zusammengefasst. Scheduler, STDP, Utility, Homeostase, Vorhersagen,
Synapsenpropagation und Entwicklung bleiben CPU-Aufgaben.

Numerische Fehlerflags werden nicht als parallel entstandene Teilzustände
veröffentlicht: Der Backend wiederholt dann den ursprünglichen CPU-Fehlerpfad
auf dem unveränderten Ausgangsbatch. Damit bleiben Fehleridentität und die
Mutationen bis zur ersten fehlschlagenden Zeile erhalten. Eine numerische
GPU-Fehlermeldung, die sich auf CPU nicht reproduziert, ist ein Backendfehler.
Transfer- und Treiberfehler führen niemals zu einem stillen CPU-Erfolg.

CPU bleibt die bitgenaue Referenz. CUDA-Zustände werden in den Tests mit
`|gpu-cpu| <= 2e-5 + 2e-6 * max(|gpu|, |cpu|)` verglichen; Zeitstempel, Zähler,
Gültigkeitsflags und Spikeentscheidungen müssen in den Testfällen exakt
übereinstimmen. Die Tests prüfen 8.192 reproduzierbar erzeugte Zustände,
GPU-Replay, eine echte Simulation mit STDP/Utility sowie Fehlerpfad und leere
Batches. Sie laufen nur bei explizitem `--ignored` mit realer GPU und NVRTC;
ein normaler Testlauf meldet sie als ignoriert. Toleranzen einzelner Zustände
garantieren keine identische Trajektorie nahe einer Feuerschwelle.

Die lokalen [Messungen](profiling.md) zeigen M0-Advance-Batches mit maximal vier
Zellen. Auf dieser Last bleibt CPU die sinnvolle Referenzausführung; der
synchrone CUDA-Prototyp dient dem Korrektheitsnachweis. Ein dichter
Synapsengraph/CSR, `TopologyDelta`, persistenter Device-Zustand und GPU-Lernen
bleiben spätere, durch repräsentative Messungen zu begründende Schritte.
`Network` behält seine `BTreeMap`-Darstellung und stabilen IDs; die Compute-API
und I/O-Grundarchitektur bleiben unverändert.

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
`RootChannel` enthält deshalb nur `ChannelId`: Die Root deklariert vorhandene
Kanäle, `Mapping` besitzt allein `(RootId, ChannelId) → FiberId`, und die
`Fiber` besitzt Neuron-Endpunkt, Delay und Gain. Eine zweite Zuordnung eines
Root-Kanals wird abgelehnt, auch bei verschiedenen Motorneuronen oder
unterschiedlichen Transportrichtungen. Der Architekturtest kompiliert `roots`
ohne Neuronen- und Nervenmodule und prüft `RootChannel` als erschöpfend
destrukturierbaren Ein-Feld-Typ. `RootRegistry` bleibt eine optionale Verwaltung
von Metadaten; der feste I/O-Pfad braucht keine Registry.

`Mapping::map_sensory` und `Mapping::map_motor` nehmen `&Root` entgegen und
validieren vor jeder Änderung den deklarierten Kanal und die passende
Root-Richtung. `MappingError::UnknownChannel` beziehungsweise
`MappingError::RootDirectionMismatch` melden Verletzungen ohne eine Zuordnung
anzulegen. Fiber-Existenz und Fiber-Richtung werden anschließend geprüft.
Root-Identitäten und Kanaldeklarationen bleiben nach dem Binden stabil; die
Prüfung findet bei der Zuordnung statt. Referenzversuche und Routing-Tests
bauen deshalb zuerst die Root und binden dann ihre Kanäle.

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
Stille keine zusätzlichen Ereignisse.

`RateSensoryTransducer` ist der erste kontinuierliche Encoder der Bibliothek.
Ein Rezeptorwert ist eine Rate in Hertz, die pro Kanal bis zur nächsten
Änderung gilt. Der Encoder integriert `r * Δt / 1_000_000` und erzeugt beim
Erreichen einer ganzen Phase einen Spike der Amplitude `1.0`. Neue Kanäle
starten ohne angesammelte Phase; Ratewechsel erhalten die Teilphase. Null
pausiert die Integration, und ein späterer positiver Wert setzt sie fort.
Erlaubt sind endliche Raten von 0 bis 1.000.000 Hz, höchstens ein Spike pro
Kanal und Mikrosekunde. Physikalische Werte werden vom Adapter in Raten
umgerechnet. Die Bibliothek interpretiert keine Sensormodalität.

Spike-Zeitpunkte werden analytisch relativ zur letzten Rateänderung mit `f64`
vorhergesagt und auf ganze Mikrosekunden aufgerundet. So werden weder absolute
Zeitstempel in Fließkommazahlen umgewandelt noch gerundete Perioden aufaddiert.
Advance-Horizonte verändern die Phase nicht; identische Eingänge erzeugen bei
jeder Aufteilung denselben Spike-Zug. Eine Rateänderung bei `t` wird nach den
bereits durch die alte Rate bei `t` fälligen Spikes angewendet. Änderungen mit
gleichem Zeitstempel behalten ihre Einfügereihenfolge, simultane Spikes
verschiedener Kanäle werden nach `ChannelId` sortiert. Vorhersagen außerhalb
der `SimTime`-Domäne erzeugen keine Ereignisse.

`PoissonSensoryTransducer::new(seed)` kodiert dieselben gehaltenen Hertz-Werte
stochastisch. Ein eigener SplitMix64-Strom pro Kanal wird aus Seed und
`ChannelId` abgeleitet; zusätzliche Kanäle oder eine andere Erzeugungsreihenfolge
verändern diesen Strom nicht. Aus gleichverteilten Werten strikt zwischen 0
und 1 entstehen exponentielle Wartehazards `-ln(U)`. Eine Rateänderung erhält
den noch ausstehenden Hazard; null Hertz pausiert dessen Verbrauch. Neue
Zufallswerte werden nur bei der Kanalinitialisierung und einer Spike-Emission
gezogen, nie durch einen Advance-Horizont oder ein redundantes Sample.

Die Zeitintegration erfolgt ereignisgetrieben mit `f64`-Differenzen zwischen
ganzzahligen Zeitstempeln. Spike-Zeitpunkte werden auf Mikrosekunden aufgerundet.
Mehrere kontinuierliche Poisson-Ankünfte in derselben Mikrosekunde erzeugen
mehrere Spikes mit gleichem Zeitstempel; sie werden nicht auf einen Spike pro
Mikrosekunde begrenzt. Hertz-Bereich, Amplitude und Sortierung zwischen Kanälen
entsprechen dem Rate-Encoder. Der gleiche Seed und die gleiche Eingangshistorie
liefern bei jeder Aufteilung der Horizonte identische Spike-Zeitpunkte und
denselben weiteren Zufallsstrom.

`tests/io_contract.rs` prüft direkte Transduktion, gehaltene Rate und Poisson über
Trait-Objekte im vollständigen neuronalen Pfad bis zum Effector.
`tests/rate_transduction.rs` prüft nichtganzzahlige Perioden, Ratewechsel,
Pausen, Kanäle, Replay, Fehler und Zeitbereichsgrenzen.
`tests/transduction_time.rs` prüft den gemeinsamen Zeitvertrag und demonstriert
zusätzlich Motorzerfall mit einer lokalen Test-Implementierung.
`tests/poisson_transduction.rs` prüft Seed-Replay, unabhängige Kanalströme,
Ratewechsel, Pausen, Fehlerrückgabe und Grenzen der Zeitdomäne. Ein Test mit
festem Seed prüft außerdem Mittelwert und Variationskoeffizient der
exponentiellen Wartezeiten. Fest erwartete I/O-Zeitpunkte schützen den
reproduzierbaren Poisson-Strom im vollständigen neuronalen Pfad.

Die I/O-Grundarchitektur ist damit als Basisvertrag festgelegt. Weitere
Sensoren und Aktuatoren verwenden `io`-Adapter und konkrete Transducer;
Änderungen an den öffentlichen Grenztypen und Traits werden als eigener
API-Umbau behandelt. Der nächste experimentelle Schritt ist ein kontinuierlicher
Sensor, zwei Motorpopulationen und ein simulierter Aktuator.

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
