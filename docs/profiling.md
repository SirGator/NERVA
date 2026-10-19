# Compute-Profiling und CUDA-Prototyp

**Messdatum:** 13. Oktober 2026

Lokaler, instrumentierter Release-Lauf auf AMD Ryzen 9 7900X / x86_64, Rust 1.98.1. GPU-Prüfung auf NVIDIA GeForce RTX 3060 (12 GB), Treiber 610.47.03, NVRTC 12.8.93. Die Bibliothek benutzt optionale cudarc-12.8-Bindings.

## CPU-Arbeitsmengen

M0 umfasst sechs feste Seeds, alle vier Gruppen, Training und jeweils beide gefrorenen Probes. Die übrigen Zeilen sind synthetische synchronisierte Eingaben über 16 Runden; jede Zelle besitzt bis zu vier feste ausgehende Verbindungen. Sie sind keine Messung einer bereits vorhandenen großen NERVA-Anwendung.

| Last | Batches | Max. Advance | Max. Integration | Advance gesamt (ms) | Integration gesamt (ms) | Propagation gesamt (ms) | Batch gesamt (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 12072 | 4 | 4 | 0.750 | 0.629 | 2.494 | 19.655 |
| Synthetisch 1 | 16 | 1 | 1 | 0.001 | 0.001 | 0.001 | 0.020 |
| Synthetisch 8 | 32 | 8 | 8 | 0.005 | 0.006 | 0.054 | 0.181 |
| Synthetisch 128 | 32 | 128 | 128 | 0.074 | 0.071 | 1.657 | 6.170 |
| Synthetisch 2048 | 32 | 2048 | 2048 | 1.299 | 1.049 | 30.287 | 120.083 |

M0 liefert **12.072 Advance-Batches mit jeweils 1–4 Zellen**. Unter den 12.072 Integrationsbatches sind 1.824 leere Wartungsbatches und 10.248 Batches mit 1–4 Zellen. Es gibt höchstens zwei Eingangsamplituden pro integrierter Zelle. Größere Compute-Batches treten in diesem Referenzversuch nicht auf.

| M0-Histogramm | 0 | 1–8 | 9–32 | 33–128 | 129–512 | 513+ |
|---|---:|---:|---:|---:|---:|---:|
| Advance | 0 | 12.072 | 0 | 0 | 0 | 0 |
| Integration | 1.824 | 10.248 | 0 | 0 | 0 | 0 |

Die [Rohdaten](benchmarks/compute_profile_2026-10-13.csv) enthalten außerdem Eingänge pro Zelle, Synapsenankünfte, Spikezahlen, ausgehende Kandidaten pro Spike sowie alle Phasen einschließlich Prä-/Post-Lern-Hooks. `PreStdp` liegt innerhalb von `Arrivals`; `Batch` umfasst die Verarbeitung. Diese Zeiten dürfen nicht als disjunkte Phasen addiert werden. Lern-Hook-Zeiten enthalten Graphsuche und Buchhaltung, auch bei NoPlasticity. Export und Veröffentlichung sind getrennt vom Backend gemessen.

## Echter GPU-Vergleich

Drei explizite GPU-Tests wurden erfolgreich ausgeführt; Hardwaretests sind im normalen Testlauf absichtlich ignoriert.

- 8.192 reproduzierbar erzeugte Zustände mit neutralen und aktiven intrinsischen Fähigkeiten, gleichen und verschiedenen Zeitkonstanten, Refraktärphasen, großen Zeitstempeln und kanonischen Eingangssummen.
- Maximal beobachteter absoluter Skalarfehler: Advance `7.6293945e-6`, Integration `3.0517578e-5`. Alle Felder erfüllen `abs_error <= 2e-5 + 2e-6 * max(abs(cpu), abs(gpu))`. Integration erzeugt 827 auf CPU/GPU identische Spikes; diskrete Zustände stimmen exakt überein. Wiederholter GPU-Lauf reproduziert die Skalar-Bitmuster.
- Eine echte Simulation mit 16 Zellen, lokaler Pair-STDP und Utility: 48 Batches, 24 Spikes, identische Ereignislogs und Synapsenzustände; Zellzustände innerhalb derselben Toleranz.
- Leere Batches und der numerische Fehlerpfad einschließlich Refraktärüberlauf und partieller Zustandsänderungen stimmen mit CPU überein.

Im kleinen Simulationsvergleich betrug die instrumentierte Batch-Zeit in einem Release-Lauf **108,310 µs auf CPU und 35.192,660 µs mit CUDA** (48 Batches). GPU-Konstruktor, NVRTC-Kompilation und Netzwerkaufbau liegen außerhalb dieser Zeiten; Erstnutzung der Kernel sowie Allokation, Packing, Transfers und Synchronisierung innerhalb der Backend-Aufrufe sind enthalten. Das ist eine lokale Einzelmessung, kein belastbarer Geschwindigkeitsbenchmark. Sie zeigt die erwarteten Kosten dieses Prototyps bei kleinen Batches.

Zustandstoleranzen garantieren keine identischen Spike-Trajektorien nahe einer Schwelle und keine plattformübergreifende GPU-Bitgleichheit. Diese Tests sind ein begrenzter Funktionsnachweis auf der genannten Hardware.

## Konsequenz und Reproduktion

CPU bleibt für M0 die Standardausführung. Die Messungen rechtfertigen aktuell keinen persistenten GPU-Spiegel für diesen Versuch. Vor einer Optimierung sollte eine repräsentative größere Last vermessen werden; in den synchronisierten CPU-Lasten beanspruchen Graphsuche, Ankünfte und Propagation bereits wesentlich mehr Zeit als reine Neuronen-Compute-Aufrufe. CSR, GPU-STDP und inkrementelle Topologieupdates werden hier noch nicht gebaut.

```bash
cargo run --release --example compute_profile
cargo test --all-features --all-targets
cargo test --release --features cuda --test cuda_compute -- --ignored --nocapture --test-threads=1
```

CUDA setzt zugängliche NVIDIA-Hardware sowie die Treiber- und NVRTC-Bibliotheken im Bibliothekssuchpfad voraus. Die Kompilation des Rust-Features benötigt kein installiertes nvcc. In dieser Sitzung wurde NVRTC isoliert nach `/tmp/nerva-nvrtc` entpackt; weder die systemweite CUDA-Installation noch globale Python-Pakete wurden geändert. Der GPU-Aufruf verwendete zusätzlich:

```bash
LD_LIBRARY_PATH=/tmp/nerva-nvrtc/nvidia/cuda_nvrtc/lib:/usr/lib/wsl/lib \
  cargo test --release --features cuda --test cuda_compute -- --ignored --nocapture --test-threads=1
```

Die Pfade sind lokal und nicht Teil der Bibliothekskonfiguration. Das optionale Binding nutzt die [cudarc Driver-/NVRTC-Schnittstellen](https://docs.rs/cudarc/0.19.9/cudarc/); die NVIDIA-Pakete für NVRTC sind in der [CUDA-Python-Installation](https://nvidia.github.io/cuda-python/cuda-bindings/12.9.0/install.html) beschrieben.

## CPU-Propagation: Profilierung und Optimierung am 14. Oktober 2026

Der Profiler unterscheidet jetzt `PropagationPlan` (Graphzugriffe, Distanz,
Amplitude und Ankunftszeit) und `PropagationEnqueue` (Scheduler-Insertion).
Beide werden pro Spike gemessen und sind in `Propagation` enthalten.
Zusammen mit `Batch` und `PreStdp` sind das verschachtelte Zeiten, die nicht
erneut zu ihren Eltern addiert werden dürfen. Ohne Profiling werden weiterhin
keine Uhrabfragen ausgeführt.

Die Messung vor der Optimierung zeigte in den größeren Lasten mehr Zeit für
das Einreihen als für die Planung. Dabei prüfte `schedule_event` für jede
soeben geplante Übertragung erneut Synapse und Endpunkte. Der synchrone
Planer hatte diese Daten bereits geprüft, und zwischen Planung und Insertion
ändert sich weder Topologie noch Lernzustand.

Die CPU-Änderungen beseitigen diese wiederholten Graphprüfungen im internen
Propagationspfad. Der Scheduler prüft Zeit und Sequenz weiterhin; bei der
späteren Batch-Verarbeitung werden alle Ankünfte erneut validiert.
Ein batchlokaler Planungspuffer ersetzt die Allokation pro Spike. Erst nach
erfolgreicher Planung **aller** ausgehenden Übertragungen eines Spikes werden
dessen Ereignisse eingereiht, sodass ein Planungsfehler keinen Teilplan
veröffentlicht. Ankünfte bündeln Transmission und Utility-Eligibility über
denselben mutablen Synapsenzugriff. Post-STDP sammelt akzeptierte Synapsen und
deren ursprüngliche Identitäten/Gewichte in einem Durchlauf, erhält aber die
Validierung aller Kopien vor der Veröffentlichung und den Hook-Aufruf bei
leerer Eingangsmenge. Utility vermeidet einen weiteren Synapsensuchlauf.

### Wiederholter Vergleich

`examples/propagation_profile.rs` misst ausdrücklich **synthetische**
Ringgraphen mit dünn verteilten stabilen IDs, rund 20 % inhibitorischen Zellen,
nichtverschwindenden Distanzen und vier oder 32 ausgehenden Kandidaten.
Jede achte Verbindung ist deaktiviert (bei Fanout 32 bleiben 28 aktiv).
16 externe, synchrone Eingaberunden sind jeweils 100.000 µs getrennt;
Synapsen verzögern um 10 µs. Die auf den Fanout skalierten Gewichte bleiben
auch mit Lernen unterhalb einer selbsttragenden rekurrenten Anregung.
Die Lernvariante führt Pair-STDP und ExcitatoryCausalUtility aus; die andere
Variante friert Lernen vollständig ein. Alle großen Lasten erzeugen 32.768
Spikes und 32 Batches. Diese Lasten ergänzen M0, ersetzen aber keinen späteren
repräsentativen Anwendungsbenchmark.

Gemessen wurde mit Rust 1.98.1 im Release-Profil auf AMD Ryzen 9 7900X unter
WSL2/Linux 6.6.87.2. Beide Binärdateien liefen nacheinander auf logischer CPU 2.
Jede Last erhielt einen Aufwärmlauf, sieben **uninstrumentierte** Messläufe
und anschließend einen instrumentierten Lauf zur Phasenanalyse. Netzwerkbau,
Eingabeplanung, Fingerabdrücke und Abbau liegen außerhalb der gemessenen
Gesamtlaufzeit. Die Baseline entspricht dem angetroffenen Workspace vor den
CPU-Optimierungen, bereits mit denselben neuen Profilphasen und demselben
Messprogramm; sie entspricht nicht dem älteren Git-HEAD.

| Last | Median vorher (ms) | Median danach (ms) | Laufzeitreduktion |
|---|---:|---:|---:|
| 8 Zellen, Fanout 4, eingefroren | 0,175 | 0,166 | 5,5 % |
| 2.048 Zellen, Fanout 4, eingefroren | 92,686 | 87,378 | 5,7 % |
| 2.048 Zellen, Fanout 32, eingefroren | 632,383 | 593,691 | 6,1 % |
| 2.048 Zellen, Fanout 4, STDP + Utility | 138,487 | 125,949 | 9,1 % |
| 2.048 Zellen, Fanout 32, STDP + Utility | 1.218,813 | 1.114,518 | 8,6 % |

In den einzelnen instrumentierten Läufen der großen Lasten sinkt
`PropagationEnqueue` um 37–51 % und die gesamte `Propagation` um 18–41 %.
Die Phasenmessungen sind Einzelbeobachtungen, keine Mediane. Insbesondere
Graphplanung und Ankunftszeiten streuen und verbessern sich nicht in jeder
Einzelmessung. Auch die Gesamtzeiten haben überlappende Min-/Max-Bereiche:
Die angegebenen Median-Unterschiede sind lokale Beobachtungen, keine
allgemeine Geschwindigkeitsgarantie. Ein erster ungebundener Durchlauf
zeigte größere Unterschiede; die Tabelle verwendet bewusst den separaten
Vergleich auf einem festen CPU-Kern. Graphzugriffe, Ankünfte und Lernen
bleiben relevante Kosten für weitere Optimierungen.

Alle sieben Läufe je Variante und der instrumentierte Lauf liefern dieselben
Reports sowie identische Fingerabdrücke über den vollständigen Graphzustand
und das geordnete Ereignislog. Auch zwischen Vorher und Nachher stimmen sie
überein. Der Fingerabdruck basiert auf Rust-Debug-Ausgaben und `DefaultHasher`;
er ist nur eine Diagnose für diese Toolchain, kein beständiges Dateiformat
und kein Ersatz für Regressionstests.

Rohdaten: [vorher](benchmarks/propagation_profile_2026-10-14_before.csv),
[danach](benchmarks/propagation_profile_2026-10-14_after.csv),
[Median-/Streuungsvergleich](benchmarks/propagation_profile_2026-10-14_comparison.csv).
Die ersten beiden Dateien enthalten `sample`-, `summary`- und `phase`-Zeilen;
der Vergleich hat ein einheitliches CSV-Schema.

Reproduktion der aktuellen Variante:

```bash
cargo build --release --example propagation_profile
taskset -c 2 target/release/examples/propagation_profile
cargo test --all-features --all-targets
cargo clippy --all-features --all-targets
```

Validierung: 315 Tests erfolgreich, drei explizite GPU-Hardwaretests ignoriert;
Clippy ohne Warnungen. Die bestehenden eingefrorenen CPU-Referenzdaten bleiben
unverändert und werden exakt reproduziert. Zusätzliche Tests vergleichen den
gepufferten Pfad mit öffentlicher Planung plus validierter Insertion, prüfen
den Planungsfehler nach einem bereits erfolgreichen Teilplan und stellen
sicher, dass ungültige Post-STDP-Kopien keine Synapsen teilweise veröffentlichen.
Die neuen Profilphasen werden auf Zählung, Verschachtelung und Zusammenführung
geprüft. Die bearbeiteten Rust-Dateien bestehen `rustfmt --check`;
`cargo fmt --all -- --check` meldet weiterhin Formatunterschiede in bereits
vorher vorhandenen, hier nicht bearbeiteten Dateien.

## Isolierter Scheduler-Vergleich

Der nächste CPU-Untersuchungsschritt ist im
[Scheduler-Benchmark vom 14. Oktober 2026](scheduler-profiling.md) dokumentiert.
Er vergleicht die bestehende Queue mit Zeitstempel-Buckets bei synchronen und
verteilten Ereignissen, Stornierungen und laufend erneuerten Spike-Vorhersagen.
Die Alternative bleibt auf das Messprogramm beschränkt.

## Größere asynchrone Netze mit Strukturentwicklung

Der [Versuch mit 2.048 und 8.192 Neuronen](self-organisation-profiling.md)
ergänzt die synchronen Propagationslasten um Pair-STDP, Utility, lokale
Verbindungsbildung und Pruning bei zeitlich verteilten Eingaben. Er misst
vollständige Scheduler-Batches samt Einzelereignisanteil und Entnahmezeit.
Die zwölf Messreihen umfassen drei Seeds, zwei Größen und jeweils eine
Kontrolle mit fester Topologie. Details, Phasenkosten und Rohdaten stehen
im verlinkten Bericht.

## Pruning-Buchhaltung optimiert

Die im asynchronen Versuch gefundene lineare Mitgliedschaftssuche ist
[durch den vorhandenen Synapsenindex ersetzt](pruning-optimization.md).
Ein erneuter Vergleich mit identischer Last und drei Seeds misst sowohl die
Gesamtlaufzeit als auch die Pruning-Beobachtung aus sieben Wiederholungen.
Vollständige Zustands- und Ereignisdateien stimmen vor und nach der Änderung
überein; Rohdaten und Reproduktion stehen im verlinkten Bericht.
