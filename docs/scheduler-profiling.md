# Isolierter Scheduler-Benchmark – 14. Oktober 2026

## Ergebnis und Entscheidung

Zeitstempel-Buckets verkürzen in diesem lokalen synthetischen Vergleich die
Scheduler-Gesamtlaufzeit bei mehreren Ereignissen pro Zeitstempel um etwa
16–30 %. Bei einem Ereignis pro Zeitstempel verlängern sie die Laufzeit jedoch
um **70,7 %**. Die Alternative wird deshalb **nicht in `Simulation` übernommen**.
Eine allgemeine Verbesserung des NERVA-Durchsatzes wurde hier nicht gemessen.

Die Messung macht die Batch-Verteilung zur entscheidenden offenen Größe:
Vor einer Änderung des produktiven Schedulers sollten größere, asynchron
aktive Netze mit Lernen und Topologieänderungen ihre tatsächlich auftretenden
Batch-Größen liefern. Eine reine Übernahme der bei synchronen Lasten schnelleren
Queue wäre durch diese Ergebnisse nicht gerechtfertigt.

## Vergleichte Implementierungen

- `flat-btree`: der unveränderte öffentliche `runtime::EventScheduler`, mit
  Ereignissen unter `(SimTime, insertion_sequence)` und einem zweiten
  `BTreeMap`-Index zum Auffinden stornierter Ereignisse.
- `bucket-btree`: ausschließlich im Beispiel implementierter Prototyp mit
  `BTreeMap<SimTime, BTreeMap<u64, E>>` und demselben Sequenz-Zeit-Index.
  Eine Batch-Entnahme entfernt einen ganzen Zeitstempel-Bucket. Die Batch-Größe
  ist direkt bekannt; separate Schlüsselallokation und Einzelentnahmen aus dem
  äußeren Baum entfallen. Dafür benötigen einzelne Zeitstempel eigene innere
  Bäume, auch wenn sie nur ein Ereignis enthalten.

Beide Varianten ordnen nach Zeit und Einfügesequenz, entfernen stornierte
Payloads sofort und verwenden keine Heap-Tombstones. Die Alternative ist ein
Messprototyp mit den benötigten Queue-Operationen, kein zusätzlicher öffentlicher
Runtime-Modus. Es kommen keine Abhängigkeiten hinzu.

## Arbeitsmengen und Mediane

Release-Build, Rust 1.98.1, AMD Ryzen 9 7900X, x86_64,
WSL2/Linux 6.6.87.2. Beide Varianten laufen im selben Prozess auf logischer
CPU 2. Die Tabelle zeigt den Median aus sieben uninstrumentierten Läufen je
Variante, nach je einem Aufwärmlauf. Die Reihenfolge der Varianten wechselt
zwischen den Messrunden.

| Synthetische Arbeitsmenge | Aktuelle Queue (ms) | Buckets (ms) | Laufzeitreduktion |
|---|---:|---:|---:|
| 128 Ereignisse, vier Zeitstempel | 0,014 | 0,010 | 28,2 % |
| 65.536 Ereignisse, 32 Zeitstempel | 13,247 | 9,227 | 30,4 % |
| 65.536 Ereignisse, 1.024 Zeitstempel | 13,559 | 11,395 | 16,0 % |
| 65.536 Ereignisse, je ein Zeitstempel | 17,328 | 29,582 | −70,7 % |
| 1.024 Zeitstempel, Hälfte per Handle storniert | 18,540 | 14,488 | 21,9 % |
| 1.024 Zeitstempel, Topologie-Stornierung per Prädikat | 14,169 | 11,831 | 16,5 % |
| 16 Wellen mit erneuerten Spike-Vorhersagen | 10,921 | 8,006 | 26,7 % |

Die ersten sechs Lasten planen vor der Entnahme alle Ereignisse ein.
Eine feste Permutation verteilt Einfügungen über die Zeitstempel; die
Handle-Stornierungen erfolgen ebenfalls in permutierter Reihenfolge.
Die Payloads sind echte `EventKind::SynapticArrival` mit deterministischen
IDs und Amplituden. Die Topologie-Last entfernt über `cancel_matching`
5.958 Ankünfte anhand ihrer Synapsen-ID. Sie führt keine Graphmutation aus.

Die Streaming-Last wechselt Einreihen, Stornieren und Entnehmen ab:
16 Wellen mit je 4.096 Ankünften auf 128 Zeitstempeln und 128 weit in der
Zukunft liegenden `IntrinsicSpike`-Vorhersagen. Vor der nächsten Welle werden
die bisherigen Vorhersagen storniert und durch neue ersetzt. Damit werden
67.584 Ereignisse eingereiht, 1.920 storniert und 65.664 in 2.176 Batches
ausgeliefert. Höchstens 4.224 Ereignisse stehen gleichzeitig in der Queue.
Die letzte Vorhersagegeneration wird am Ende ausgeliefert.

## Messumfang und Grenzen

Eingabegenerierung, Queue-Konstruktion, exakte Referenzvergleiche, CSV-Ausgabe
und abschließender Abbau der leeren Queue liegen außerhalb der Gesamtzeit.
Gemessen werden Queue-Operationen einschließlich ihrer internen Allokationen,
Ausgabevektoren und deren Freigabe, plus gemeinsame Schleifen/Buchhaltung und
`black_box`-Aufrufe. Graphzugriffe, Neuronenintegration, STDP und CUDA sind
nicht Bestandteil dieses isolierten Vergleichs.

Ein zusätzlicher instrumentierter Lauf je Variante trennt `enqueue`, `cancel`,
`matching` und `dequeue`; Uhrabfragen erfolgen pro Operationsblock, nicht pro
Ereignis. Diese Phasen sind **Einzelmessungen**, keine Mediane. Ihre Summe ist
wegen gemeinsamer Buchhaltung und Instrumentierung nicht exakt die Gesamtzeit.
Die uninstrumentierten `sample`-, `median`-, `min`- und `max`-Zeilen lassen
Phasenfelder leer, statt nicht gemessene Zeiten als null auszugeben.

Auch mit CPU-Bindung streuen die Läufe. Besonders die kurze 128-Ereignis-Last
enthält deutliche Ausreißer; Min-/Max-Bereiche größerer Lasten überlappen zum
Teil. Die Prozentwerte beschreiben lokale Beobachtungen und keine statistisch
abgesicherte Beschleunigung. Es wurden weder Speicherverbrauch noch
Hardware-Leistungszähler gemessen. Die Payload-Freigabe bei Stornierung wird
separat funktional geprüft.

## Korrektheit und Reproduktion

Vor jeder Messreihe werden beide vollständigen Ablaufspuren direkt verglichen:
Einfügesequenzen, Stornierungsergebnisse, Prädikat-Trefferzahlen, Batch-Grenzen,
alle geordneten Ereignisse mit Payload sowie Queue-Länge, aktuelle Zeit und
nächste Zeit nach jedem Operationsblock. Dafür wird **kein Fingerabdruck**
verwendet. Gemessene Läufe prüfen anschließend dieselben Ereignis-/Batch-Zahlen
und die Bilanz `eingereiht = storniert + ausgeliefert` bei leerer End-Queue.

Vier zusätzliche Tests decken diese vollständigen Lastvergleiche, Fehler ohne
Teilmutation, Sequenzüberlauf, sofortige Payload-Freigabe und dieselbe
Besuchsreihenfolge bei zustandsbehafteten Stornierungsprädikaten ab.
Projektvalidierung: **319 Tests bestanden**, drei explizite GPU-Hardwaretests
ignoriert; Clippy mit allen Features/Targets und `-D warnings` ohne Warnungen.
Die neuen Rust-Dateien bestehen `rustfmt --check`.

```bash
cargo build --release --example scheduler_profile
taskset -c 2 target/release/examples/scheduler_profile > /tmp/nerva-scheduler-profile.csv
cargo test --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
rustfmt --edition 2024 --check examples/scheduler_profile.rs examples/scheduler_profile/queues.rs
```

`taskset` setzt Linux und eine verfügbare logische CPU 2 voraus. Ohne Bindung
kann das Programm auch mit `cargo run --release --example scheduler_profile`
ausgeführt werden. `--offline` kann bei vorhandenem Cargo-Cache ergänzt werden;
die abgelegten Messungen und Prüfungen wurden offline ausgeführt.

Quellcode: [Messprogramm](../examples/scheduler_profile.rs),
[Queue-Prototyp und Tests](../examples/scheduler_profile/queues.rs).
Rohdaten: [alle Einzelwerte und Phasen](benchmarks/scheduler_profile_2026-10-14.csv),
[Median-/Streuungsvergleich](benchmarks/scheduler_profile_2026-10-14_comparison.csv).
Im Vergleich bedeutet `reduction_percent = 100 * (1 - bucket_median / flat_median)`;
negative Werte bedeuten eine Verlängerung. Der Vergleich lässt sich aus den
`sample`-Zeilen als Median, Minimum und Maximum je Last/Queue berechnen.

## Anschlussversuch

Die damals offene Batch-Verteilung wird inzwischen mit einem
[größeren asynchronen Entwicklungsversuch](self-organisation-profiling.md)
gemessen. Auch bei 8.192 Neuronen bleiben etwa 61 % der Batches Einzelereignisse;
die strukturelle Buchhaltung wird dabei zu einem wesentlichen Kostenblock.
