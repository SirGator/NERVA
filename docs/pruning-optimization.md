# Pruning-Optimierung und A/B-Vergleich – 14. Oktober 2026

## Umsetzung

`PruningController::observe` verwendet jetzt den bereits vorhandenen
Synapsenindex, um Timer entfernter Verbindungen zu löschen:

```rust
self.states.retain(|id, _| network.synapse(*id).is_some());
```

Die zuvor aufgebaute Liste aller lebenden Synapsen entfällt. Für `T`
gespeicherte Timer und `S` lebende Synapsen sinkt der Aufwand dieser
Bereinigung von `O(S + T * S)` auf `O(T * log S)`; die zusätzliche
`Vec`-Allokation entfällt ebenfalls. `Network::synapse` greift auf die
maßgebliche `BTreeMap` zu. Timer, Utility-Berechnung, Pruning-Kriterien,
Iteration und Veröffentlichungsreihenfolge bleiben unverändert.

Die Produktionsänderung betrifft ausschließlich diese Bereinigung in
[pruning.rs](../src/development/pruning.rs). Eine zusätzliche Regression
prüft gleichzeitig entfernte Synapsen, unveränderte Timer verbleibender
Synapsen und neue dünn verteilte IDs mit eigener Mindestwartezeit.

## Erneuter Vergleich derselben Last

Der [asynchrone Selbstorganisations-Benchmark](self-organisation-profiling.md)
wurde vor **und** nach der Änderung neu ausgeführt: 2.048 und 8.192 Neuronen,
drei Seeds, jeweils mit Strukturentwicklung und als Kontrolle mit fester
Topologie. Pro Fall ein Aufwärmlauf, sieben Messläufe ohne Runtime-Profiling
und ein zusätzlicher instrumentierter Lauf. Eingabefolgen und Parameter
bleiben identisch. Die vorher abgelegten historischen Messungen werden nicht
als aktuelle Baseline wiederverwendet.

Rust 1.98.1, Release-Build mit Feature `development`, AMD Ryzen 9 7900X,
WSL2/Linux 6.6.87.2, logische CPU 2. Die Baseline-Binärdatei wurde vor der
Produktionsänderung gesichert. Erst nach Abschluss ihres vollständigen Laufs
wurde die optimierte Variante gebaut und ausgeführt; Tests und Compiler liefen
nicht gleichzeitig mit den Benchmarks. Innerhalb jeder Variante wechseln
Kontrolle und Entwicklung ihre Ausführungsreihenfolge zwischen Messrunden.

Die Messgrenzen sind unverändert: Netzaufbau, Eingabegenerierung,
Referenzvergleiche und optionale Diagnoseexporte liegen außerhalb der
Gesamtzeit. Öffentliche Eingabeplanung, Runtime, Planung, Pruning-Beobachtung
und atomarer Entwicklungs-Commit liegen darin.

### Gesamtlaufzeit mit Strukturentwicklung

Median aus sieben Messungen, in Millisekunden:

| Neuronen | Seed | Vorher | Nachher | Laufzeitreduktion |
|---|---:|---:|---:|---:|
| 2.048 | 7 | 169,339 | 138,486 | 18,2 % |
| 2.048 | 42 | 170,102 | 138,612 | 18,5 % |
| 2.048 | 2026 | 164,855 | 135,534 | 17,8 % |
| 8.192 | 7 | 1.203,977 | 780,739 | 35,2 % |
| 8.192 | 42 | 1.199,598 | 801,399 | 33,2 % |
| 8.192 | 2026 | 1.192,441 | 792,224 | 33,6 % |

### Vollständige Pruning-Beobachtung

Auch diese Werte sind **Mediane aus den sieben Messläufen**, keine einzelnen
Phasenbeobachtungen. Das bestehende Messprogramm erfasst die Host-Abschnitte
bereits separat mit Uhrabfragen pro Kontrollpunkt, auch ohne Runtime-Profiling.
Gemessen wird das gesamte `observe`, einschließlich der unveränderten
Utility-/Timer-Verarbeitung, nicht ausschließlich die ersetzte innere Schleife.

| Neuronen | Vorher, je nach Seed (ms) | Nachher, je nach Seed (ms) | Reduktion |
|---|---:|---:|---:|
| 2.048 | 32,441–32,980 | 3,851–4,042 | 87,7–88,1 % |
| 8.192 | 443,891–447,892 | 19,023–20,741 | 95,3–95,8 % |

Die Vergleichsdatei enthält alle zwölf Fälle einschließlich der Kontrollen,
Minima und Maxima. In der Kontrolle wird `observe` nicht aufgerufen; ihre
Mediane liegen zwischen 0,4 % unter und 7,2 % über der jeweiligen Baseline.
Diese Abweichungen ohne Pruning-Behandlung zeigen die Grenzen des lokalen
Versuchs. Die Versionen wurden nacheinander, nicht abwechselnd gemessen;
zeitliche Drift wurde nicht getrennt untersucht. Die angegebenen Reduktionen
sind lokale Beobachtungen, keine allgemeine Performancegarantie.

## Direkte Korrektheitsprüfung zwischen den Binärdateien

Das Beispiel kann mit `NERVA_REPLAY_DIR` die deterministischen Ergebnisse
seiner Aufwärmläufe außerhalb der Zeitmessung vollständig exportieren:
Zählungen, Kontrollpunkte einschließlich neu gebildeter und entfernter
Verbindungen, den kompletten Graphzustand und das geordnete Ereignislog.
Die Option verändert weder Last noch Messgrenzen und ist standardmäßig aus.

Alle **zwölf Dateipaare** vor/nach der Änderung sind direkt byteweise identisch,
zusammen 843.800.189 Bytes je Version. Es werden keine Hashwerte verglichen.
Die Debug-Darstellung gilt nur für dieselbe Toolchain und ist kein
Persistenzformat; die großen Diagnosedateien liegen nur unter `/tmp`.

Innerhalb jeder Binärdatei vergleichen zudem alle sieben Messläufe und der
instrumentierte Lauf vollständige Netzwerkzustände, Ereignislogs und
Entwicklungsergebnisse direkt mit dem Aufwärmlauf. Zwischen den CSV-Dateien
stimmen alle nicht zeitlichen Zählungen, Kontrollpunkte, Histogramme und
Phasenaufrufzahlen überein. Der neue Timer-Test prüft insbesondere, dass die
Bereinigung den Beginn einer laufenden Mindestwartezeit nicht verschiebt.

**324 Tests bestanden**, drei GPU-Hardwaretests erwartungsgemäß ignoriert;
Clippy mit allen Features/Targets und `-D warnings` ohne Warnungen.
Die beiden bearbeiteten Rust-Dateien bestehen `rustfmt --check`.

## Reproduktion

Aktuelle Variante:

```bash
cargo build --release --features development --example self_organisation_profile
NERVA_REPLAY_DIR=/tmp/nerva-pruning-after-replays \
  taskset -c 2 target/release/examples/self_organisation_profile > /tmp/pruning-after.csv
cargo test --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
rustfmt --edition 2024 --check src/development/pruning.rs examples/self_organisation_profile.rs
```

Für einen erneuten A/B-Vergleich zuerst die alte `Vec`-/`contains`-Variante
mit demselben Messprogramm und derselben Toolchain bauen und als separate
Binärdatei sichern. Die Baseline dieser Sitzung liegt unter
`/tmp/nerva-pruning-before-bin`; sie stammt vom angetroffenen Workspace vor
der Retain-Änderung, nicht vom älteren Git-HEAD. Beide Versionen müssen
verschiedene Replay-Verzeichnisse verwenden. Vergleich der Diagnoseexporte:

```bash
for replay in /tmp/nerva-pruning-before-replays/*.txt; do
  cmp "$replay" "/tmp/nerva-pruning-after-replays/${replay##*/}" || exit 1
done
```

Alle Cargo-Aufrufe dieser Messung verwendeten zusätzlich `--offline`.
`taskset` setzt eine verfügbare logische CPU 2 voraus.

Rohdaten: [vorher](benchmarks/pruning_profile_2026-10-14_before.csv),
[nachher](benchmarks/pruning_profile_2026-10-14_after.csv),
[Median-/Streuungsvergleich](benchmarks/pruning_profile_2026-10-14_comparison.csv).
Die Vergleichsspalten basieren auf den sieben `sample`-Zeilen pro Fall für
`Total` bzw. `PruningObserve`.

Der bisherige Pruning-Engpass ist damit gezielt reduziert. Die vorhandenen
Kontrollen und der unveränderte Ablauf bilden nun eine bessere Grundlage
für einen anschließenden Versuch zum Nutzen gelernter Strukturen in einer
Umweltaufgabe; diese Optimierung selbst liefert dafür keinen neuen Nachweis.
