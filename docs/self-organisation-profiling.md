# Asynchrone Netze mit Strukturentwicklung – 14. Oktober 2026

**Nachtrag:** Die hier dokumentierte Ausgangsmessung führte inzwischen zur
[umgesetzten Pruning-Optimierung mit neuem Vorher-/Nachher-Vergleich](pruning-optimization.md).
Die ursprünglichen Messdaten und Tabellen bleiben als Ausgangsbefund erhalten.

## Ergebnis

Der nächste offene Schritt nach dem isolierten Scheduler-Vergleich ist
umgesetzt: größere räumliche Netze laufen mit asynchronen Eingaben,
Pair-STDP, Utility, Verbindungsbildung und Pruning. Drei Seeds und eine
Kontrolle mit unveränderter Topologie machen die Last reproduzierbar.

Die vollständigen Scheduler-Batches bleiben klein. In der Entwicklungsvariante
enthalten bei 2.048 Neuronen **85,4–85,9 %** genau ein Ereignis, bei 8.192
Neuronen **60,6–60,9 %**. Die größten Batches enthalten sieben bzw. elf Ereignisse;
auch die Neuronenintegration erreicht höchstens elf Zellen pro Aufruf.
Die zuvor getestete Bucket-Queue wird weiterhin nicht übernommen. Die Zahl
der Neuronen allein erzeugt hier keine großen Compute-Batches für CUDA.

Der deutlichste neue CPU-Kostenblock ist die **Pruning-Beobachtung**. Sie
benötigt bei 8.192 Neuronen in den instrumentierten Läufen 438–455 ms.
Verbindungsplanung und atomarer Commit kommen hinzu. Die vorhandenen lokalen
Regeln werden erfolgreich in größerem Maßstab ausgeführt; die Messung bewertet
noch keine gelernte Anwendungsaufgabe.

## Lastmodell

- 2.048 oder 8.192 Zellen auf einem zweidimensionalen Raster mit Abstand 0,05;
  etwa 20 % inhibitorisch, stabile dünn verteilte IDs, anfangs acht gerichtete
  Verbindungen pro Zelle (16.384 bzw. 65.536 Synapsen). Die Raster-Nachbaroffsets
  werden modular fortgesetzt, einschließlich Verbindungen an Rasterrändern.
- Anfangsgewicht 0,06, seeded Verzögerungen von 10–200 µs. Pair-STDP ist in
  **beiden** Varianten aktiv, mit `a_plus = 0,001`, `a_minus = 0,0012` und
  Gewichtsgrenzen 0,01–0,1. ExcitatoryCausalUtility beurteilt exzitatorische
  Quellen; inhibitorische Quellen bleiben von Utility-Pruning ausgenommen.
- 400 ms Simulationszeit mit unabhängig erzeugten exponentiellen
  Hintergrund-Wartezeiten (12 Hz je Zelle, auf ganze Mikrosekunden aufgerundet).
  Deren Amplitude 0,7 liegt einzeln unter der Schwelle 1,0.
- Lokale Achtergruppen besitzen voneinander unabhängig gezogene Phasen. Vier
  Zellen pro Gruppe erhalten acht Pakete im Abstand von 50 ms, Amplitude 1,05,
  mit 75 µs Abstand zwischen Gruppenmitgliedern und zusätzlichem Jitter von
  0–19 µs. Damit gibt es lokale Korrelation, ohne populationsweiten Eingabetakt.
- Utility-Eligibility-Zeitkonstante 2 ms, Utility-Gedächtnis 200 ms,
  Lernfaktor 0,2. Pruning-Schwelle 0,03, Mindestdauer darunter 80 ms.

Alle 100 ms pausiert der Host nach Verarbeitung der bereits fälligen Ereignisse
und führt `PruningController::observe`, `DevelopmentPlan::build`,
`populate_plan` und den atomaren Commit aus. Der nächste Eingabeabschnitt wird
erst danach eingereiht. Diese Host-Termine sind Experimentkontrollpunkte,
keine Neuronen-Zeitschritte. Intervalle sind halboffen: Eingaben genau an einer
Grenze werden vor der Verarbeitung dieses Zeitstempels gemeinsam mit bereits
unterwegs befindlichen Ankünften eingereiht. Am Ende wird ein auf 10 ms
begrenzter Ereignisnachlauf verarbeitet; alle gemessenen Queues sind danach leer.

Jede sechzehnte Zelle erhält anfangs eine lokale Wachstumsanforderung von 2,5.
Eine Bildung verbraucht 1,0 davon, maximal zwei Bildungen pro ausgewählter
Zelle. Die Kandidatenwahl verwendet räumliche Distanz, Aktivität und zeitliche
Korrelation mit Radius 0,16. Verbindungen entstehen und verschwinden nur über
die vorhandenen Entwicklungsregeln; konkrete neue Endpunkte werden nicht vom
Versuchsprogramm vorgegeben. Die Wachstumsanforderung wird ausdrücklich
**initialisiert**: laufende Homöostase, Zellvermehrung und intrinsisch autonome
Aktivität sind in diesem Versuch nicht aktiviert. Es handelt sich weiterhin
um ein kontrolliertes Aktivitätsmodell.

## Wiederholte Laufzeiten

Release-Build, Rust 1.98.1, AMD Ryzen 9 7900X unter
WSL2/Linux 6.6.87.2, logische CPU 2. Je Fall ein Aufwärmlauf und sieben
uninstrumentierte Messungen, danach ein instrumentierter Lauf. Die beiden
Varianten werden pro Messrunde abwechselnd zuerst ausgeführt. Insgesamt:
zwölf Fälle, 84 Messläufe und zwölf instrumentierte Läufe.

| Neuronen | Seed | Feste Topologie, Median (ms) | Entwicklung, Median (ms) |
|---|---:|---:|---:|
| 2.048 | 7 | 136,722 | 170,861 |
| 2.048 | 42 | 131,579 | 168,036 |
| 2.048 | 2026 | 133,099 | 167,854 |
| 8.192 | 7 | 656,610 | 1.195,247 |
| 8.192 | 42 | 633,423 | 1.180,751 |
| 8.192 | 2026 | 651,991 | 1.186,817 |

Pro Seed teilen beide Varianten Anfangsnetz und Eingabefolge. Da Entwicklung
Topologie, Ankunftszahlen und anschließend neuronale Trajektorien verändert,
ist die Differenz **kein isolierter Verwaltungs-Overhead** und kein
Vorher-nachher-Geschwindigkeitsvergleich. Minima, Maxima und Einzelwerte stehen
in den CSV-Dateien; diese lokalen Mediane sind keine allgemeine Laufzeitgarantie.

Netzaufbau und Erzeugung der Eingabefolgen liegen außerhalb der Gesamtzeit.
Die gemessene Gesamtzeit umfasst die öffentliche Eingabeplanung, Runtime,
Entwicklungsplanung und Commit einschließlich Transaktionskopien, Stornierung
unterwegs befindlicher Ankünfte und Bereinigung der Pruning-Timer. Analytische
Endauswertung, CSV-Ausgabe und Vergleiche liegen außerhalb.

## Strukturänderungen

Über alle drei Seeds entstehen je Lauf 256 bzw. 1.024 neue Synapsen. Bei
2.048 Neuronen werden 8.487–8.520 Verbindungen entfernt; bei 8.192 Neuronen
38.878–38.916. Am Ende existieren noch 250–255 bzw. 1.013–1.021 der neu
gebildeten Synapsen. Davon liegen 105–106 bzw. 425–433 aktuell über der
Utility-Schwelle. Der Verbleib einer Verbindung allein beweist keine positive
Utility, da die Mindestdauer unter der Schwelle eine Entfernung verzögert.
`useful_new` in den Rohdaten bezeichnet ausschließlich diesen momentanen
Schwellenvergleich; in der Zusammenfassung heißt die Spalte präziser
`new_above_utility_threshold`.

Beispiel 8.192 Neuronen, Seed 42: 38.540 Spikes, 318.392 verarbeitete
Ereignisse, 203.373 Batches und 216.961 protokollierte Gewichtsänderungen.
Die anfänglichen 65.536 Synapsen werden durch 1.024 Bildungen und 38.885
Entfernungen zu 27.675 Synapsen. Diese Zahlen belegen die Ausführung des
Lern-/Entwicklungszyklus, keine Verbesserung einer Aufgabenleistung.

## Wo die Zeit verbleibt

Die folgende Tabelle verwendet ausschließlich die zusätzlichen instrumentierten
Läufe mit Seed 42. Phasenwerte sind Einzelbeobachtungen, keine Mediane.

| Disjunkte Host-Abschnitte | 2.048 Neuronen (ms) | 8.192 Neuronen (ms) |
|---|---:|---:|
| Eingabeplanung | 1,938 | 9,624 |
| Runtime einschließlich Scheduler-Entnahme | 158,156 | 638,174 |
| Pruning-Beobachtung | 33,996 | 437,548 |
| Verbindungsplanung | 4,567 | 73,916 |
| Pruning-Kandidaten | 0,866 | 5,075 |
| Atomarer Commit und Timer-Bereinigung | 11,980 | 133,769 |
| Gesamt einschließlich gemeinsamer Buchhaltung | 211,529 | 1.298,201 |

Innerhalb der großen Runtime entfallen 101,177 ms auf Ankünfte einschließlich
47,331 ms Prä-STDP, 99,185 ms auf Post-STDP, 45,998 ms auf Scheduler-Entnahme
und 66,645 ms auf Propagation. Letztere enthält 35,303 ms Planung und
21,396 ms Einreihen. Advance-/Integrate-Compute zusammen benötigen 19,699 ms.
Prä-STDP darf nicht erneut zu Ankünften addiert werden; Propagationsunterphasen
dürfen nicht erneut zu Propagation addiert werden. `SchedulerDequeue` liegt
außerhalb der bisherigen `Batch`-Messung, innerhalb des Host-Abschnitts Runtime.

Bei vierfacher Zellzahl wächst die Pruning-Beobachtungszeit hier etwa um
Faktor 12,9. In der damals gemessenen Ausgangsversion prüfte
`PruningController::observe` für jeden gespeicherten Timer die Zugehörigkeit
zu einem `Vec` lebender Synapsen über `contains`.
Dieser Teil hat bei vielen Timern und Synapsen quadratischen Aufwand. Auch die
Kandidatensuche läuft für jeden Wachstumskandidaten über die gesamte Population.
Diese Befunde motivierten die inzwischen umgesetzte Optimierung der
Pruning-Buchhaltung. Dieser ursprüngliche Bericht isoliert die innere
`contains`-Schleife noch nicht zeitlich; der verlinkte Nachtrag vergleicht die
vollständige Pruning-Beobachtung vor und nach dem Wechsel zum Synapsenindex.

## Profiler und Prüfung

Der opt-in Profiler erfasst jetzt `scheduler_batch_size` vor der Runtime-
Validierung sowie `Histogram::singletons` für die exakte Häufigkeit von
Einzelereignissen. `SchedulerDequeue` misst die Entnahme einschließlich
fehlgeschlagener oder leerer Versuche. Fehlgeschlagene Entnahmen erzeugen keine
Histogrammprobe und verbrauchen kein Batch. Alle neuen Zähler werden mit
konstantem Speicherbedarf zusammengeführt; deaktiviertes Profiling liest
weiterhin keine Uhr.

Jeder Messlauf und jeder instrumentierte Lauf wird außerhalb der Zeitmessung
direkt mit seinem Aufwärmlauf verglichen: vollständiger Netzwerkzustand,
geordnetes Ereignislog, Entwicklungspläne/Ergebnisse, Kontrollpunktzustände,
aktuelle Zeit und Zählungen. Es wird kein Zustandsfingerabdruck verwendet.
Zusätzlich gelten Topologiebilanz und Übereinstimmung von Scheduler-Histogramm
mit verarbeiteten Ereignissen/Batches. Die instrumentierten Werte sind durch
ihre Uhrabfragen beeinflusst und werden nicht als uninstrumentierte Laufzeiten
interpretiert.

**323 Tests bestanden**, drei GPU-Hardwaretests ignoriert. Neue Tests prüfen
das Entwicklungsmodell samt exakter Profiling-Regression, Seed-/Zeitfenster-
Eigenschaften, gleichzeitige Eingaben und Ankünfte an Host-Grenzen sowie den
Scheduler-Limit-Fehlerpfad. Clippy mit allen Features/Targets und `-D warnings`
ist ohne Warnungen; die bearbeiteten Rust-Dateien bestehen `rustfmt --check`.

```bash
cargo build --release --features development --example self_organisation_profile
taskset -c 2 target/release/examples/self_organisation_profile > /tmp/nerva-self-organisation.csv
cargo test --all-features --all-targets
cargo clippy --all-features --all-targets -- -D warnings
```

Die Messungen und Prüfungen verwendeten zusätzlich `--offline`. `taskset`
erfordert eine verfügbare logische CPU 2; ohne Bindung kann das Beispiel direkt
über `cargo run --release --features development --example self_organisation_profile`
ausgeführt werden. Die Größen, Seeds und sieben Wiederholungen sind im Beispiel
festgelegt. Fortschritt geht nach stderr, das einheitliche CSV nach stdout.

Quellcode: [Versuch](../examples/self_organisation_profile.rs),
[Profiler](../src/runtime/profile.rs), [Profiling-Tests](../tests/runtime_profile.rs).
Rohdaten: [Einzelwerte, Phasen, Histogramme und Kontrollpunkte](benchmarks/self_organisation_profile_2026-10-14.csv),
[Zusammenfassung aller zwölf Fälle](benchmarks/self_organisation_profile_2026-10-14_summary.csv).
Die Zusammenfassung übernimmt `count`-/`histogram`- und `host-phase`-Werte;
Laufzeitmedian und Streuung werden aus den sieben `sample`-Zeilen mit Metrik
`Total` pro Größe, Seed und Modus berechnet.
