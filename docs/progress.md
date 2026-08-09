# DSVLM – Gedankliche Entwicklung des Projekts

## Zweck dieses Dokuments

Dieses Dokument hält nicht nur den aktuellen technischen Stand von DSVLM fest, sondern den Weg dorthin:

```text
Grobe Idee
→ technische Übersetzung
→ Problem oder Widerspruch
→ nächste Idee
→ Erkenntnis
```

Die Entwicklung ist chronologisch rekonstruiert. Frühere Ideen sind deshalb nicht automatisch weiterhin gültige Bestandteile der aktuellen Architektur. Einige wurden erweitert, andere korrigiert oder bewusst verworfen.

---

## 1. Die Grundidee: ein dynamischer Vektorraum

### Grobe Idee

Die erste Grundidee war ein **dynamisch selbstorganisierendes Vektorraum-Lernmodell**. Das System sollte nicht mit einem fertig trainierten Netzwerk starten, sondern während des Betriebs eigene Strukturen entwickeln.

Wissen sollte nicht aus festgelegten Begriffen bestehen, sondern aus stabilen, wieder aktivierbaren Zuständen.

### Technische Übersetzung

Ein Neuron sollte unter anderem besitzen:

- eine ID,
- eine Position in einem Vektorraum,
- eine Aktivierung,
- eine vorherige Aktivierung,
- einen Stabilitätswert,
- gerichtete Verbindungen zu anderen Neuronen.

Verbindungen sollten als sparsamer Graph gespeichert werden. Nähe, Aktivierung, Einfluss, Folge und Stabilität sollten bestimmen, wie sich der Graph entwickelt.

### Problem

Ein Vektorraum allein erklärt noch nicht, wie aus Eingaben Lernen entsteht. Eine Position im Raum sagt nicht automatisch:

- welches Neuron ein anderes beeinflusst,
- welche Reihenfolge wichtig ist,
- warum eine Verbindung stärker wird,
- wie Handlungen entstehen.

Außerdem war zunächst unklar, was ein Vektor überhaupt repräsentiert, wenn noch keine Sensorik und keine Umwelt existieren.

### Nächste Idee
