# dsvlm-rust

Rust-Migration fuer DSVLM.

Der bestehende C++-Code im Projekt-Root bleibt vorerst der eingefrorene Referenzstand. Neue Portierungen laufen in diesem Ordner inkrementell nach Rust.

Aktuell portiert:

- `world/bit_world_vm`
- `body/peripheral_bridge`
- `sensory/sensor_transduction`
- ein schlanker deterministischer `runtime`-Slice fuer Phase 2

Tests laufen mit:

```bash
cargo test
```
