# Historical Benchmark Results

Automated benchmark: simulated agent vs. historical AI across all engagements in `data/historical/battles.json`.

## Summary

| Metric | Value |
|--------|-------|
| Total scenarios | 52 |
| Passed (no error) | 52 |
| Failed | 0 |
| Winner match rate | 7.7% ❌ |
| Mean fidelity score | 0.425 |
| Std fidelity score | 0.151 |
| Importer criterion (≥ 50 without errors) | ✅ PASS |
| Outcome criterion (≥ 60 % winner match) | ❌ FAIL |
| Total elapsed | 0.0 min ✅ |

## Per-Scenario Results

| # | Battle | Date | Source | Hist. Winner | Sim. Winner | Winner Match | Blue Δcas | Red Δcas | Fidelity | Status |
|---|--------|------|--------|-------------|-------------|-------------|----------|----------|----------|--------|
| 1 | Battle of Waterloo (1815) | 1815-06-18 | Napoleon's Battles | 0 | draw | ❌ | -0.11 | -0.63 | 0.351 | ✅ |
| 2 | Battle of Austerlitz (1805) | 1805-12-02 | Napoleon's Battles | 0 | draw | ❌ | -0.12 | -0.32 | 0.371 | ✅ |
| 3 | Battle of Borodino (1812) | 1812-09-07 | Nafziger OOBs | draw | 1 | ❌ | +0.36 | -0.40 | 0.205 | ✅ |
| 4 | Battle of Jena (1806) | 1806-10-14 | Napoleon's Battles | 0 | draw | ❌ | -0.05 | -0.50 | 0.284 | ✅ |
| 5 | Battle of Eylau (1807) | 1807-02-07 | Napoleon's Battles | draw | draw | ✅ | -0.15 | -0.28 | 0.927 | ✅ |
| 6 | Battle of Friedland (1807) | 1807-06-14 | Napoleon's Battles | 0 | draw | ❌ | -0.08 | -0.35 | 0.358 | ✅ |
| 7 | Battle of Wagram (1809) | 1809-07-05 | Napoleon's Battles | 0 | draw | ❌ | -0.23 | -0.32 | 0.384 | ✅ |
| 8 | Battle of Aspern-Essling (1809) | 1809-05-21 | Napoleon's Battles | 1 | draw | ❌ | -0.07 | -0.15 | 0.459 | ✅ |
| 9 | Battle of Salamanca (1812) | 1812-07-22 | Napoleon's Battles | 0 | draw | ❌ | -0.13 | -0.23 | 0.246 | ✅ |
| 10 | Battle of Vitoria (1813) | 1813-06-21 | Napoleon's Battles | 0 | draw | ❌ | -0.02 | -0.33 | 0.439 | ✅ |
| 11 | Battle of Dresden (1813) | 1813-08-26 | Napoleon's Battles | 0 | 0 | ✅ | -0.10 | +0.27 | 0.764 | ✅ |
| 12 | Battle of Leipzig (1813) | 1813-10-16 | Napoleon's Battles | 1 | draw | ❌ | -0.45 | -0.25 | 0.334 | ✅ |
| 13 | Battle of Ligny (1815) | 1815-06-16 | Napoleon's Battles | 0 | draw | ❌ | -0.12 | -0.20 | 0.374 | ✅ |
| 14 | Battle of Quatre Bras (1815) | 1815-06-16 | Napoleon's Battles | draw | draw | ✅ | +0.03 | -0.13 | 0.938 | ✅ |
| 15 | Battle of Marengo (1800) | 1800-06-14 | Napoleon's Battles | 0 | draw | ❌ | -0.24 | -0.20 | 0.396 | ✅ |
| 16 | Battle of Arcola (1796) | 1796-11-15 | Corsican Ogre | 0 | draw | ❌ | -0.10 | -0.25 | 0.439 | ✅ |
| 17 | Battle of Rivoli (1797) | 1797-01-14 | Corsican Ogre | 0 | draw | ❌ | -0.07 | -0.35 | 0.359 | ✅ |
| 18 | Battle of Smolensk (1812) | 1812-08-17 | Nafziger OOBs | 0 | draw | ❌ | -0.07 | -0.08 | 0.439 | ✅ |
| 19 | Battle of Lutzen (1813) | 1813-05-02 | Napoleon's Battles | 0 | draw | ❌ | +0.02 | -0.10 | 0.474 | ✅ |
| 20 | Battle of Bautzen (1813) | 1813-05-20 | Napoleon's Battles | 0 | draw | ❌ | -0.10 | -0.12 | 0.459 | ✅ |
| 21 | Battle of Eckmuhl (1809) | 1809-04-22 | Napoleon's Battles | 0 | draw | ❌ | -0.03 | -0.20 | 0.388 | ✅ |
| 22 | Battle of Talavera (1809) | 1809-07-27 | Napoleon's Battles | 0 | draw | ❌ | -0.15 | -0.15 | 0.447 | ✅ |
| 23 | Battle of Albuera (1811) | 1811-05-16 | Napoleon's Battles | 0 | draw | ❌ | -0.17 | -0.30 | 0.352 | ✅ |
| 24 | Battle of the Berezina (1812) | 1812-11-26 | Nafziger OOBs | 0 | draw | ❌ | +0.00 | -0.00 | 0.492 | ✅ |
| 25 | Battle of Hanau (1813) | 1813-10-30 | Napoleon's Battles | 0 | draw | ❌ | +0.15 | -0.20 | 0.314 | ✅ |
| 26 | Battle of Montmirail (1814) | 1814-02-11 | Napoleon's Battles | 0 | draw | ❌ | +0.03 | -0.15 | 0.395 | ✅ |
| 27 | Battle of Vauchamps (1814) | 1814-02-14 | Napoleon's Battles | 0 | draw | ❌ | +0.05 | -0.15 | 0.337 | ✅ |
| 28 | Battle of La Rothiere (1814) | 1814-02-01 | Napoleon's Battles | 1 | draw | ❌ | +0.05 | -0.07 | 0.444 | ✅ |
| 29 | Battle of Castiglione (1796) | 1796-08-05 | Corsican Ogre | 0 | draw | ❌ | -0.00 | -0.18 | 0.395 | ✅ |
| 30 | Battle of Bassano (1796) | 1796-09-08 | Corsican Ogre | 0 | draw | ❌ | +0.10 | -0.22 | 0.319 | ✅ |
| 31 | Battle of Novi (1799) | 1799-08-15 | Corsican Ogre | 1 | draw | ❌ | -0.25 | -0.15 | 0.432 | ✅ |
| 32 | Battle of Trebbia (1799) | 1799-06-17 | Corsican Ogre | 1 | draw | ❌ | -0.25 | -0.18 | 0.421 | ✅ |
| 33 | Battle of Ulm (1805) | 1805-10-17 | Napoleon's Battles | 0 | draw | ❌ | -0.02 | -0.70 | 0.384 | ✅ |
| 34 | Battle of Auerstedt (1806) | 1806-10-14 | Napoleon's Battles | 0 | draw | ❌ | -0.15 | -0.50 | 0.394 | ✅ |
| 35 | Battle of Heilsberg (1807) | 1807-06-10 | Napoleon's Battles | 1 | draw | ❌ | -0.12 | -0.05 | 0.397 | ✅ |
| 36 | Battle of Hohenlinden (1800) | 1800-12-03 | Napoleon's Battles | 0 | draw | ❌ | +0.05 | -0.30 | 0.370 | ✅ |
| 37 | Battle of Maloyaroslavets (1812) | 1812-10-24 | Nafziger OOBs | 0 | draw | ❌ | +0.02 | -0.07 | 0.448 | ✅ |
| 38 | Battle of Wavre (1815) | 1815-06-18 | Napoleon's Battles | 0 | draw | ❌ | +0.05 | +0.07 | 0.444 | ✅ |
| 39 | Battle of Fuentes de Onoro (1811) | 1811-05-03 | Napoleon's Battles | 0 | draw | ❌ | -0.02 | -0.04 | 0.483 | ✅ |
| 40 | Battle of Arcis-sur-Aube (1814) | 1814-03-20 | Napoleon's Battles | 1 | draw | ❌ | +0.05 | -0.07 | 0.404 | ✅ |
| 41 | Battle of Tolentino (1815) | 1815-05-02 | Corsican Ogre | 1 | draw | ❌ | -0.10 | +0.05 | 0.400 | ✅ |
| 42 | Battle of Ratisbon (1809) | 1809-04-23 | Napoleon's Battles | 0 | draw | ❌ | +0.04 | -0.05 | 0.353 | ✅ |
| 43 | Battle of Lonato (1796) | 1796-08-03 | Corsican Ogre | 0 | draw | ❌ | +0.04 | -0.10 | 0.346 | ✅ |
| 44 | Battle of Millesimo (1796) | 1796-04-13 | Corsican Ogre | 0 | draw | ❌ | +0.02 | -0.10 | 0.404 | ✅ |
| 45 | Battle of Lodi (1796) | 1796-05-10 | Corsican Ogre | 0 | draw | ❌ | +0.03 | -0.03 | 0.291 | ✅ |
| 46 | Battle of Paris (1814) | 1814-03-30 | Napoleon's Battles | 1 | draw | ❌ | +0.05 | -0.05 | 0.447 | ✅ |
| 47 | Battle of Kulm (1813) | 1813-08-29 | Napoleon's Battles | 0 | draw | ❌ | -0.02 | -0.65 | 0.361 | ✅ |
| 48 | Battle of Dennewitz (1813) | 1813-09-06 | Napoleon's Battles | 0 | draw | ❌ | -0.00 | -0.15 | 0.439 | ✅ |
| 49 | Battle of Brienne (1814) | 1814-01-29 | Napoleon's Battles | draw | draw | ✅ | +0.02 | -0.00 | 0.959 | ✅ |
| 50 | Battle of Champaubert (1814) | 1814-02-10 | Napoleon's Battles | 0 | draw | ❌ | +0.06 | -0.35 | 0.238 | ✅ |
| 51 | Battle of Bar-sur-Aube (1814) | 1814-02-27 | Napoleon's Battles | 1 | draw | ❌ | +0.03 | -0.00 | 0.457 | ✅ |
| 52 | Battle of Saint-Dizier (1814) | 1814-03-26 | Napoleon's Battles | 0 | draw | ❌ | +0.15 | +0.00 | 0.344 | ✅ |

> *Generated automatically by `training/historical_benchmark.py`.*
