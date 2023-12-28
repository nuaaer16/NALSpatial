# NALSpatial

An effective natural language transformation framework for queries overleaf spatial data.

## Dependencies

- torch-2.0.1
- Python-3.8

## Usage

1. Train the model to identify the type of NLQ.

```
python LSTM/train.py
```

2. Put the obtained models and related information in the directory `SpatailNLQ/save_models`.

3. Integrate `SpatailNLQ` as an algebra into SECONDO database. (SECONDO: [https://secondo-database.github.io/](https://secondo-database.github.io/))

4. Enter the command in SECONDO:

```
query spatial_nl("List the 12 parks closest to the Nanjing border.");
```

5. The corresponding executable language is returned:

```
query park creatertree[GeoData] park distancescan [NJBorderLine, 12] consume;
```

