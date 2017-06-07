
# `SFDB_to_mat` usage

in the Matlab prompt:

convert LIGO Hanford data files
```
>> SFDB_to_mat('/storage/pss/ligo_h/sfdb/O2/128/')
```

convert LIGO Livingston data files
```
>> SFDB_to_mat('/storage/pss/ligo_l/sfdb/O2/128/')
```

# data informations

* detectors: LIGO Hanford, LIGO Livingston
* observing run: O2 (most recent data)
* calibration: C00 (first calibration, then we will use C01)
* SFDB band: "256" (from 10 Hz to 128 Hz)


