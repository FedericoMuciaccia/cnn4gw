
## `convert_SFDB09_to_mat` usage

in the Matlab prompt:

convert LIGO Hanford data files
```
>> convert_SFDB09_to_mat('/storage/pss/ligo_h/sfdb/O2/128/')
```

convert LIGO Livingston data files
```
>> convert_SFDB09_to_mat('/storage/pss/ligo_l/sfdb/O2/128/')
```

or
```
>> convert_SFDB09_to_mat('/storage')
```


## data informations

* detectors: LIGO Hanford, LIGO Livingston
* observing run: O2 (most recent data)
* calibration: C00 (first calibration, then we will use C01)
* SFDB band: "256" (from 10 Hz to 128 Hz)


