# TODO

## Filter units by automatic curation

Add a filter for the results of automatic curation tools to **Load Spikes → Filter neurons**, next to the curation label, the Kilosort label and the User filter.

What the tools write (to check against real output folders):

| Tool | File | Label |
|---|---|---|
| Bombcell (Python) | `cluster_bc_unitType.tsv` in the Kilosort folder | column `bc_unitType`: `GOOD`, `MUA`, `NOISE`, `NON-SOMA` |
| Bombcell (MATLAB) | `templates._bc_qMetrics.parquet` in a separate save folder | unit type, not confirmed |
| IBL | cluster metrics table in the ALF folder (file name not confirmed) | column `label`: 0, 1/3, 2/3, 1 (1 = good) |
| SpikeInterface `export_to_phy` | one `cluster_<metric>.tsv` per metric in the phy folder | metric values |

- [ ] Get example output folders from Bombcell and from the IBL pipeline.
- [ ] The Bombcell TSV already works with the **User** filter (column `bc_unitType`). Check that, and check if the value match ignores case.
- [ ] Add an "Automatic curation" filter that finds the sources in the folder and offers their labels.
- [ ] Decide how to read parquet files (new dependency), or read the CSV variants that the tools also write.

## Compressed formats (Zarr, NWB)

- [ ] Export preprocessed data from a Zarr or NWB recording. It is disabled now, because an export copies the SpikeGLX or Open Ephys metadata files of the original.
- [ ] NWB: let the user choose the electrical series when the file holds several. Now NPXplorer shows the one with the highest sample rate.
- [ ] NWB files with a Zarr backend (hdmf-zarr).
- [ ] Zarr v3.
- [ ] Zarr: segments after the first.
- [ ] Lossy WavPack: the `oxideav-wavpack` decoder gives values up to 7 ADC counts away from the reference decoder (libwavpack) on a few channel pairs. Lossless decoding is exact. Report it upstream, or replace the decoder.
- [ ] Test Zarr and NWB files of AP-band data, of SpikeGLX recordings and of multi-shank probes. Only one Open Ephys NP 1.0 LFP recording is tested.
