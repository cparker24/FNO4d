# loc_libs — local helper library for FNO4d data I/O, visualisation, and training.
# Install with `pip install -e ./loc_libs` to make all modules importable without
# sys.path manipulation.
#
# Modules
# -------
# read_3d_root      – read 3-D ROOT files into numpy arrays (read_3d_data, coarsen_array)
# read_3d_hdf5      – read HDF5 variants of the same format
# datasets          – PyTorch Dataset classes (FluidDataset)
# memory_utils      – process memory helpers (MemoryPrinter, calculate_model_memory)
# training_utils    – training helpers (enable_log_tee, patch_trainer_checkpoint)
# parse_training_log – parse and plot neuralop training logs
# NormFn / NormFn3d  – 3-D+1 normalisation functions (see README_NormFn3d.md)
