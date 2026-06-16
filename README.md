Derivative work of https://github.com/HuguesTHOMAS/KPConv-PyTorch, aimed first at evaluating US3D dataset, and now expanding to adapt to general purpose LAS/LAZ datasets.

We'll add the requisite LICENSE file in subsequent commits, but as with the source repository (https://github.com/HuguesTHOMAS/KPConv-PyTorch/blame/master/README.md#L55) this work will be released under the MIT license.

To create a Conda environment to run `kpconv-pdal`, run the following command (assuming you have Conda installed of course) from within the root directory of the repository.

```bash
conda env create -f environment.yml
conda activate kpconv-pdal-env
```

We currently still use some of the C++ extensions provided in the original KPConv-Pytorch release. To compile these, from within the `cpp_wrappers` directory, issue the following command.

```bash
sh compile_wrappers.sh
```

At this point, you should be able to train models and predict classification labels. After modifying the `LASConfig` class within `train_LAS.py`, namely to adjust paths to the dataset and the tensorboard run directory, simply execute the following.

```bash
python train_LAS.py
```

The current means of predicting labels is to modify `test_LAS.py` to point to the results log directory, and to uncomment line 24 of `datasets/LAS.py`. You can then run the following.

```bash
python test_LAS.py
```

## Future revisit notes

This branch includes several changes to get the project running again, at least for inference with previously trained LAS weights.

The environment has been moved toward `pixi`. The `pixi.toml` and lockfile capture updated dependencies, including `python-pdal`, PyTorch, NumPy, scikit-learn, and the other runtime packages needed by this codebase. `pixi.toml` also includes tasks for rebuilding the original KPConv C++ wrappers:

```bash
pixi run build-cpp-wrappers
```

The branch also adds `apply_classification.py`. This standalone script takes a sampled input point cloud and an ASCII text file containing one predicted label per point, in matching point order. It writes a new LAZ file with the same point data and an updated `Classification` dimension:

```bash
pixi run python apply_classification.py sampled_input.laz predicted_labels.txt
```

By default, the output is written next to the input with `_classified.laz` appended to the input stem.

The classified sampled cloud is not the final deliverable if the desired output is classification on the original, unsampled input. After creating the sampled classified LAZ, post-process the original input with PDAL using either `pdal translate` or a JSON pipeline with `filters.neighborclassifier`. The original point cloud should be the target layer, and the classified sampled cloud from `apply_classification.py` should be used as the candidate layer so classifications are transferred back onto the full-resolution points.

There are still many manual changes required before this workflow is reliable. In particular, paths, train/validation/test split selection, ignored labels, checkpoint locations, and parts of `datasets/LAS.py`, `train_LAS.py`, and `test_LAS.py` may need hand editing for a given dataset or previously trained model. Future work should make these settings configurable, reduce hard-coded local paths, document the PDAL post-processing pipeline explicitly, and add tests or smoke checks around the inference and label-transfer path.
