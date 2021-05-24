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
