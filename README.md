# Affects AI Research Dataset Toolkit (AARDT)
AARDT, pronouced "_art_," is a utility library for working with AER Datasets available to the academic community for research in 
automated emotion recognition. While it may likely be applied to datasets in other research areas, the author(s)' 
are primarily focused on AER. 

Quick Index of this README:
- Want to know if you can use it? Jump to [Intended Use and License](#license)
- Want to know how to use it? Jump to [Quick Start](#quickstart)
- Want to help out? Jump to [Contributing](#contributing)

## Intended Use and License
<a name="license"></a>
This library is intended for use by only by academic researchers to facilitate advancements in emotion research. It is 
__not for commercial use__ under any circumstances.

This library is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) 
International License. 

__You are free to__:
* __Share__ — copy and redistribute the material in any medium or format 
* __Adapt__ — remix, transform, and build upon the material

__Under the followiung terms__:
* __Attribution__ — You must [give appropriate credit](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-appropriate-credit), provide a link to the license, and [indicate if changes were made](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-indicate-changes). You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* __NonCommercial__ — You may not use the material for [commercial purposes](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-commercial-purposes)
* __ShareALike__ — If you remix, transform, or build upon the material, you must distribute your contributions under the [same license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-same-license) as the original.
* __No additional restrictions__ — You may not apply legal terms or [technological measures](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en#ref-technological-measures) that legally restrict others from doing anything the license permits.

## Quick Start 
<a name="quickstart"></a>
### Concepts
AARDT is designed with a few simple concepts:
1. A _trial_ is a single session in which a participant is exposed to an emotional stimulus, and includes data from one or more sensors captured during the session. This may include ECG, EEG, video or audio recordings of the participant, or whatever else you can think of.
2. A _dataset_ is a collection of trials from multiple participants
3. Sensor data from a trial may need to be processed before being used, and you can do so using the _preprocessor_ pipeline

Most importantly, AER datasets __are not distributed__ with this library. You need to request access to the datasets from the 
dataset authors and download them before following this guide.

### Loading signals from dataset trials
In this example we assume that you have downloaded DREAMER, which is provided in a single JSON file, and that it is stored at `${DREAMER_HOME}/DREAMER_Data.json`.

__Step 1 - Instantiate an AERDataset:__
The AERDataset is the baseclass for all AER datasets, and the details of interacting with each one is in its subclasses, which currently includes `aardt.datasets.ascertain.AscertainDataset` and `aardt.datasets.dreamer.DreamerDataset`. Instantiate a `DreamerDataset` like so:

```python
import os
from aardt.datasets.dreamer import DreamerDataset

# Typically you'd load this from a configuration file... we'll get to that later.
dreamer_home = os.environ['DREAMER_HOME']
ecg_dataset = DreamerDataset(dreamer_home, signals=['ECG'])
```

The `signals` argument takes a list of signals to load into the AERDataset, and can be any proper subset of the signals available within 
the dataset in question. DREAMER provides ECG and EEG recordings, so you can specify any of `['EEG']`, `['ECG']`, or `['EEG','ECG']`. The
order specified does not matter.

__Step 2 - preload and load the dataset__:
Now that you have the DreamerDataset, there are two steps to get it ready for use: _preload_, and _load_. 

The _preload_ step performs any preprocessing of the raw dataset provided by the dataset authors necessary to get it ready to use in AARDT. DREAMER, for example, is provided 
as a single JSON file that is several gigabytes in size. AARDT's preload breaks the JSON into individual Numpy files for each trial, without
ever loading the entire JSON file into memory. This allows it to be used on memory constrained systems, and enables efficient prefetching from 
NAS storage. The preload mechanism is cached, and therefore only runs the first time it is invoked on a given dataset. The preload mechanism only
preloads the signals listed when the dataset was constructed, and will automatically re-run if a new signal is requested that was not 
included in the previous preload.

The _load_ step populates the datasets list of trials, with metadata only. Signal data is lazy-loaded later.

```python
# preload only runs once, regardless of how many times you call it
# so there is no need to check. 
ecg_dataset.preload()

# after preloading, you can load the trials
ecg_dataset.load_trials()
```

__Step 3 - obtain signal data from the trials__: 
With the trials loaded, you can now obtain the signal data and do your analysis on it.

```python
for trial in ecg_signal.trials:
    ecg_signal = trial.load_signal_data('ECG')
    process_ecg(ecg_signal)
```

That's it! And its the same regardless of which AER dataset you are using. If you want to use ASCERTAIN instead of DREAMER, 
just replace `aardt.datasets.dreamer.DreamerDataset` with `aardt.datasets.ascertain.AscertainDataset`, everything else remains
the same.

### Preprocessing signals
The first step in virtually all workloads is to preprocess the signal data, and you can use AARDTs _preprocessors_ to build 
an automated pipeline to do that when signals are loaded from a trial.

For example, lets assume you want to trim each ECG signal in the DREAMER dataset to the final 30 seconds of the sample. You can use 
the `FixedDurationPreprocessor` to do this automatically, like so:

```python
import os
from aardt.datasets.dreamer import DreamerDataset
from aardt.preprocessors import FixedDurationPreprocessor

# Typically you'd load this from a configuration file... we'll get to that later.
dreamer_home = os.environ['DREAMER_HOME']
ecg_dataset = DreamerDataset(dreamer_home, signals=['ECG'])

# Add the preprocessor pipeline to the dataset, for the signal it should be applied to.
# Each signal type can have its own preprocessor pipeline.
ecg_dataset.signal_preprocessors['ECG'] = FixedDurationPreprocessor(signal_duration=30, sample_rate=256, padding_value=0)

# Preload and load the dataset...
ecg_dataset.preload()
ecg_dataset.load_trials()

for trial in ecg_dataset.trials:
    # When you request the signal data from the trial, if the dataset
    # has a preprocessor for that signal type, it will be applied to the
    # signal before it is returned. You are guaranteed to have a 30s 
    # sample here.
    #
    # If the signal was less than 30s originally, it was padded on the left 
    # with 0 values. 
    ecg_signal_30s = trial.load_signal_data('ECG')
    
    # Do something with ecg_signal_30s
```

### Creating your own preprocessor, and preprocessor chaining
You can subclass `SignalPreprocessor` to create your own, and preprocessors can be chained together. For example,
let's say we want to normalize the signal to values between 0 and 1, and also trim them to 30 seconds fixed duration.

```python
import os
import numpy as np 
from sklearn import preprocessing as p 

from aardt.datasets.dreamer import DreamerDataset
from aardt.preprocessors import FixedDurationPreprocessor


class MyNormalizer(aardt.preprocessors.SignalPreprocessor):
    def __init__(self, parent_preprocessor=None):
        super().__init__(parent_preprocessor)

    def process_signal(self, signal):
        min_max_scaler = p.MinMaxScaler() 
        return min_max_scaler.fit_transform(signal) 

dreamer_home = os.environ['DREAMER_HOME']
ecg_dataset = DreamerDataset(dreamer_home, signals=['ECG'])

# Create a pipeline by instantiating MyNormalizer, and passing in a 
# FixedDurationPreprocessor as its parent. You can chain as many 
# preprocessors together as you need. The parent will always be called
# first - so the outermost preprocessor is the last one to execute.
pipeline = MyNormalizer(
    FixedDurationPreprocessor(signal_duration=30, sample_rate=256, padding_value=0)
)

ecg_dataset.signal_preprocessors['ECG'] = pipeline

# Preload and load the dataset...
ecg_dataset.preload()
ecg_dataset.load_trials()

for trial in ecg_dataset.trials:
    # Here, the signal data is already trimmed or padded to be 30s long, 
    # and has been normalized using the MinMaxScaler to values between 
    # 0 and 1.
    ecg_signal = trial.load_signal_data('ECG')
    
```

Note that the order of your pipeline is critically important. Here, we apply `FixedDurationPreprocessor` first, _before_ 
we normalize the values. This may be problematic, since ECG signals are prone to baseline wander. Padding zero values in 
before normalization will artificially skew the normalization results. It would be better to normalize the signal first, 
then apply the `FixedDurationPreprocessor`:

```python
pipeline = FixedDurationPreprocessor(
    signal_duration=30, 
    sample_rate=256, 
    padding_value=0, 
    parent_preprocessor=MyNormalizer()    
)
```

### Using with TensorFlow
To facilitate use with TensorFlow, use the `TFDatasetWrapper` to decorate your `AERDataset` as a `tf.data.Dataset` suitable 
for use with `tf.model.fit()`

```python
import aardt.datasets

# Don't forget to setup your preprocessor pipelines, then preload and 
# load the dataset first!
tfdsw = aardt.datasets.TFDatasetWrapper(ecg_dataset)

# Create the tf.data.Dataset 
tfdataset = tfdsw(batch_size=64, buffer_size=500, repeat=1)

# Setup your tensorflow model, then use the tfdataset:
myModel = get_tensorflow_model()

# Train your model using preprocessed signals from the AERDataset
myModel.fit(tfdataset)
```

___todo:__ need to implement method for train/test splits_

TFDatasetWrapper provides a `tf.data.dataset` which will prefetch up to `buffer_size` trials at random, creating batches of 
size `batch_size`, and will iterate the dataset `repeat` times. The prefetch queue uses `tf.data.AUTOTUNE` to self-optimize.


## Contributing <a name="contributing"></a>
We are happy to support you by accepting pull requests that make this library more broadly applicable, or by accepting
issues to do the same. If you have an AER dataset you would like us to integrate, please open an issue for that as well, 
but we will be unable to process issues requesting integration with non-AER datasets at this time.

If you would like to get involved by maintaining dataset integrations in other areas of research, please get in touch 
and we'd be happy to have the help!