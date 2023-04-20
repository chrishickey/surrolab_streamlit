
import torch
import math
import numpy as np

from surrolab.dataloader.pytorch.converters import TorchNumpyzToAudio
from surrolab.datamanager.data_converters import BytesToNumpyzCompressedFile
from surrolab.datamanager.data_validators import AudioValidator
from surrolab.datamanager.boto3.tabular import Boto3S3TabularDataManager
from surrolab.dataloader.pytorch.boto3 import Boto3Dataset
from surrolab.models.pytorch.audio import SurrolabVGGish
from surrolab.dataloader.pytorch.converters import TorchNumpyWaveformToVggishInput

NUM_EPOCHS = 5

manager = Boto3S3TabularDataManager()
manager.config_from_mapping(
    dict(
        boto3_endpoint_url="http://10.10.30.23:30522",
        boto3_aws_access_key_id='minio',
        boto3_aws_secret_access_key='minio123'
    )
)


class AudioPreparer:

    def __call__(self, data):
        # Extracting timestamps and audio values from the numpy array.
        numpy_array = np.array(data)
        timestamps = numpy_array[:, 0]
        audiovals = numpy_array[:, 1]
        # Calculating the sample rate and saving the file to the output path.
        num_samples, = timestamps.shape
        sample_rate = num_samples // math.ceil(timestamps[-1])
        return {
            "data": audiovals,
            "sample_rate": sample_rate
        }


manager.upload_zip_of_tabular_data(
    s3_path='chris_demo_example_5567', # Identifier for where to store the uploaded data in s3
    zip_path="test_data.zip", # Path to the zip file
    file_type=".csv", # Specific type of files to look for to upload (non csv files will be ignored)
    # Some callable function that will be used to validate each line in the csv file.
    data_converter=BytesToNumpyzCompressedFile(
        prepare_pre_save=AudioPreparer(),
        validate_pre_save=AudioValidator()
    ),
    raise_invalid=False, # Boolean dictating whether to raise exception if any line fails validation.
    compress=False, # Boolean indicating whether to compress files before uploading to Minio
)


mel_spectrogram_transform = torch.nn.Sequential(
    TorchNumpyzToAudio(),
    TorchNumpyWaveformToVggishInput(num_seconds=3)
)


dataset = Boto3Dataset(
    data_manager=manager, # The manager class used to access data
    data_path="chris_demo_example_5567", # The s3_path to
    label_identifiers=[
        "normal",
        "belt"
    ],  # Substrings in the file names used to identify the classes from the file names
    decompress=False,  # Flag to indicate whether data needs to be decompressed
    converter=mel_spectrogram_transform,
    label_converter=int
)

data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                sampler=None,
)

model = SurrolabVGGish(num_classes=2)
# loss function
criterion = torch.nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# scheduler
for _ in range(NUM_EPOCHS):
    total_acc = 0
    for i, (data, label) in enumerate(data_loader):
        output = model(data)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        total_acc += acc
        if i and not i % 10:
            print(f"Accuracy mid batch is {round(float(total_acc / (i + 1)), 2)}")
    print(f"Epoch accuracy on train set is {round(float(total_acc / (i + 1)), 2)}")
