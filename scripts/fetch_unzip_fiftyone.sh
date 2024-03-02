#!/bin/bash

# download from google drive
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1UxbJs-YEuI9rhUygtZf5mtxH5iXOAWX0', 'fiftyone.tgz', quiet=False)"

# unzip into dataset directory
tar -xvzf fiftyone.tgz -C $DATASET_DIR

# create fiftyone dataset
fiftyone datasets create --name rach3 -d $DATASET_DIR --type fiftyone.types.FiftyOneDataset

# remove the tgz file
rm fiftyone.tgz
