#!/bin/bash
mkdir data
cd data
mkdir deepfashion2
cd deepfashion2
gdown --id 12DmrxXNtl0U9hnN1bzue4XX7nw1fSMZ5
gdown --id 1hsa-UE-LX8sks8eAcGLL-9QDNyNt6VgP
gdown --id 1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK
gdown --id 1O45YqhREBOoLudjA06HcTehcEebR0o9y

unzip -P 2019Deepfashion2** json_for_validation.zip
unzip -P 2019Deepfashion2** test.zip
unzip -P 2019Deepfashion2** train.zip
unzip -P 2019Deepfashion2** validation.zip
rm *.zip

cd ..
cd ..
gdown 1SwtyBnaLMEUtdh7NyMnfNQ9qGz1QOz4O
unzip OneDrive_2_2022-6-25.zip