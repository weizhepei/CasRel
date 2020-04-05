Download the CopyR's WebNLG dataset at https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj

Unzip it and all we need is webnlg/pre_processed_data/*

Run generate.py to transfromer the numerical mentions to string mentions. Move the transformed files to ../

dev.json -> new_dev.json (then rename it test.json)
train.json -> new_train.json (then rename it train.json)
valid.json -> new_valid.json (then rename it dev.json)

Also, the test data will be split by sentence types, i.e., Normal, SEO, and EPO. Then move the split files to ../test_split_by_type/

