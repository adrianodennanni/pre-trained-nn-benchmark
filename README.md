# Pre-trained Neural Networks Benchmark
Benchmark comparing pre-trained vs random-init Neural Networks

#### Downloading pets dataset
- `bundle install`
- `bundle exec ruby download_dataset_open_images.rb`

#### Generating a captcha dataset
- `bundle install`
- `bundle exec ruby generate_dataset_captcha.rb`

## Results

### Pet Classifier (Xception)

##### Pre-trained
| Epoch | Validation Accuracy |
|:-----:|:-------------------:|
|     1 |              87.41% |
|     2 |              90.90% |
|     3 |              93.83% |
|     4 |              93.69% |
|     5 |              91.64% |
|     6 |              89.91% |
|     7 |              91.75% |
|     8 |              96.47% |
|     9 |              91.85% |
|    10 |              83.49% |


##### Random-init
| Epoch | Validation Accuracy |
|:-----:|:-------------------:|
|     1 |              00.00% |
|     2 |              00.00% |
|     3 |              00.00% |
|     4 |              00.00% |
|     5 |              00.00% |
|    10 |              00.00% |
