<h1 align="center">Automated Grading of Baseball Trading Cards using Convolutional Neural Networks</h1>  

### Authors:  
- Brian Miller (brian.s.miller@du.edu)
- Sean Wendlandt (sean.wendlandt@du.edu)

### Project Overview:
The condition of a trading card significantly influences its worth. However, the current methods to assess a card's condition are insufficient. Non-professionals often lack the accuracy required for card grading, and obtaining professional evaluations is too expensive and time-consuming.

This project aims to automate the card grading process using machine learning. Our neural network utilizes the pretained model ResNet-50 
combined with custom classification and output layers. 

**Data Preprocessing Steps:**
- Web scraping to obtain data
- Manual removal of innacurate data
- Dataset balancing
- Automated removal of PSA label
- Standardize photo dimensions for CNN

### Description of CNN:
![alt text](readme_photos/photo_diagram.png)
- ResNet-50 (pretrained on ImageNet, global average pooled) feeds into dropout and a single softmax prediction layer  
- The diagram shows the original design with three dense layers [1024, 512, 128] in between; they made the model
  memorise the training images, so they are now optional (`--dense_units 1024 512 128`)  
- The training targets spread some weight over the neighbouring grades, so that predicting a PSA 8 for a PSA 9
  counts as less wrong than predicting a PSA 2  

Training is done in two stages:
- Warm-up: 3 epochs, ResNet-50 is frozen and only the prediction layer is trained
- Fine-tuning: up to 15 epochs, ResNet-50 is trained as well, with a low learning rate that decays over the epochs

A tenth of the training images is held out as a validation set, which picks the best epoch and stops the training
early when it no longer improves. The test images are only used for the final evaluation.

### Setup:
The code is one installable Python package, `card_grading` (Python 3.10+):
```
pip install -e ".[all]"
```
Smaller installs are available: `.[train]` (TensorFlow), `.[scrape]`, `.[scan]` (OpenCV, for splitting scans),
`.[notebooks]` and `.[dev]` (tests and linting).

### Usage:
Installing the package adds one command per step of the pipeline:

| Step | Command |
| --- | --- |
| 1. Download PSA's scans and grades of baseball cards | `card-grading-fetch-psa --start_cert 80000000 --count 5000 --output_dir ./final_10k` |
| 2. Crop the PSA label off and standardize the size | `card-grading-crop --parent_dir ./final_10k --output_dir ./final10k_cropped` |
| 3. Add flipped copies of the training images | `card-grading-flip ./data/train` |
| 4. Train and evaluate the model | `card-grading-train --base_path ./data` |

Run any command with `--help` for all of its options. Each training run writes its training log (`log.csv`),
parameters, test results (`metrics.json`) and the best model (`best_model.keras`) to `logs/<timestamp>-<pid>/`.
Progress is saved after every epoch: `card-grading-train --base_path ./data --run_dir logs/<run>` continues a run
that was interrupted.

### Assessing your own cards:
Two commands turn scans of loose cards into a spreadsheet:
```
card-grading-split-scan ./scans --output_dir ./cards
card-grading-assess --images ./cards --output assessment.csv --model logs/<run>/best_model.keras
```
- `card-grading-split-scan` cuts every scan in a folder into one straightened, upright image per card, numbered row
  by row. Put the cards on a plain background that contrasts with their edges (black paper behind white-bordered
  cards) and leave a gap between them: cards that touch are left out and reported. Flatbed scans at 300 dpi or
  more work best; photos taken from straight above work too.
- `card-grading-assess` writes one row per card image with its centring (the left/right and top/bottom borders as
  shares, such as `45/55`) and, when a trained model is given with `--model`, the estimated grade, how confident the
  model is, and the probability of every grade. Without `--model` only the centring is measured, which needs no
  TensorFlow.
- The centring is only meaningful for a single card cropped to its edges that has a plain border around its picture.
- A model trained on photos of cards in PSA cases has not seen loose cards, so treat its grades for those as rough
  hints until it has been trained on images like yours.

### Collecting images through the PSA API:
`card-grading-fetch-psa` uses the [PSA Public API](https://www.psacard.com/publicapi) to look cert numbers up and
download PSA's own scans of the baseball cards among them, sorted into `psa1` ... `psa10` folders by grade.
- It needs an API token: sign in at the link above with a PSA account, generate the token on the documentation
  page and put it in the `PSA_API_TOKEN` environment variable.
- A token alone is not enough. PSA only answers accounts it has approved for the API; others get
  "Access to this API is limited to approved customers." Ask PSA for access (`collectors-apis@collectors.com`).
- PSA's standard limit is 100 API calls per day and a card takes up to two calls, so a large dataset needs a higher
  limit as well. The download stops cleanly at the limit; running the same command again continues where it stopped.
- Cards of other categories, half grades and grades with a qualifier (such as OC) are skipped. Every cert number
  that was looked up is listed in `psa_metadata.csv`.
- PSA only has scans for certs of recent years. Use `--certs_file` to look up specific cert numbers instead of a range.
- Check PSA's API agreement before using the images, in particular for training models.

`card-grading-scrape` is the scraper the original dataset was collected with. Collectors.com no longer offers the
search it reads, so it is kept for reference only.

### Training on a public dataset:
The original images are not public, and Collectors.com no longer offers the search they were scraped from.
[PSA-Baseball-Grades](https://github.com/samsilverman/PSA-Baseball-Grades) is a public set of 11,500 slab photos
(1,150 per PSA grade, 150x200 pixels) that the pipeline can be run on:
```
git clone --depth 1 https://github.com/samsilverman/PSA-Baseball-Grades
card-grading-crop --parent_dir PSA-Baseball-Grades --output_dir data --target_width 150
card-grading-train --base_path data --image_height 160 --image_width 160
```
Things to know about that dataset: its repository states no license, its author warns that some grades are wrong
and some images are irrelevant, and it contains exact duplicate images (the train/test split sets those aside in
`data/duplicates/`).

With the commands above, the model reaches the following on the 2,103 test images (guessing at random would get
10% of the exact grades right):

| Exact grade | Within 1 grade | Within 2 grades | Mean grade difference |
| --- | --- | --- | --- |
| 27% | 65% | 82% | 1.5 |

The images are too small to show much of the wear that separates neighbouring grades, and the model is at its
best on the lowest and highest grades.

### Development:
```
ruff check .
pytest
```

### Data:
All 10k cards were downloaded from [Collectors.com](https://www.collectors.com/trading-cards/sport-baseball-cards/20003?lowgrade=1&highgrade=10&gradingservice=2&page=1)

### Description of Files:
```
pyproject.toml            package metadata, dependencies and tool settings
src/card_grading/
    data.py               PSA grades, file naming and the train/test split
    psa_api.py            downloader for PSA's scans and grades (PSA Public API)
    scraping.py           web scraper for Collectors.com (the original data source, no longer online)
    cropping.py           PSA label removal and image resizing
    scans.py              cuts a scan of several cards into one image per card
    centering.py          measures how well a card's picture is centred between its borders
    assess.py             centring and estimated grade for a folder of card images
    augmentation.py       flipped copies of the training images
    training.py           ResNet-50 model, training and evaluation
scripts/                  shell scripts that run the commands in bulk (dropout sweep, grid search, all grades)
notebooks/                metadata building, dataset balancing and EDA
tests/                    unit tests
testing_data_folder/      a small sample of the dataset, in the layout the training expects
```
