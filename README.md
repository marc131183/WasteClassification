# WasteClassification

This is a project to help seperate waste with an unmanned station. In the scope of this project a dataset of about 5000 images is collected (see [data/cleaned](https://github.com/marc131183/WasteClassification/tree/main/data/cleaned)) and used to train various image classification models.

## Dataset

The dataset consists of 12 classes. Since the rarity of items differs (which makes it hard to make a balanced dataset with so many classes), it was decided to focus on 4 classes (7133, 7051, 7055, 7042) and have the other 8 combined in one. Information about the collected classes can be seen below:

| **Norwegian Waste ID** | **Number of images** | **Contents**                  |
|------------------------|----------------------|-------------------------------|
| 7133                   | 1037                 | Cleaning products             |
| 7051                   | 1029                 | Paint, glue and varnish waste |
| 7055                   | 1029                 | Spray                         |
| 7042                   | 902                  | Organic solvents and halogen  |
| 7023                   | 247                  | Petrol and diesel filters     |
| 7121                   | 192                  | Isocyanates                   |
| 7134                   | 117                  | Acidic organic waste          |
| 7132                   | 92                   | Inorganic bases               |
| 7152                   | 84                   | Tarry waste                   |
| 7151                   | 78                   | Organic waste with halogen    |
| 7123                   | 74                   | Hardeners                     |
| 7011                   | 66                   | Waste oil                     |
