# SemEval-2014 Task 4 (Restaurants) — local XML

Place the official SemEval-2014 restaurant aspect XML files here (not committed to git):

- `Restaurants_Train_v2.xml` — training split
- `Restaurants_Test_Gold.xml` — test split

The `download` DVC stage (`python -m src.data.downloader`) reads these paths and writes `data/raw/sentences.csv` and `data/raw/aspects.csv`.

Obtain the files from the SemEval-2014 Task 4 distribution (restaurants domain) per the task license/terms.
