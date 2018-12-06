JOB_ID=`whoami`_titanic_prediction_`date +%m_%d_%Y_%H_%M_%S`

python -m trainer.train \
    --train-files data/train.csv \
    --eval-files data/eval.csv \
    --job-dir titanic_ltung \
    --train-steps 1000 \
    --eval-steps 100


gcloud ml-engine local train --package-path trainer \
    --module-name trainer.train \
    --job-dir titanic_ltung \
    -- \
    --train-files data/train.csv \
    --eval-files data/eval.csv \
    --train-steps 1000 \
    --eval-steps 100


gcloud ml-engine jobs submit training ${JOB_ID} \
    --runtime-version 1.8 \
    --job-dir=gs://ltung-training/titanic/trained-models/${JOB_ID} \
    --module-name train \
    --region us-central1 \
    --package-path ./ \
    -- \
    --train-files gs://ltung-training/titanic/train.csv \
    --eval-files gs://ltung-training/titanic/eval.csv \
    --eval-steps 100

tensorboard --logdir=titanic_ltung
tensorboard --logdir=gs://ltung-training/titanic/trained-models/${JOB_ID} --port=8080

tensorboard --logdir=gs://ltung-training/titanic/trained-models/ltung_titanic_prediction_08_10_2018_16_22_23 --port=8080

