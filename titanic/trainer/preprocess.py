import pandas as pd


def get_age_range():
    df = pd.read_csv('data/train.csv')
    min_age = df['Age'].min()
    max_age = df['Age'].max()
    return min_age, max_age


def get_fare_range():
    df = pd.read_csv('data/train.csv')
    min_fare = df['Fare'].min()
    max_fare = df['Fare'].max()
    return min_fare, max_fare


print(get_age_range())
print(get_fare_range())
