import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter
from IPython.display import display


def size(obj):
    return "{0:.2f} MB".format(sys.getsizeof(obj) / (1000 * 1000))

visits = pd.read_csv('mydata.csv')

print('visits -- dimensions: {0};   size: {1}'.format(visits.shape, size(visits)))

display(visits.head())

visits = visits.set_index('visit_id')['product_id'].rename('item_id')

print('dimensions: {0};   size: {1};   unique_visits: {2};   unique_items: {3}'
      .format(visits.shape, size(visits), len(visits.index.unique()), len(visits.value_counts())))

def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else:
        return pd.Series(Counter(iterable)).rename("freq")

def visit_count(visit_item):
    return len(set(visit_item.index))

def get_item_pairs(visit_item):
    visit_item = visit_item.reset_index().values
    for visit_id, visit_object in groupby(visit_item, lambda x: x[0]):
        item_list = [item[1] for item in visit_object]

        for item_pair in combinations(item_list, 2):
            yield item_pair

def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
            .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
            .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))

def association_rules(visit_item, min_support):
    print("Starting visit_item: {:22d}".format(len(visit_item)))

    item_stats            = freq(visit_item).to_frame("freq")
    item_stats['support'] = item_stats['freq'] / visit_count(visit_item)

    qualifying_items      = item_stats[item_stats['support'] >= min_support].index
    visit_item            = visit_item[visit_item.isin(qualifying_items)]

    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining visit_item: {:21d}".format(len(visit_item)))

    visit_size             = freq(visit_item.index)
    qualifying_visits      = visit_size[visit_size >= 2].index
    visit_item             = visit_item[visit_item.index.isin(qualifying_visits)]

    print("Remaining visits with 2+ items: {:11d}".format(len(qualifying_visits)))
    print("Remaining visit_item: {:21d}".format(len(visit_item)))

    item_stats             = freq(visit_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / visit_count(visit_item)


    # Get item pairs generator
    item_pair_gen          = get_item_pairs(visit_item)


    # Calculate item pair frequency and support
    item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_visits)

    print("Item pairs: {:31d}".format(len(item_pairs)))

    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)

    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])

    # Return association rules sorted by lift in descending order
    return item_pairs.sort_values('lift', ascending=False)

rules = association_rules(visits, 0.001)

display(rules)
