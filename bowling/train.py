import numpy as np

def select_schedule_item(score, schedule, thresholds):
    assert len(schedule) == len(thresholds)
    return schedule[ np.max( (score >= thresholds)*np.arange(len(thresholds))) ]

