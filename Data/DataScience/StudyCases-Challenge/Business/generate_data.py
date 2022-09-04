import numpy as np

def random_acquisition_channel():
    channels = ["TV" , 'Radio', 'Web', 'Billboard']
    return np.random.choice(channels)

def generate_random_date_hour():
    # We assume ages from 18 to 80
    year = int(np.random.uniform(1942,2004))
    print('year : {}'.format(year))

    months_days = {1: 31, 2: 28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    month = int(np.random.uniform(1,13))
    print('month : {}'.format(month))
    #print(list(range(1, months_days[1]+1)))
    day = np.random.choice(list(range(1, months_days[1]+1)))
    print('day : {}'.format(day))

    hour = int(np.random.uniform(0,24))
    print('hour : {}'.format(hour))

    minute = int(np.random.uniform(0,60))
    print('minute : {}'.format(minute))

    second = int(np.random.uniform(0,60))
    print('second : {}'.format(second))

    result = '{}/{}/{}-{}:{}:{}'.format(year, month, day, hour, minute, second)
    return result