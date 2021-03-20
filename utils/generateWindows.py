import datetime
import calendar

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = calendar.monthrange(year,month)[1]
    return datetime.datetime(year, month, day)

def generate_windows_from(start_date, stop_date, interval):
    # Format of argument:
    # start: yyyymmdd
    # stop: yyyymmdd
    # interval: Number of months for each window (E.g. Jan to Mar is 3)
    start = datetime.datetime(int(start_date[:4]), int(start_date[4:6]), int(start_date[-2:]))
    stop = datetime.datetime(int(stop_date[:4]), int(stop_date[4:6]), int(stop_date[-2:]))
    end = add_months(start, interval-1)
    res = []

    while end <= stop:
        res.append((start.strftime('%Y%m%d'), end.strftime('%Y%m%d')))
        start = add_months(start, 3)
        end = add_months(end, 3)

    return res

# def generate_txt(lst):
#     file1 = open("windows.txt","w") 
#     for rng in lst:
#         # print (rng)
#         output = rng[0] + "," + rng[1] + "\n"
#         file1.writelines(output)