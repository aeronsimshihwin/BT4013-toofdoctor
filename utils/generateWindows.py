import datetime
import calendar
from datetime import timedelta

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = calendar.monthrange(year,month)[1]
    return datetime.datetime(year, month, day)

def backdate(date_input, num_days):
    res = date_input + timedelta(1)
    # date_input is Sunday, get the latest Friday
    if date_input.isoweekday() == 7: res = res - timedelta(2)

    # date_input is Saturday, get the latest Friday
    elif date_input.isoweekday() == 6: res = res - timedelta(1)
    
    elif date_input.isoweekday() != 5:
        res = res + timedelta(5 - date_input.isoweekday())
        num_days += (5 - date_input.isoweekday())

    while num_days > 5:
        res = res - timedelta(7)
        num_days -= 5

    if num_days != 0:
        res = res - timedelta(num_days)
    return res

def generate_windows_from(_start, _stop, _interval):
    # Format of argument:
    # _start: yyyymmdd
    # _stop: yyyymmdd
    # _interval: Number of months for each window (E.g. Jan to Mar is 3)

    start_date_input = datetime.datetime(int(_start[:4]), int(_start[4:6]), int(_start[-2:]))
    start_date = backdate(start_date_input, 504)

    stop_date = datetime.datetime(int(_stop[:4]), int(_stop[4:6]), int(_stop[-2:]))
    end = add_months(start_date, _interval-1)
    res = []

    while end <= stop_date:
        res.append((start_date.strftime('%Y%m%d'), end.strftime('%Y%m%d')))
        end = add_months(end, _interval-1)

    return res

# def generate_txt(lst):
#     file1 = open("windows.txt","w") 
#     for rng in lst:
#         # print (rng)
#         output = rng[0] + "," + rng[1] + "\n"
#         file1.writelines(output)