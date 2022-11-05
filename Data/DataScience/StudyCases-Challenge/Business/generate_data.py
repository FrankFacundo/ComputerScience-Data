import numpy as np
import pandas as pd
import re
import uuid
from tqdm import tqdm


def random_acquisition_channel():
    channels = ["TV" , 'Radio', 'Web', 'Billboard']
    return np.random.choice(channels)

def generate_random_date_hour():
    # We assume ages from 18 to 80
    year = int(np.random.uniform(1942,2004))
    # print('year : {}'.format(year))

    months_days = {1: 31, 2: 28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    month = int(np.random.uniform(1,13))
    # print('month : {}'.format(month))
    #print(list(range(1, months_days[1]+1)))
    day = np.random.choice(list(range(1, months_days[1]+1)))
    # print('day : {}'.format(day))

    hour = int(np.random.uniform(0,24))
    # print('hour : {}'.format(hour))

    minute = int(np.random.uniform(0,60))
    # print('minute : {}'.format(minute))

    second = int(np.random.uniform(0,60))
    # print('second : {}'.format(second))

    result = '{}/{}/{}-{}:{}:{}'.format(year, month, day, hour, minute, second)
    return result

def generate_id(without_dash=True):
    if without_dash:
        return re.sub('-', '', str(uuid.uuid4()))
    else:
        return uuid.uuid4()

def generate_customer_dataframe(number_rows=10):
    cols = ['customer_id', 'birth_date', 'acquisition_channel']
    # df = pd.DataFrame(columns=cols)
    rows = []
    for i in tqdm(range(number_rows)):
        row = [generate_id(), generate_random_date_hour(), random_acquisition_channel()]
        rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    # print(df.head())
    return df

# generate_customer_dataframe()

#############################################################

def generate_order_dataframe(number_rows=10):
    return

def generate_random_number_orders():
    number_of_orders = np.minimum(int(np.random.exponential(scale=1.0)+1), 5)
    return number_of_orders

def get_list_number_orders_by_customer(rows_in_orders=100):
    total_orders = 0
    list_number_orders_by_customer = []
    pbar = tqdm(total=rows_in_orders)
    while True:
        number_of_orders = generate_random_number_orders()
        pbar.update(number_of_orders)
        if total_orders + number_of_orders < rows_in_orders:
            total_orders += number_of_orders
            list_number_orders_by_customer.append(number_of_orders)
        if total_orders + number_of_orders == rows_in_orders:
            list_number_orders_by_customer.append(number_of_orders)
            pbar.close()
            return list_number_orders_by_customer
        if total_orders + number_of_orders > rows_in_orders:
            list_number_orders_by_customer.append(rows_in_orders - total_orders)
            pbar.close()
            return list_number_orders_by_customer

def generate_customers_id_for_sales(customers_id, rows_in_orders=100):
    list_number_orders_by_customer = get_list_number_orders_by_customer(rows_in_orders)
    # print(list_number_orders_by_customer)

    # Generar lista de customers id con la ayuda de list_number_orders_by_customer
    # Agregar esa lista como columna CUSTOMER_ID en la tabla sales_50

    customers_id_orders = []
    i = 0

    for number_of_orders in tqdm(list_number_orders_by_customer):
        to_add = [customers_id[i%len(customers_id)]] * number_of_orders
        customers_id_orders = customers_id_orders + to_add
        i+=1
    return customers_id_orders

def generate_orders_dataframe(rows=None):
    TRANSACTION_DATE = "transacion_date"
    PRODUCT = "product"
    PRICE = "price"
    QUANTITY = "quantity"
    ORDERS_ID = "orders_id"
    CUSTOMER_ID = "customer_id"

    sales = pd.read_csv("Year-2010-2011.csv", encoding='unicode_escape')
    if rows:
        sales = sales.head(rows)
    rows = len(sales)
    sales.rename(columns={"InvoiceDate": TRANSACTION_DATE,
                    "Description": PRODUCT, "Price": PRICE, "Quantity": QUANTITY}, inplace=True)
    sales = sales[[TRANSACTION_DATE, PRODUCT, PRICE, QUANTITY]]
    sales[TRANSACTION_DATE] = pd.to_datetime(sales[TRANSACTION_DATE], format='%m/%d/%Y %H:%M')
    # print(sales[TRANSACTION_DATE])
    sales[TRANSACTION_DATE] = sales[TRANSACTION_DATE].apply(lambda x: x.replace(year = np.random.choice([2016, 2017, 2018])))
    orders_id = [generate_id() for i in range(len(sales))]
    sales[ORDERS_ID] = orders_id

    customers = generate_customer_dataframe(int(rows/10))
    customers_id = customers[CUSTOMER_ID].tolist()

    customers_id_orders = generate_customers_id_for_sales(customers_id, rows)

    sales[CUSTOMER_ID] = customers_id_orders

    return customers, sales

# customer_table = generate_customer_dataframe()
customer_table, orders_table = generate_orders_dataframe()

customer_table.to_csv("customer_data.csv", index=False)
orders_table.to_csv("order_data.csv", index=False)
