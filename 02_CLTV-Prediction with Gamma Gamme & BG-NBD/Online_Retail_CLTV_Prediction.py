######################################################################################################
####                                                                                              ####
####    Online Retail CLTV PREDİCTİON --> {CLTV VALUE PREDİCTİON WİTH BG/NBD AND GAMMA GAMMA}     ####
####                                                                                              ####
######################################################################################################

# -------------------------------------------------------------------------------------------------------------------

# Bussines Problem:

""""
The UK-based retail company wants to set a roadmap for its sales and marketing activities.
In order for the company to make a medium-long-term plan,
it is necessary to estimate the potential value that existing customers will provide to the company in the future.
"""

# About the Dataset:
""""
The data set named Online Retail II includes online sales transactions of a UK-based retail company between 
01/12/2009 - 09/12/2011. The company's product catalog includes souvenirs and it is known 
that most of its customers are wholesalers.
"""

# Features:

# Total Features : 8
# Total Row : 541.909
# CSV File Size : 45.6 MB

""""
- InvoiceNo : Invoice Number (Additional information: If this code starts with C, it means that the transaction has been cancelled.)
- StackCode : Product Code (Unique for each product)
- Description : Product Name
- Quantity : Number of Products (How many of the products on the invoice were sold)
- InvoiceDate : Invoice Date
- UnitPrice : Invoice Price (in Sterling)
- CustomerID : Unique Customer Number
- Country : Country Name
"""

# ----------------------------------------------------------------------------------------------------------------------


import datetime as dt
import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 600)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()


def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

missing_values_analysis(df)

""""
             Total Missing Values  Ratio
Description                  1454  0.270
Customer ID                135080 24.930
"""

df.dropna(inplace = True)


# Outlier Analysis
def outlier_thresholds(df, feautre):
    q_1 = df[feautre].quantile(0.01)
    q_3 = df[feautre].quantile(0.99)
    IQR_Range = q_3 - q_1
    up_limit = q_3 + 1.5 * IQR_Range
    low_limit = q_1 - 1.5 * IQR_Range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    ####dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def create_cltv_p(dataframe, country ,month=3):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe['Country'].str.contains(country, na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

    return cltv_final

print("---------------------------------------")

_6_month_ = create_cltv_p(df,country='United Kingdom',month=6)

_6_month_.head()

# 6 MONTH {England - > Country} -->  A 6-month CLTV Forecast for UK customers using 2010-2011 data!
""""
    Customer ID  recency      T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit       clv
0    12747.000   52.286 52.857         11   381.455                 0.202                  0.808                  2.406                  387.823  1937.009
1    12748.000   53.143 53.429        209   154.564                 3.238                 12.916                 38.490                  154.709 12366.072
2    12749.000   29.857 30.571          5   814.488                 0.167                  0.666                  1.976                  844.095  3445.923
3    12820.000   46.143 46.714          4   235.585                 0.104                  0.415                  1.234                  247.081   631.934
4    12822.000    2.286 12.571          2   474.440                 0.129                  0.513                  1.511                  520.829  1612.133
"""

# Calculating 1 month CLTV for UK Customers

_1_month_ = create_cltv_p(df,country='United Kingdom',month=1)

_1_month_.sort_values(by = 'clv', ascending = False).head(10)

"""
       Customer ID  recency      T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit       clv
2486    18102.000   52.286 52.571         60  3584.888                 0.965                  3.851                 11.474                 3595.193 14884.500
589     14096.000   13.857 14.571         17  3159.077                 0.723                  2.873                  8.490                 3191.387  9855.142
2184    17450.000   51.286 52.571         46  2629.530                 0.745                  2.972                  8.856                 2639.420  8434.508
2213    17511.000   52.857 53.429         31  2921.952                 0.507                  2.024                  6.031                 2938.275  6394.139
1804    16684.000   50.429 51.286         28  2120.047                 0.477                  1.901                  5.665                 2133.204  4360.926
587     14088.000   44.571 46.143         13  3859.602                 0.260                  1.036                  3.083                 3911.320  4355.369
406     13694.000   52.714 53.429         50  1267.363                 0.798                  3.184                  9.489                 1271.785  4354.335
1173    15311.000   53.286 53.429         91   667.597                 1.429                  5.699                 16.984                  668.894  4098.737
133     13089.000   52.286 52.857         97   605.187                 1.532                  6.112                 18.211                  606.294  3983.922
1485    16000.000    0.000  0.429          3  2055.787                 0.416                  1.641                  4.780                 2181.326  3843.409
"""


# Calculating 12 month CLTV for UK Customers

_12_month_ = create_cltv_p(df,country='United Kingdom',month=12)

_12_month_.sort_values(by = 'clv', ascending = False).head(10)

""""
      Customer ID  recency      T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit        clv
2486    18102.000   52.286 52.571         60  3584.888                 0.965                  3.851                 11.474                 3595.193 163586.718
589     14096.000   13.857 14.571         17  3159.077                 0.723                  2.873                  8.490                 3191.387 104893.742
2184    17450.000   51.286 52.571         46  2629.530                 0.745                  2.972                  8.856                 2639.420  92691.903
2213    17511.000   52.857 53.429         31  2921.952                 0.507                  2.024                  6.031                 2938.275  70283.955
1804    16684.000   50.429 51.286         28  2120.047                 0.477                  1.901                  5.665                 2133.204  47889.189
406     13694.000   52.714 53.429         50  1267.363                 0.798                  3.184                  9.489                 1271.785  47870.662
587     14088.000   44.571 46.143         13  3859.602                 0.260                  1.036                  3.083                 3911.320  47687.833
1173    15311.000   53.286 53.429         91   667.597                 1.429                  5.699                 16.984                  668.894  45066.570
133     13089.000   52.286 52.857         97   605.187                 1.532                  6.112                 18.211                  606.294  43794.263
1057    15061.000   52.571 53.286         48  1108.308                 0.769                  3.069                  9.145                 1112.347  40347.776
"""

# Segment all customers into 4 segments [according] to 6 MONTHS CTLV FOR CUSTOMERS 2010-2011
_6_month_["SEGMENT"] = pd.qcut(_6_month_["clv"], 4, labels=["D", "C", "B", "A"])

_6_month_.sort_values(by = 'clv', ascending = False).head(10)

""""
      Customer ID  recency      T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_purc_3_month  expected_average_profit       clv SEGMENT
2486    18102.000   52.286 52.571         60  3584.888                 0.965                  3.851                 11.474                 3595.193 85648.497       A
589     14096.000   13.857 14.571         17  3159.077                 0.723                  2.873                  8.490                 3191.387 55646.829       A
2184    17450.000   51.286 52.571         46  2629.530                 0.745                  2.972                  8.856                 2639.420 48531.953       A
2213    17511.000   52.857 53.429         31  2921.952                 0.507                  2.024                  6.031                 2938.275 36796.033       A
1804    16684.000   50.429 51.286         28  2120.047                 0.477                  1.901                  5.665                 2133.204 25082.350       A
406     13694.000   52.714 53.429         50  1267.363                 0.798                  3.184                  9.489                 1271.785 25060.002       A
587     14088.000   44.571 46.143         13  3859.602                 0.260                  1.036                  3.083                 3911.320 25009.458       A
1173    15311.000   53.286 53.429         91   667.597                 1.429                  5.699                 16.984                  668.894 23590.685       A
133     13089.000   52.286 52.857         97   605.187                 1.532                  6.112                 18.211                  606.294 22927.005       A
1057    15061.000   52.571 53.286         48  1108.308                 0.769                  3.069                  9.145                 1112.347 21122.487       A
"""

# ----------------------------------------------------------------------------------------------------------------------