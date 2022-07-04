#########################################
# Online_Retail_II RFM ANALYSİS
#########################################


# -------------------------------------------------------------------------------------------------------------------

# Bussines Problem:

""""
The UK-based retail company wants to segment its customers and determine its marketing strategies according to these segments.
He thinks that conducting special marketing studies for customer segments that exhibit common behaviors will increase revenues.

Accordingly, RFM Analysis will be used according to segments in this project.
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

# Project steps:

""""
1.) Understanding and preparing data
2.) Calculating RFM Metrics
3.) Generating RFM Scores and Converting to a Single Variable
4.) Segment Definition of RFM Score
5.) Action Time!
"""

# -------------------------------------------------------------------------------------------------------------------

# Project :


import datetime as dt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

df_ = pd.read_excel("online_retail_II 01.27.45.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()


# 1.) Understanding and preparing data
# -------------------------------------------------------------------------------------------------------------------


df.head()
df.shape # Out: 541910, 8

df.describe().T

""""
                 count      mean      std        min       25%       50%    75%       max  
Quantity    541910.000     9.552  218.081 -80995.000     1.000     3.000   10.000    80995.000  
Price       541910.000     4.611   96.760 -11062.060     1.250     2.080   4.130     38970.000  
Customer ID 406830.000 15287.684 1713.603  12346.000 13953.000 15152.000   16791.000 18287.000            
"""

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

missing_values_analysis(df) # Out: ---

df.Description.nunique() # Out: 3896

df.Description.value_counts().head()

""""
WHITE HANGING HEART T-LIGHT HOLDER    2070
REGENCY CAKESTAND 3 TIER              1905
JUMBO BAG RED RETROSPOT               1662
ASSORTED COLOUR BIRD ORNAMENT         1418
PARTY BUNTING                         1416
"""

# Number of most ordered products :
df.groupby("Description").agg({"Quantity": "sum"}).head()
""""
                                Quantity
Description                             
 4 PURPLE FLOCK DINNER CANDLES       140
 50'S CHRISTMAS GIFT BAG LARGE      1883
 DOLLY GIRL BEAKER                  2391
 I LOVE LONDON MINI BACKPACK         360
 I LOVE LONDON MINI RUCKSACK           1

"""

# Number of most ordered items (Sorted from most to least!):
df.groupby("Description").agg({"Quantity": "sum"}).sort_values(by = "Quantity", ascending = False).head()

""""
                                    Quantity
Description                                 
WORLD WAR 2 GLIDERS ASSTD DESIGNS      53215
JUMBO BAG RED RETROSPOT                45066
ASSORTED COLOUR BIRD ORNAMENT          35314
WHITE HANGING HEART T-LIGHT HOLDER     34147
PACK OF 72 RETROSPOT CAKE CASES        33409
"""

# "C" in invoices stands for Canceled transactions.
# Removing cancellations from the dataset!
df = df[~df["Invoice"].str.contains("C", na=False)]


# As observed below, the descriptive statistics of the data set, the negative values were blown away.
df.describe().T
""""
                 count      mean      std       min       25%       50%     75%       max   
Quantity    397925.000    13.022  180.420     1.000     2.000     6.000     12.000    80995.000 
Price       397925.000     3.116   22.097     0.000     1.250     1.950     3.750     8142.750
Customer ID 397925.000 15294.309 1713.173 12346.000 13969.000 15159.000     16795.000 18287.000 
"""

# Creating a variable called totalprice that expresses the total earnings per invoice!
# Price * Quantity  = Total Price
df['TotalPrice'] = df["Price"] * df["Quantity"]

# 2.) Calculating RFM Metrics
# -------------------------------------------------------------------------------------------------------------------

""""
We could add 2 more or more of the day we analyzed and make calculations. 
From this, the recency values of the customers will be calculated according to the difference transaction.
"""

df["InvoiceDate"].max() # Out : Timestamp('2011-12-09 12:50:00')

today_date = dt.datetime(2011, 12, 11)

type(today_date) # out: datetime.datetime

# InvoiceDate = recency
# Invoice = frequency
# TotalPrice = monetary
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda recency: (today_date - recency.max()).days,
                                     'Invoice': lambda frequency: frequency.nunique(),
                                     'TotalPrice': lambda monetary: monetary.sum()})

rfm.head()

""""
             InvoiceDate  Invoice  TotalPrice
Customer ID                                  
12346.000            326        1   77183.600
12347.000              3        7    4310.000
12348.000             76        4    1797.240
12349.000             19        1    1757.550
12350.000            311        1     334.400
"""

rfm.columns = ['RECENCY', 'FREQUENCY', 'MONETARY']

rfm.head()

""""
             RECENCY  FREQUENCY  MONETARY
Customer ID                              
12346.000        326          1 77183.600
12347.000          3          7  4310.000
12348.000         76          4  1797.240
12349.000         19          1  1757.550
12350.000        311          1   334.400
"""

rfm.describe().T
""""
             count     mean      std   min     25%     50%      75%        max
RECENCY   4339.000   93.041  100.008 1.000  18.000  51.000  142.500    374.000
FREQUENCY 4339.000    4.272    7.705 1.000   1.000   2.000    5.000    210.000
MONETARY  4339.000 2053.797 8988.248 0.000 307.245 674.450 1661.640 280206.020
"""


# 3.) Generating RFM Scores and Converting to a Single Variable
# -------------------------------------------------------------------------------------------------------------------

# RECENCY, FREQUENCY , MONETARY -> converting values to scores

rfm["RECENCY_SCORE"] = pd.qcut(rfm['RECENCY'], 5, labels=[5, 4, 3, 2, 1])

rfm["FREQUENCY_SCORE"] = pd.qcut(rfm['FREQUENCY'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["MONETARY_SCORE"] = pd.qcut(rfm['MONETARY'], 5, labels=[1, 2, 3, 4, 5])

# create RFM score
rfm["RF_SCORE"] = (rfm['RECENCY_SCORE'].astype(str) +
                    rfm['FREQUENCY_SCORE'].astype(str))

""""
X-axis = Innovation
Y axis = Frequency

!   !   !   !
Attention: Within the scope of CRM Analytics studies, 
frequency and transaction are more important in customer relations with us. 
Therefore, the above process only includes the Novelty and Frequency Metrics. 
We can sell more to a customer who has already interacted with us. 
However, we can say that it would not make sense or would make less sense to comment on the Monetary value of,
a customer who does not have a frequency or interaction with us. 
In summary, here, a classification will be made over two dimensions of Novelty and Frequency.
!   !   !   !
"""

rfm.head(3)

""""
             RECENCY  FREQUENCY  MONETARY RECENCY_SCORE FREQUENCY_SCORE    MONETARY_SCORE   RFM_SCORE  
Customer ID                                                                         
12346.000        326          1 77183.600             1               1          5             11   
12347.000          3          7  4310.000             5               5          5             55
12348.000         76          4  1797.240             2               4          4             24
"""


# 4.) Segment Definition of RFM Score
# -------------------------------------------------------------------------------------------------------------------

# Regular Expressions(Regex)
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['SEGMENT'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head(3)

""""
             RECENCY  FREQUENCY  MONETARY RECENCY_SCORE FREQUENCY_SCORE  MONETARY_SCORE   RF_SCORE      SEGMENT
Customer ID                                                               
12346.000        326          1 77183.600             1               1         5            11         hibernating
12347.000          3          7  4310.000             5               5         5            55         champions
12348.000         76          4  1797.240             2               4         4            24         at_Risk
"""

# 5.) Action Time!
# -------------------------------------------------------------------------------------------------------------------


# BY SEGMENTS - Observing RECENCY, FREQUENCY , MONETARY Values
rfm[["SEGMENT", "RECENCY", "FREQUENCY", "MONETARY"]].groupby("SEGMENT").agg(["mean", "count"])

""""
                    RECENCY       FREQUENCY       MONETARY      
                       mean count      mean count     mean count
SEGMENT                                                         
about_to_sleep       53.312   352     1.162   352  471.994   352
at_Risk             153.786   593     2.879   593 1084.535   593
cant_loose          132.968    63     8.381    63 2796.156    63
champions             6.362   633    12.417   633 6857.964   633
hibernating         217.605  1071     1.102  1071  488.643  1071
loyal_customers      33.608   819     6.480   819 2864.248   819
need_attention       52.428   187     2.326   187  897.628   187
new_customers         7.429    42     1.000    42  388.213    42
potential_loyalists  17.399   484     2.010   484 1041.222   484
promising            23.421    95     1.000    95  290.913    95
"""
# -------------------------------------
""""
1.) cant_loose

Total number of our loyal customers: 63 
Average earnings : 2796.156

2.) champions 

Our total number of champion customers: 633
Average earnings : 6857.964

3.) at_Risk

Number of customers at risk:  593
Average earnings : 1084.535

"""

# Capturing the customerIDs of our customers at risk and printing
# them in an excel file (we can make special campaigns for them... etc):

new_df = pd.DataFrame()
new_df["at_risk_customer_id"] = rfm[rfm["SEGMENT"] == "at_Risk"].index

new_df["at_risk_customer_id"] = new_df["at_risk_customer_id"].astype(int)

# Result !
new_df.to_excel("at_risk_customer_id.xlsx")