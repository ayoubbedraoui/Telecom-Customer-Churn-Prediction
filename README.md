# Telecom-Customer-Churn-Prediction
The Telco customer churn data contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. It indicates which customers have left, stayed, or signed up for their service. Multiple important demographics are included for each customer, as well as a Satisfaction Score, Churn Score, and Customer Lifetime Value (CLTV) index.


# <span style="font-family:serif; font-size:28px;"> Content</span>

1. [Introduction](#1)
    * [What is Customer Churn?](#2)
    * [How can customer churn be reducded?](#3)
    * [Objectives](#4)
2. [Loading libraries and data](#5)
3. [Undertanding the data](#6)
4. [Visualize missing values](#7)
5. [Data Manipulation](#8)
6. [Data Visualization](#9)
7. [Data Preprocessing](#10)
   * [Standardizing numeric attributes](#111)
8. [Machine Learning Model Evaluations and Predictions](#11)
   * [KNN](#101)
   * [SVC](#102)
   * [Random Forest](#103)
   * [Logistic Regression](#104)
   * [Decision Tree Classifier](#105)
   * [AdaBoost Classifier](#106)
   * [Gradient Boosting Classifier](#107)
   * [Voting Classifier](#108)



<a id = "1" ></a>
# <span style="font-family:serif; font-size:28px;"> 1. Introduction</span>
<a id = "introduction" ></a>


<a id = "2" ></a>
#### <b>What is Customer Churn?</b>
<span style="font-size:16px;">  Customer churn is defined as when customers or subscribers discontinue doing business with a firm or service. </span>

<span style="font-size:16px;"> Customers in the telecom industry can choose from a variety of service providers and actively switch from one to the next. The telecommunications business has an annual churn rate of 15-25 percent in this highly competitive market.</span>

<span style="font-size:16px;"> Individualized customer retention is tough because most firms have a large number of customers and can't afford to devote much time to each of them. The costs would be too great, outweighing the additional revenue. However, if a corporation could forecast which customers are likely to leave ahead of time, it could focus customer retention efforts only on these "high risk" clients. The ultimate goal is to expand its coverage area and retrieve more
customers loyalty. The core to succeed in this market lies in the customer itself. 
</span>

<span style="font-size:16px;"> Customer churn is a critical metric because it is much less expensive to retain existing customers than it is to acquire new customers.</span>

<a id="churn"></a>
<a id = "3" ></a>

<span style="font-size:16px;"><b>To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.</b></span> 

<span style="font-size:16px;"> To detect early signs of potential churn, one must first develop a holistic view of the customers and their interactions across numerous channels, including store/branch visits, product purchase histories, customer service calls, Web-based transactions, and social media interactions, to mention a few. </span> 

<span style="font-size:16px;">As a result, by addressing churn, these businesses may not only preserve their market position, but also grow and thrive. More customers they have in their network, the lower the cost of initiation and the larger the profit. As a result, the company's key focus for success is reducing client attrition and implementing effective retention strategy. </span> 
<a id="reduce"></a>

<a id = "4" ></a>
#### <b> Objectives</b>
I will explore the data and try to answer some questions like:
* What's the % of Churn Customers and customers that keep in with the active services?
* Is there any patterns in Churn Customers based on the gender?
* Is there any patterns/preference in Churn Customers based on the type of service provided?
* What's the most profitable service types?
* Which features and services are most profitable?
* Many more questions that will arise during the analysis
<a id="objective"></a>

<a id = "5" ></a>
# <span style="font-family:serif; font-size:28px;"> 2. Loading libraries and data</span>
<a id="loading"></a>

### display the data

<a id = "6" ></a>
# <span style="font-family:serif; font-size:28px;"> 3. Undertanding the data</span>
<a id = "Undertanding the data" ></a>

**The data set includes information about:**
* **Customers who left within the last month** – the column is called Churn

* **Services that each customer has signed up for** – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies

* **Customer account information** - how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges

* **Demographic info about customers** – gender, age range, and if they have partners and dependents

<a id = "7" ></a>
# <span style="font-family:serif; font-size:28px;"> 4. Visualize missing values </span>
<a id = "missingvalue" ></a>


<a id = "8" ></a>
# <span style="font-family:serif; font-size:28px;"> 5. Data Manipulation </span>
<a id = "8" ></a>


<a id = "9" ></a>
# <span style="font-family:serif; font-size:28px;"> 6. Data Visualization </span>
<a id = "datavisualization" ></a>


<a id = "10" ></a>
# <span style="font-family:serif; font-size:28px;"> 7. Data Preprocessing</span>
<a id = "datapreprocessing" ></a>

<a id = "1111" ></a>
#### **Splitting the data into train and test sets**
<a id = "Split" ></a>

Since the numerical features are distributed over different value ranges, I will use standard scalar to scale them down to the same range.
<a id = "111" ></a>
#### **Standardizing numeric attributes**
<a id = "Standardizing" ></a>
**NOTES**

**Loading Libraries and Data:**
In this step, we import the necessary Python libraries for data analysis and visualization, such as pandas, numpy, and matplotlib. We then load the dataset into our environment using pandas' `read_csv()` function. This step ensures that we have access to the tools and data needed for our analysis.

**Understanding the Data:**
Here, we explore the structure and content of the dataset. We examine the dimensions of the data (number of rows and columns), view the first few rows to understand the variables, and check data types. This initial exploration helps us gain insights into the dataset and understand its underlying characteristics.

**Visualize Missing Values:**
Visualizing missing values allows us to identify any patterns or trends in the missing data. We may use seaborn's heatmap or matplotlib's bar plots to visualize the distribution of missing values across different variables. Understanding missingness is crucial for deciding how to handle missing data during data preprocessing.

**Data Manipulation:**
Data manipulation involves cleaning and transforming the dataset to prepare it for analysis. This may include tasks such as removing duplicates, handling missing values, converting data types, and creating new variables. Data manipulation ensures that the dataset is in a suitable format for further analysis.

**Data Visualization:**
Data visualization is an essential step for exploring relationships and patterns within the data. We create visualizations such as histograms, scatter plots, and box plots to gain insights into the distribution of variables, identify outliers, and explore relationships between variables. Visualization helps us better understand the underlying structure of the data.

**Data Preprocessing:**
Data preprocessing involves preparing the data for modeling by addressing issues such as feature scaling, encoding categorical variables, and splitting the dataset into training and testing sets. Preprocessing ensures that the data is in a format suitable for machine learning algorithms and improves the accuracy of our models.

**Standardizing Numeric Attributes:**
Standardizing numeric attributes involves scaling numerical variables to have a mean of 0 and a standard deviation of 1. This step ensures that all variables are on the same scale, preventing features with larger magnitudes from dominating the model. Standardization is particularly important for algorithms that rely on distance metrics, such as k-nearest neighbors and support vector machines.

