# LOAD PACKAGES & DATA
import pandas as pd
import numpy as np
from numpy import mean, std, NaN
import scipy.stats as stats

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot

callcenter_file = r"call_centre_data_location.csv"

cc = pd.read_csv(callcenter_file)

# replacing missing values with 0
cc['Speed of answer in seconds'] = cc['Speed of answer in seconds'].fillna(0)
cc['AvgTalkDuration'] = cc['AvgTalkDuration'].fillna('00:00:00')
cc['Satisfaction rating'] = cc['Satisfaction rating'].fillna('0')


cc['Date'] = pd.to_datetime(cc['Date'], errors='coerce')
cc.set_index('Date',inplace=True)

cc['Mon'] = cc.index.month
cc['Day'] = cc.index.day
cc['Hour'] = cc['Time'].str[:2].astype(float)
# obtain name of month from the string
cc['Month']=cc.index.strftime('%B')



cc['AvgTalkDurationSeconds'] = cc['AvgTalkDuration'].str.split(':').apply(lambda x: int(x[0]) * 3600 + int(x[1])*60 + int(x[2])   )

#print(cc.dtypes)
print(cc.columns)
cc.head(2)

### BRIEF INSPECTION
#cc.nunique()
cc.describe(include='object')

cc.describe()

cc.hist(figsize = (8,8), bins=100)
plt.show()

cc.info()

# convert to categorical
cc.Agent = cc.Agent.astype("category")
cc.Topic = cc.Topic.astype("category")
cc['Answered (Y/N)'] = cc['Answered (Y/N)'].astype("category")
cc['Resolved'] = cc['Resolved'].astype("category")
cc['Caller_District'] = cc['Caller_District'].astype("category")
cc['Satisfaction rating'] = cc['Satisfaction rating'].astype("int")
#cc.info()

# TIME SERIES
#### Bar Graph: Number of Calls per Day
df2 = cc.copy()
df2 = df2.reset_index()

january_calls = df2[df2['Month'] == 'January']
calls_per_day_january = january_calls.groupby(df2['Date'].dt.day)['Topic'].count()

# Plotting
plt.figure(figsize=(10, 2))
calls_per_day_january.plot(kind='bar', fontsize=10, color='darkblue') #color='skyblue')
plt.title('Number of Calls in January (Daily)', fontsize = 10)
plt.xlabel('Day of January', fontsize = 10)
plt.ylabel('Number of Calls', fontsize = 10)
plt.title('Number of Calls Per Day', fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=10)
#plt.tight_layout()
plt.show()

#### Line Graph: Average Rating per Day
daily_average_rating = cc[['Satisfaction rating']].resample('D').mean()
daily_average_rating.index

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(daily_average_rating.index, daily_average_rating['Satisfaction rating'], marker='o', linestyle='-', color='b')
plt.title('Average Rating per Day', fontsize=10)
plt.xlabel('Date', fontsize=10)
plt.ylabel('Average Rating', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()

#### Line Graph: Average Rating per Month
# Resample to monthly frequency; determine mean @Monthly (M) or Start of the month (MS)
monthly_average_rating = cc[['Satisfaction rating']].resample('MS').mean()
print(monthly_average_rating)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(monthly_average_rating.index, monthly_average_rating['Satisfaction rating'], marker='o', linestyle='-', color='b')
plt.title('Average Rating per Month', fontsize=10)
plt.xlabel('Month', fontsize=10)
plt.ylabel('Average Rating', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()


# --------------------------------METRICS -----------------------
# FIRST RESPONSE TIME (FRT)
TotalCustomerWaitingTime = cc['Speed of answer in seconds'].sum()
TotalNoQueries = cc.shape[0]

FRT = TotalCustomerWaitingTime / TotalNoQueries
FRT

#### FRT per Agent
agg_agent = cc.groupby([ "Agent"]).agg( CustomerWaitingTimeSum = ('Speed of answer in seconds', "sum"),
                                        AvgTalkDurationSeconds = ('AvgTalkDurationSeconds','mean'),
                                        NumberCallsReceived = ("Unnamed: 0", "count")
                                        )

agg_agent['FRT_per_Agent'] = np.round( (agg_agent['CustomerWaitingTimeSum'] / agg_agent['NumberCallsReceived']), 2)
agg_agent.sort_values(by = 'FRT_per_Agent', ascending=True)

# Bar Graph
agg_agent = agg_agent.reset_index()
agg_agent[['Agent','FRT_per_Agent']].sort_values(by='FRT_per_Agent', ascending=False).plot(kind='barh', x='Agent', fontsize=(10), color='darkblue' , stacked=True, figsize=(8,2), legend=False, rot=0)
plt.title("FRT per Agent", fontsize = 10)
plt.ylabel("Agent", fontsize = 10)
plt.title('FRT per Agent', fontsize=10)
plt.xticks(rotation=0, ha='right', fontsize=10)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.show()

# FIRST CALL RESOLUTION (FCR)
# FTR - Agent --------------- note, both unresolved and unanswered calls constitute UNRESOLVED

callAgentResolution = pd.crosstab(index = cc.Agent, columns = cc.Resolved,
                                 margins = True, margins_name= "Total")


callAgentResolution.drop('Total', axis=0, inplace=True)
callAgentResolution = callAgentResolution.reset_index() #

callAgentResolution['FCR_Agent'] = ( callAgentResolution['Y']/callAgentResolution['Total'] )*100
callAgentResolution

#BAR GRAPH
callAgentResolution[['Agent','FCR_Agent']].sort_values(by='FCR_Agent', ascending=True).plot(kind='barh', x='Agent', color='darkblue', fontsize=(10), stacked=True, figsize=(8,2), legend=False, rot=0)
plt.title("FCR per Agent", fontsize = 10)
plt.ylabel("Agent", fontsize = 10)
plt.xticks(rotation=0, ha='right', fontsize=10)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.show()

# Non - Resolution
# NON-RESOLUTION

# unanswered calls
nonresolved_calls = cc.loc[cc['Resolved'] == 'N']
nonresolved_calls = nonresolved_calls.reset_index()

nonresolved = nonresolved_calls.groupby(["Topic"]).agg(NumberIncomingCalls=("Unnamed: 0", "count")
                                                       )
nonresolved = nonresolved.rename(columns={'NumberIncomingCalls': 'Non_resolved'})

nonresolved[['Non_resolved']].sort_values(by='Non_resolved', ascending=True).plot(kind='barh', stacked=True,
                                                                                  figsize=(8, 2), color='darkblue',
# plot                                                                                fontsize=(10), legend=False, rot=0)
plt.title("Unresolved Calls per Topic", fontsize=10)
plt.ylabel("Topic", fontsize=10)
plt.xticks(rotation=0, ha='right', fontsize=10)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.show()


# Resolution

# unanswered calls
resolved_calls = cc.loc[cc['Resolved'] == 'Y']
resolved_calls = resolved_calls.reset_index()

resolved = resolved_calls.groupby(["Topic"]).agg(NumberIncomingCalls=("Unnamed: 0", "count"),
                                                 Avg_Rating=("Satisfaction rating", "mean")
                                                 )
resolved = resolved.rename(columns={'NumberIncomingCalls': 'Resolved'})

resolved


#plot
resolved[['Resolved']].sort_values(by='Resolved',ascending=True).plot(kind='barh',stacked=True,figsize=(8,2),color='darkblue', fontsize=(10), legend=False, rot=0)
plt.title("Resolved Calls per Topic", fontsize = 10)
plt.ylabel("Topic", fontsize = 10)
plt.xticks(rotation=0, ha='right', fontsize=10)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.show()


# AVERAGE CALL ABANDONMENT RATE
# unanswered calls
unanswered_calls = cc.loc[cc['Answered (Y/N)'] == 'N']
unanswered_calls = unanswered_calls.reset_index()

unanswered = unanswered_calls.groupby(["Agent"]).agg(NumberIncomingCalls=("Unnamed: 0", "count")
                                                     )
unanswered = unanswered.rename(columns={'NumberIncomingCalls': 'UnansweredCalls'})

print(unanswered.shape)
# print(unanswered)

unanswered[['UnansweredCalls']].sort_values(by='UnansweredCalls', ascending=True).plot(kind='barh', stacked=True,
                                                                                       color='darkblue', figsize=(8, 2),
                                                                                       fontsize=(10), legend=False,
                                                                                       rot=0)
plt.title("Abandoned Calls per Agent", fontsize=10)
plt.ylabel("Agent", fontsize=10)
plt.xticks(rotation=0, ha='right', fontsize=10)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.show()

#### Calls not answered per day by Agent 1: Diane
calls_not_answered_agent1['Date'] = pd.to_datetime(calls_not_answered_agent1['Date'])
calls_not_answered_agent1.set_index(calls_not_answered_agent1['Date'])


# Plotting
plt.figure(figsize=(10, 3))
plt.plot(calls_not_answered_agent1.index, calls_not_answered_agent1['CallsNotAnsweredAgent1'], marker='o', linestyle='-', color='b')
plt.title('Calls not Answered by Agent:Diane', fontsize=10)
plt.xlabel('Date', fontsize=10)
plt.ylabel('Abandoned Calls', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()

#### Unanswered calls per month (all agents)
calls_not_answered_by_agent_per_month = cc00[cc00['Answered (Y/N)'] == 'N'].groupby(['Agent', cc00['Date'].dt.to_period("M")]).size().reset_index(name='CallsNotAnsweredByAgentPerMonth')

#calls_not_answered_agent1
calls_not_answered_by_agent_per_month[:5]

pd.pivot_table(calls_not_answered_by_agent_per_month, index = 'Date', columns = 'Agent',
               values = 'CallsNotAnsweredByAgentPerMonth')

# bar chart - calls not answered per month
pd.pivot_table(calls_not_answered_by_agent_per_month, index = 'Date', columns = 'Agent',
               values = 'CallsNotAnsweredByAgentPerMonth').plot(kind = 'bar')


#### Bar Graph - Abandoned Topics - Count
unanswered_topics = unanswered_calls.groupby([ "Topic"]).agg(AbandonedTopics = ("Unnamed: 0", "count") )
unanswered_topics[['AbandonedTopics']].sort_values(by='AbandonedTopics', ascending=True).plot(kind='barh', stacked=True, figsize=(8,2),color='darkblue', fontsize=10, legend=False, rot=0)
plt.title("Abandoned Topics", fontsize = 10)
plt.ylabel("Topic", fontsize = 10)
plt.xticks(rotation=0, ha='right', fontsize=10)
plt.yticks(rotation=0, ha='right', fontsize=10)
plt.show()

# ABANDONMENT RATE

# total number of calls received for the day
total_calls = cc.groupby('Date').size().reset_index(name='TotalCalls')

#number of calls answered for the day
calls_answered = cc[cc['Answered (Y/N)'] == 'Y'].groupby('Date').size().reset_index(name='CallsAnswered')
calls_unanswered = cc[cc['Answered (Y/N)'] == 'N'].groupby('Date').size().reset_index(name='CallsNotAnswered')


result_df = total_calls.merge(calls_answered, on='Date', how='left').merge(calls_unanswered, on='Date', how='left')

print(calls_answered.shape, calls_unanswered.shape, result_df.shape)

result_df['Abandonment_Rate'] = result_df['CallsNotAnswered']/result_df['TotalCalls']


result_df[:5]

# AHT
Overall_Avg_Handle_Time = (cc['Speed of answer in seconds'].sum() + cc['AvgTalkDurationSeconds'].sum()) / len(cc)
print(f"Overall AHT = {Overall_Avg_Handle_Time} seconds\n")
print(f"Overall AHT = {Overall_Avg_Handle_Time/60} minutes\n")


agg_topic = cc.groupby([ "Topic"]).agg( CustomerWaitingTimeSum = ('Speed of answer in seconds', "sum"),
                                        AvgTalkDurationSeconds = ('AvgTalkDurationSeconds','sum'),
                                        NumberCallsReceived = ("Unnamed: 0", "count")
                                        )

agg_topic['Avg_Handle_Time_secs'] =  (agg_topic['CustomerWaitingTimeSum']+ agg_topic['AvgTalkDurationSeconds']) / agg_topic['NumberCallsReceived']

agg_topic['Avg_Handle_Time_mins0'] =   (agg_topic['CustomerWaitingTimeSum']+ agg_topic['AvgTalkDurationSeconds']) /60
agg_topic['Avg_Handle_Time_mins'] = agg_topic['Avg_Handle_Time_mins0']/ agg_topic['NumberCallsReceived']

agg_topic.sort_values(by = 'Avg_Handle_Time_secs', ascending=True)


# -----------------------------HYPOTHESIS TESTING----------------
cc[['Speed of answer in seconds','AvgTalkDurationSeconds','Satisfaction rating']].corr(method = 'spearman')

stats.spearmanr(cc['Speed of answer in seconds'], cc['Satisfaction rating'], nan_policy='omit')

r, p_value = stats.spearmanr(cc['Speed of answer in seconds'], cc['Satisfaction rating'],
                             nan_policy = "omit")
r # correlation co-efficient

p_value

"""" 
<b>p-value<b> > 0.05
hence reject hypothesis of no relationship. 
Conclude there is significant relationship between time customer spends waiting to 
have phone answered and rating.
"""
# -------------------end----------------------