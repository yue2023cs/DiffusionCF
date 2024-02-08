# ============================================================================
# Paper:
# Author(s):
# Create Time: 01/23/2023
# ============================================================================
from pkg_manager import *
from para_manager import *
import statistics

originalDataAddr = "D:\\researchCodes\\interpretableML\\data\\Dodgers.csv"
processedDataAddr = "D:\\researchCodes\\interpretableML\\data\\dodgersLoopSensor.csv"
eventAddr = "D:\\researchCodes\\interpretableML\\data\\Dodgers_events.csv"

dfOriginalData = pd.read_csv(originalDataAddr, encoding='ISO-8859-1', header = None)
dfProcessedData = pd.read_csv(processedDataAddr, encoding='ISO-8859-1')
dfEvent = pd.read_csv(eventAddr, encoding='ISO-8859-1', header = None)

datesToDelete = dfOriginalData[dfOriginalData[1] == -1][0].str[:10]
dfProcessedData = dfProcessedData[~dfProcessedData['Date'].str[:10].isin(datesToDelete)]

dfProcessedData['Date'] = pd.to_datetime(dfProcessedData['Date'])
dfProcessedData['Date_only'] = dfProcessedData['Date'].dt.date
grouped = dfProcessedData.groupby('Date_only')

dfEvent[0] = pd.to_datetime(dfEvent[0], format='%m/%d/%y')

weekday = 1
# game = 0
flowCollectorWithGame = []
flowCollectorWithoutGame = []
for date, group in grouped:
    if weekday == 0 and (group['Date'].iloc[0].weekday() == 5 or group['Date'].iloc[0].weekday() == 6):
        # with game
        if dfEvent[0].isin([group['Date'].iloc[0].strftime('%Y-%m-%d')]).any():
            if len(list(group['Flow'])) == 8:
                flowCollectorWithGame.append(list(group['Flow']))

        # without game
        elif not dfEvent[0].isin([group['Date'].iloc[0].strftime('%Y-%m-%d')]).any():
            if len(list(group['Flow'])) == 8:
                flowCollectorWithoutGame.append(list(group['Flow']))

    elif weekday == 1 and (group['Date'].iloc[0].weekday() != 5 or group['Date'].iloc[0].weekday() != 6):
        if dfEvent[0].isin([group['Date'].iloc[0].strftime('%Y-%m-%d')]).any():
            if len(list(group['Flow'])) == 8:
                flowCollectorWithGame.append(list(group['Flow']))
        elif not dfEvent[0].isin([group['Date'].iloc[0].strftime('%Y-%m-%d')]).any():
            if len(list(group['Flow'])) == 8:
                flowCollectorWithoutGame.append(list(group['Flow']))

avgWithGame = [sum(sublist) / len(sublist) for sublist in zip(*flowCollectorWithGame)]
avgWithGame.append(avgWithGame[0])
avgWithoutGame = [sum(sublist) / len(sublist) for sublist in zip(*flowCollectorWithoutGame)]
avgWithoutGame.append(avgWithoutGame[0])

avgWithGame = avgWithGame[1:]
avgWithoutGame = avgWithoutGame[1:]

plt.figure(figsize=(9, 8))
plt.plot(['03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '00:00'], avgWithGame, label = 'game days', linewidth=3)
plt.plot(['03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '00:00'], avgWithoutGame, label = 'no game days', linewidth=3)
# plt.errorbar(['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '00:00+'], avgWithoutGame,yerr=stdWithoutGame, fmt='o-')
#plt.xlabel('Time', fontsize=15)
plt.ylabel('Average Flow', fontsize=20)
plt.xticks(fontsize=18, rotation = 45)
plt.yticks(fontsize=18)
plt.legend(fontsize=20)
plt.title('Weekends', fontsize = 30)
plt.show()


#
# avg = [sum(sublist) / len(sublist) for sublist in zip(*flowCollector)]
# avg.append(avg[0])
# std = [statistics.stdev(sublist) for sublist in zip(*flowCollector)]
# std.append(std[0])
#
# plt.figure(figsize=(7, 5))
# plt.plot(['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '00:00+'], avg)
# plt.errorbar(['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '00:00+'], avg, yerr=std, fmt='o-')
# plt.xlabel('Time', fontsize=15)
# plt.ylabel('Average Flow', fontsize=15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.show()
