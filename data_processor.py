# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/10/2023
# ============================================================================

from pkg_manager import *
from para_manager import *

class DataProcessor:
    def __init__(self):
        self.allTimeSeries = []
        self.allDfByMonth = []

        self.scalers = {}
        self.allDataPredictorsYMS = None
        self.allDataForecastYMS = None
        self.trainPredictorsYMS = None
        self.trainForecastYMS = None
        self.valPredictorsYMS = None
        self.valForecastYMS = None
        self.testPredictorsYMS = None
        self.testForecastYMS = None

        self.allData = []
        self.allTrainData = []
        self.allValData = []
        self.allTestData = []

        self.allDataPredictors = None
        self.allDataForecast = None
        self.allDataTargets = None
        self.trainPredictors = None
        self.trainForecast = None
        self.trainTargets = None
        self.valPredictors = None
        self.valForecast = None
        self.valTargets = None
        self.testPredictors = None
        self.testForecast = None
        self.testTargets = None

        self.allDataPredictions = []
        self.allDataCfTargets = []
        self.allDataYu = []

    def standardizeByMonth(self, dfByMonth):                                                                         # 'allDfByMonth' is a list comprosed of several dataframes
        scaler = StandardScaler()
        scalerDict = {}
        for month in dfByMonth['Month'].unique():
            month_data = dfByMonth[dfByMonth['Month'] == month][X]
            scaled = scaler.fit_transform(np.array(month_data).reshape(-1, 1))
            scaled = scaled.reshape(-1, len(month_data)).flatten()
            dfByMonth.loc[dfByMonth['Month'] == month, X] = scaled
            dfByMonth.loc[dfByMonth['Month'] == month, X] = scaled
            scalerDict[month] = {'mean': scaler.mean_, 'scale': scaler.scale_}

        station = dfByMonth['Station'][0]
        self.scalers[station] = scalerDict

        return dfByMonth

    def readCsv(self, csvPath):
        df = pd.read_csv(csvPath)
        # df[X] = df[X].str.replace(',', '').astype(float)
        df = df[df[X] < anomaly]

        if dataset == "GSOD":
            csvFileName = csvPath.rsplit('\\', 1)
            station = csvFileName[-1].replace('.csv', '')
            df['Station'] = station
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['Year'] = df['DATE'].dt.year
            df['Month'] = df['DATE'].dt.month

            dfByMonth = df.groupby(['Station', 'Year', 'Month'])[X].mean().reset_index()
            dfByMonth[X] = dfByMonth[X].apply(lambda x: format(x, '.2f'))
            dfByMonth['YearMonth'] = dfByMonth['Year'].astype(str) + '-' + dfByMonth['Month'].astype(str).str.zfill(2)
            dfByMonth = self.standardizeByMonth(dfByMonth)

            if multiTimeSeries == 'multivariate':
                Z = list(F)
                for thisClimateNameIndex in F:
                    thisPath = climateIndexFolder + thisClimateNameIndex + ".csv"
                    thisClimateName = pd.read_csv(thisPath)
                    dfByMonth = pd.merge(dfByMonth, thisClimateName, on='YearMonth')                                        # some X might be removed since there could be no match with F by YearMonth.
                Z.append(X)
                Z.append('YearMonth')
                Z.append('Station')
                timeSeries = np.array([dfByMonth[column].to_numpy() for column in Z])
            else:
                timeSeries = dfByMonth[[X, 'YearMonth', 'Station']].to_numpy()

        elif dataset == 'Weather':
            df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], format='%Y-%m-%d %H:%M:%S.%f %z')
            df = df.sort_values(by='Formatted Date').astype(str)

            scaler = StandardScaler()
            hour_data = df[X]
            scaled = scaler.fit_transform(np.array(hour_data).reshape(-1, 1))
            scaled = scaled.reshape(-1, len(hour_data)).flatten()
            df[X] = scaled

            dfByMonth = df[['Formatted Date', 'Temperature (C)']]
            timeSeries = dfByMonth.to_numpy()

        elif dataset == 'S&P500':
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
            df = df.sort_values(by='Date').astype(str)
            # df[X] = df[X].replace(',', '').astype(float)

            scaler = StandardScaler()
            hour_data = df[X]
            scaled = scaler.fit_transform(np.array(hour_data).reshape(-1, 1))
            scaled = scaled.reshape(-1, len(hour_data)).flatten()
            df[X] = scaled

            dfByMonth = df[['Date', X]]
            timeSeries = dfByMonth.to_numpy()

        elif dataset == 'dodgersLoopSensor':
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M')
            df = df.sort_values(by='Date').astype(str)
            scaler = StandardScaler()
            hour_data = df[X]
            scaled = scaler.fit_transform(np.array(hour_data).reshape(-1, 1))
            scaled = scaled.reshape(-1, len(hour_data)).flatten()
            df[X] = scaled

            dfByMonth = df[['Date', X]]
            timeSeries = dfByMonth.to_numpy()

            scalerDict = {'mean': scaler.mean_, 'scale': scaler.scale_}
            self.scalers = scalerDict                                                                                   # {'mean': array([759.36034732]), 'scale': array([403.94069766])}

        return dfByMonth, timeSeries

    def readFolder(self):
        if dataset == "GSOD":
            readCsv = []
            for csvPath in os.listdir(rootFolder):
                readCsv.append(csvPath)
                csvPath = rootFolder + '\\' +  csvPath
                thisDfByMonth, thisTimeSeries = self.readCsv(csvPath)

                self.allDfByMonth.append(thisDfByMonth)
                self.allTimeSeries.append(thisTimeSeries)

        elif dataset == "Weather" or "S&P500" or "dodgersLoopSensor":
            csvPath = rootFolder
            thisDfByMonth, thisTimeSeries = self.readCsv(csvPath)
            self.allDfByMonth.append(thisDfByMonth)
            self.allTimeSeries.append(thisTimeSeries)


    def splitTrainValTest(self, totalTimesteps, trainTestRatio):
        print("splitting train/val/test data...")
        if splitMethod == "byDistribution":
            if multiTimeSeries == 'multivariate':
                for thisTimeSeries in self.allTimeSeries:
                    thisData = []
                    for i in range(0, thisTimeSeries[-1].shape[0]):
                        if i + totalTimesteps <= thisTimeSeries[-1].shape[0]:
                            thisData.append(thisTimeSeries[:, i:i + totalTimesteps])
                    self.allData.append(thisData)
                self.allData = [item for sublist in self.allData for item in sublist]

                self.allTrainData, self.allTestData = train_test_split(self.allData, test_size=1 - trainTestRatio, random_state=randomState)
                random.shuffle(self.allTrainData)
                self.allTrainData, self.allValData = train_test_split(self.allTrainData, test_size=1 - trainTestRatio, random_state=randomState)

            elif multiTimeSeries == 'univariate':
                for thisTimeSeries in self.allTimeSeries:
                    thisData = []
                    for i in range(0, thisTimeSeries.shape[0]):
                        if i + totalTimesteps <= thisTimeSeries.shape[0]:
                            thisData.append(thisTimeSeries[i:i + totalTimesteps])
                    self.allData.append(thisData)
                self.allData = [item for sublist in self.allData for item in sublist]                                   # convert from [[np, np, ...]] to [np, np, ...]

                self.allTrainData, self.allTestData = train_test_split(self.allData, test_size = 1 - trainTestRatio, random_state = randomState)
                random.shuffle(self.allTrainData)
                self.allTrainData, self.allValData = train_test_split(self.allTrainData, test_size = 1 - trainTestRatio,random_state = randomState)

        elif splitMethod == "byTime":
            if multiTimeSeries == 'multivariate':
                for thisTimeSeries in self.allTimeSeries:
                    thisData = []
                    for i in range(0, thisTimeSeries[-1].shape[0]):
                        if i + totalTimesteps <= thisTimeSeries[-1].shape[0]:
                            thisData.append(thisTimeSeries[:, i:i + totalTimesteps])
                    self.allData.append(thisData)
                self.allData = [item for sublist in self.allData for item in sublist]

                for row in self.allData:
                    endData =datetime.datetime.strptime(row[-2][-1], '%Y-%m')
                    if endData < trainValSplitYear:
                        self.allTrainData.append(row)
                    elif trainValSplitYear <= endData < valTestSplitYear:
                        self.allValData.append(row)
                    else:
                        self.allTestData.append(row)

            elif multiTimeSeries == 'univariate':
                for thisTimeSeries in self.allTimeSeries:
                    thisData = []
                    for i in range(0, thisTimeSeries.shape[0]):
                        if i + totalTimesteps <= thisTimeSeries.shape[0]:
                            thisData.append(thisTimeSeries[i:i + totalTimesteps])
                    self.allData.append(thisData)
                self.allData = [item for sublist in self.allData for item in sublist]                                   # type is list[np.array]

                for row in self.allData:
                    # startDate = datetime.datetime.strptime(row[0][1], '%Y-%m')
                    if dataset == 'GSOD':
                        endData =datetime.datetime.strptime(row[-1][1], '%Y-%m')
                    elif dataset == 'Weather':
                        endData = datetime.datetime.strptime(row[-1][0], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
                    elif dataset == 'S&P500':
                        endData = datetime.datetime.strptime(row[-1][0], '%Y-%m-%d').replace(tzinfo=None)
                    elif dataset == 'dodgersLoopSensor':
                        endData = datetime.datetime.strptime(row[-1][0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=None)

                    if endData < trainValSplitYear:
                        self.allTrainData.append(row)
                    elif trainValSplitYear <= endData and  endData < valTestSplitYear:
                        self.allValData.append(row)
                    elif endData >= valTestSplitYear:
                        self.allTestData.append(row)

        print("splitting train/val/test data completed")
        print('total:', len(self.allData), '| train:', len(self.allTrainData), '| val:', len(self.allValData), '| test:', len(self.allTestData))

    def createPredictors(self, dataset, predictorTimesteps):
        predictors = []
        for i in range(len(dataset)):
            if multiTimeSeries == 'multivariate':
                thisPredictor = []
                for j in range(len(dataset[i])):
                    thisPredictor.append(dataset[i][j][0:predictorTimesteps])
            else:
                thisPredictor = dataset[i][0:predictorTimesteps]
            predictors.append(thisPredictor)

        return np.array(predictors)

    def createForecast(self, dataset, predictorTimesteps):
        forecast = []
        for i in range(len(dataset)):
            if multiTimeSeries == 'multivariate':
                thisForecast = []
                for j in range(len(dataset[i])):
                    thisForecast.append(dataset[i][j][predictorTimesteps:])
            else:
                thisForecast = dataset[i][predictorTimesteps:]
            forecast.append(thisForecast)

        return np.array(forecast)

    def createTargets(self, data, predictorTimesteps):
        targets = []
        for i in range(len(data)):
            if multiTimeSeries == 'univariate':
                if dataset == 'GSOD':
                    thisTarget = np.max(data[i][:,0][predictorTimesteps:])
                elif dataset == 'Weather' or 'S&P500':
                    thisTarget = np.max(data[i][:,1][predictorTimesteps:])

            elif multiTimeSeries == 'multivariate':
                thisTarget = np.max(data[i][predictorTimesteps:])

            targets.append(thisTarget)

        return np.array(targets)

    def splitPredictorTarget(self, predictorTimesteps):
        print("splitting predictors/targets...")
        if multiTimeSeries == 'multivariate':
            self.allDataPredictors = self.createPredictors(self.allData, predictorTimesteps)
            self.allDataPredictorsYMS = self.allDataPredictors[:, -2:]
            self.allDataPredictors = self.allDataPredictors[:, :-2]
            self.allDataPredictors = self.allDataPredictors.reshape(self.allDataPredictors.shape[0], self.allDataPredictors.shape[1], self.allDataPredictors.shape[2])
            self.allDataPredictors = torch.from_numpy(self.allDataPredictors.astype(np.float32)).to(device)

            self.trainPredictors = self.createPredictors(self.allTrainData, predictorTimesteps)
            self.trainPredictorsYMS = self.trainPredictors[:, -2:]
            self.trainPredictors = self.trainPredictors[:, :-2]
            self.trainPredictors = self.trainPredictors.reshape(self.trainPredictors.shape[0], self.trainPredictors.shape[1], self.trainPredictors.shape[2])
            self.trainPredictors = torch.from_numpy(self.trainPredictors.astype(np.float32)).to(device)

            self.valPredictors = self.createPredictors(self.allValData, predictorTimesteps)
            self.valPredictorsYMS = self.valPredictors[:, -2:]
            self.valPredictors = self.valPredictors[:, :-2]
            self.valPredictors = self.valPredictors.reshape(self.valPredictors.shape[0], self.valPredictors.shape[1], self.valPredictors.shape[2])
            self.valPredictors = torch.from_numpy(self.valPredictors.astype(np.float32)).to(device)

            self.testPredictors = self.createPredictors(self.allTestData, predictorTimesteps)
            self.testPredictorsYMS = self.testPredictors[:, -2:]
            self.testPredictors = self.testPredictors[:, :-2]
            self.testPredictors = self.testPredictors.reshape(self.testPredictors.shape[0], self.testPredictors.shape[1], self.testPredictors.shape[2])
            self.testPredictors = torch.from_numpy(self.testPredictors.astype(np.float32)).to(device)

            self.allDataForecast = self.createForecast(self.allData, predictorTimesteps)
            self.allDataForecastYMS = self.allDataForecast[:, -2:]
            self.allDataForecast = self.allDataForecast[:, :-2]
            self.allDataForecast = self.allDataForecast.reshape(self.allDataForecast.shape[0], self.allDataForecast.shape[1], self.allDataForecast.shape[2])
            self.allDataForecast = torch.from_numpy(self.allDataForecast.astype(np.float32)).to(device)

            self.trainForecast = self.createForecast(self.allTrainData, predictorTimesteps)
            self.trainForecastYMS = self.trainForecast[:, -2:]
            self.trainForecast = self.trainForecast[:, :-2]
            self.trainForecast = self.trainForecast.reshape(self.trainForecast.shape[0], self.trainForecast.shape[1], self.trainForecast.shape[2])
            self.trainForecast = torch.from_numpy(self.trainForecast.astype(np.float32)).to(device)

            self.valForecast = self.createForecast(self.allValData, predictorTimesteps)
            self.valForecastYMS = self.valForecast[:, -2:]
            self.valForecast = self.valForecast[:, :-2]
            self.valForecast = self.valForecast.reshape(self.valForecast.shape[0], self.valForecast.shape[1], self.valForecast.shape[2])
            self.valForecast = torch.from_numpy(self.valForecast.astype(np.float32)).to(device)

            self.testForecast = self.createForecast(self.allTestData, predictorTimesteps)
            self.testForecastYMS = self.testForecast[:, -2:]
            self.testForecast = self.testForecast[:, :-2]
            self.testForecast = self.testForecast.reshape(self.testForecast.shape[0], self.testForecast.shape[1], self.testForecast.shape[2])
            self.testForecast = torch.from_numpy(self.testForecast.astype(np.float32)).to(device)

            self.allDataTargets = self.createTargets([arr[-3] for arr in self.allData], predictorTimesteps)
            self.allDataTargets = self.allDataTargets.reshape(-1, 1)
            self.allDataTargets = torch.from_numpy(self.allDataTargets.astype(np.float32)).to(device)

            self.trainTargets = self.createTargets([arr[-3] for arr in self.allTrainData], predictorTimesteps)
            self.trainTargets = self.trainTargets.reshape(-1, 1)
            self.trainTargets = torch.from_numpy(self.trainTargets.astype(np.float32)).to(device)

            self.valTargets = self.createTargets([arr[-3] for arr in self.allValData], predictorTimesteps)
            self.valTargets = self.valTargets.reshape(-1, 1)
            self.valTargets = torch.from_numpy(self.valTargets.astype(np.float32)).to(device)

            self.testTargets = self.createTargets([arr[-3] for arr in self.allTestData], predictorTimesteps)
            self.testTargets = self.testTargets.reshape(-1, 1)
            self.testTargets = torch.from_numpy(self.testTargets.astype(np.float32)).to(device)

        elif multiTimeSeries == 'univariate':
            if dataset == 'GSOD':
                self.allDataPredictors = self.createPredictors(self.allData, predictorTimesteps)
                self.allDataPredictorsYMS = self.allDataPredictors[:, :, -2:]
                self.allDataPredictors = self.allDataPredictors[:, :, 0]
                self.allDataPredictors = self.allDataPredictors.reshape(self.allDataPredictors.shape[0],  self.allDataPredictors.shape[1], 1)
                self.allDataPredictors = torch.from_numpy(self.allDataPredictors.astype(np.float32)).to(device)

                self.trainPredictors = self.createPredictors(self.allTrainData, predictorTimesteps)
                self.trainPredictorsYMS = self.trainPredictors[:, :, -2:]
                self.trainPredictors = self.trainPredictors[:, :, 0]
                self.trainPredictors = self.trainPredictors.reshape(self.trainPredictors.shape[0],self.trainPredictors.shape[1], 1)
                self.trainPredictors = torch.from_numpy(self.trainPredictors.astype(np.float32)).to(device)

                self.valPredictors = self.createPredictors(self.allValData, predictorTimesteps)
                self.valPredictorsYMS = self.valPredictors[:, :, -2:]
                self.valPredictors = self.valPredictors[:, :, 0]
                self.valPredictors = self.valPredictors.reshape(self.valPredictors.shape[0], self.valPredictors.shape[1], 1)
                self.valPredictors = torch.from_numpy(self.valPredictors.astype(np.float32)).to(device)

                self.testPredictors = self.createPredictors(self.allTestData, predictorTimesteps)
                self.testPredictorsYMS = self.testPredictors[:, :, -2:]
                self.testPredictors = self.testPredictors[:, :, 0]
                self.testPredictors = self.testPredictors.reshape(self.testPredictors.shape[0], self.testPredictors.shape[1], 1)
                self.testPredictors = torch.from_numpy(self.testPredictors.astype(np.float32)).to(device)

                self.allDataForecast = self.createForecast(self.allData, predictorTimesteps)
                self.allDataForecastYMS = self.allDataForecast[:, :, -2:]
                self.allDataForecast = self.allDataForecast[:, :, 0]
                self.allDataForecast = self.allDataForecast.reshape(self.allDataForecast.shape[0], self.allDataForecast.shape[1], 1)
                self.allDataForecast = torch.from_numpy(self.allDataForecast.astype(np.float32)).to(device)

                self.trainForecast = self.createForecast(self.allTrainData, predictorTimesteps)
                self.trainForecastYMS = self.trainForecast[:, :, -2:]
                self.trainForecast = self.trainForecast[:, :, 0]
                self.trainForecast = self.trainForecast.reshape(self.trainForecast.shape[0], self.trainForecast.shape[1], 1)
                self.trainForecast = torch.from_numpy(self.trainForecast.astype(np.float32)).to(device)

                self.valForecast = self.createForecast(self.allValData, predictorTimesteps)
                self.valForecastYMS = self.valForecast[:, :, -2:]
                self.valForecast = self.valForecast[:, :, 0]
                self.valForecast = self.valForecast.reshape(self.valForecast.shape[0], self.valForecast.shape[1], 1)
                self.valForecast = torch.from_numpy(self.valForecast.astype(np.float32)).to(device)

                self.testForecast = self.createForecast(self.allTestData, predictorTimesteps)
                self.testForecastYMS = self.testForecast[:, :, -2:]
                self.testForecast = self.testForecast[:, :, 0]
                self.testForecast = self.testForecast.reshape(self.testForecast.shape[0], self.testForecast.shape[1], 1)
                self.testForecast = torch.from_numpy(self.testForecast.astype(np.float32)).to(device)

                self.allDataTargets = self.createTargets(self.allData, predictorTimesteps)
                self.allDataTargets = self.allDataTargets.reshape(-1, 1)
                self.allDataTargets = torch.from_numpy(self.allDataTargets.astype(np.float32)).to(device)

                self.trainTargets = self.createTargets(self.allTrainData, predictorTimesteps)
                self.trainTargets = self.trainTargets.reshape(-1, 1)
                self.trainTargets = torch.from_numpy(self.trainTargets.astype(np.float32)).to(device)

                self.valTargets = self.createTargets(self.allValData, predictorTimesteps)
                self.valTargets = self.valTargets.reshape(-1, 1)
                self.valTargets = torch.from_numpy(self.valTargets.astype(np.float32)).to(device)

                self.testTargets = self.createTargets(self.allTestData, predictorTimesteps)
                self.testTargets = self.testTargets.reshape(-1, 1)
                self.testTargets = torch.from_numpy(self.testTargets.astype(np.float32)).to(device)

            elif dataset == 'Weather' or 'S&P500':
                self.allDataPredictors = self.createPredictors(self.allData, predictorTimesteps)
                self.allDataPredictorsYMS = self.allDataPredictors[:, :, 0]
                self.allDataPredictors = self.allDataPredictors[:, :, 1]
                self.allDataPredictors = self.allDataPredictors.reshape(self.allDataPredictors.shape[0],self.allDataPredictors.shape[1], 1)
                self.allDataPredictors = torch.from_numpy(self.allDataPredictors.astype(np.float32)).to(device)

                self.trainPredictors = self.createPredictors(self.allTrainData, predictorTimesteps)
                self.trainPredictorsYMS = self.trainPredictors[:, :, 0]
                self.trainPredictors = self.trainPredictors[:, :, 1]
                self.trainPredictors = self.trainPredictors.reshape(self.trainPredictors.shape[0], self.trainPredictors.shape[1], 1)
                self.trainPredictors = torch.from_numpy(self.trainPredictors.astype(np.float32)).to(device)

                self.valPredictors = self.createPredictors(self.allValData, predictorTimesteps)
                self.valPredictorsYMS = self.valPredictors[:, :, 0]
                self.valPredictors = self.valPredictors[:, :, 1]
                self.valPredictors = self.valPredictors.reshape(self.valPredictors.shape[0], self.valPredictors.shape[1], 1)
                self.valPredictors = torch.from_numpy(self.valPredictors.astype(np.float32)).to(device)

                self.testPredictors = self.createPredictors(self.allTestData, predictorTimesteps)
                self.testPredictorsYMS = self.testPredictors[:, :, 0]
                self.testPredictors = self.testPredictors[:, :, 1]
                self.testPredictors = self.testPredictors.reshape(self.testPredictors.shape[0], self.testPredictors.shape[1], 1)
                self.testPredictors = torch.from_numpy(self.testPredictors.astype(np.float32)).to(device)

                self.allDataForecast = self.createForecast(self.allData, predictorTimesteps)
                self.allDataForecastYMS = self.allDataForecast[:, :, 0]
                self.allDataForecast = self.allDataForecast[:, :, 1]
                self.allDataForecast = self.allDataForecast.reshape(self.allDataForecast.shape[0], self.allDataForecast.shape[1], 1)
                self.allDataForecast = torch.from_numpy(self.allDataForecast.astype(np.float32)).to(device)

                self.trainForecast = self.createForecast(self.allTrainData, predictorTimesteps)
                self.trainForecastYMS = self.trainForecast[:, :, 0]
                self.trainForecast = self.trainForecast[:, :, 1]
                self.trainForecast = self.trainForecast.reshape(self.trainForecast.shape[0], self.trainForecast.shape[1], 1)
                self.trainForecast = torch.from_numpy(self.trainForecast.astype(np.float32)).to(device)

                self.valForecast = self.createForecast(self.allValData, predictorTimesteps)
                self.valForecastYMS = self.valForecast[:, :, 0]
                self.valForecast = self.valForecast[:, :, 1]
                self.valForecast = self.valForecast.reshape(self.valForecast.shape[0], self.valForecast.shape[1], 1)
                self.valForecast = torch.from_numpy(self.valForecast.astype(np.float32)).to(device)

                self.testForecast = self.createForecast(self.allTestData, predictorTimesteps)
                self.testForecastYMS = self.testForecast[:, :, 0]
                self.testForecast = self.testForecast[:, :, 1]
                self.testForecast = self.testForecast.reshape(self.testForecast.shape[0], self.testForecast.shape[1], 1)
                self.testForecast = torch.from_numpy(self.testForecast.astype(np.float32)).to(device)

                self.allDataTargets = self.createTargets(self.allData, predictorTimesteps)
                self.allDataTargets = self.allDataTargets.reshape(-1, 1)
                self.allDataTargets = torch.from_numpy(self.allDataTargets.astype(np.float32)).to(device)

                self.trainTargets = self.createTargets(self.allTrainData, predictorTimesteps)
                self.trainTargets = self.trainTargets.reshape(-1, 1)
                self.trainTargets = torch.from_numpy(self.trainTargets.astype(np.float32)).to(device)

                self.valTargets = self.createTargets(self.allValData, predictorTimesteps)
                self.valTargets = self.valTargets.reshape(-1, 1)
                self.valTargets = torch.from_numpy(self.valTargets.astype(np.float32)).to(device)

                self.testTargets = self.createTargets(self.allTestData, predictorTimesteps)
                self.testTargets = self.testTargets.reshape(-1, 1)
                self.testTargets = torch.from_numpy(self.testTargets.astype(np.float32)).to(device)

        print("splitting predictors/target completed")
