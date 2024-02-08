# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/10/2023
# ============================================================================
from data_processor import *
from deepExtrema import *
from counterfactual_explanation import *
from CSDI import *
from CSDI_train import *
from VAE import *
from VAE_anomaly_detection import *
from BaseNN import *
from NativeGuide import *
from counterfactual_gan import CounterfactualTimeGAN

from pkg_manager import *
from para_manager import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def checkWholeDistribution(predictors):
    checkpoint = torch.load(foldername + multiTimeSeries + '_' + str(vae_batch_size) +  '_' +str(predictorTimesteps) + '_' + X + '_vae_anomaly-detection.ckpt')
    model = VAEAnomalyTabular(predictorTimesteps, 2, 2500, 0.005, 1000).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    this_test_dataset = np.repeat(predictors[np.newaxis, :], vae_anomaly_detection_batch_size, axis=0)
    this_test_loader = DataLoader(this_test_dataset, batch_size=vae_anomaly_detection_batch_size)
    with torch.no_grad():
        for batch_idx, data in enumerate(this_test_loader):
            data = data.float().to(device)
            prob = model.reconstructed_probability(data).mean()
    probNll = -math.log(prob + 0.00001)

    return probNll
def checkSamplingDistribution(cfWindowSize, samples, startIndex):
    if startIndex < 0:
        startIndex = 0
    if startIndex + (cfWindowSize) > predictorTimesteps:
        startIndex =  predictorTimesteps - (cfWindowSize)

    checkpoint = torch.load(foldername + multiTimeSeries + '_' + str(vae_batch_size) +  '_' +str(cfWindowSize) + '_' + X + '_vae_anomaly-detection.ckpt')
    model = VAEAnomalyTabular(cfWindowSize, 2, 2500, 0.005, 1000).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    this_test_dataset = samples[startIndex: startIndex + cfWindowSize]
    this_test_dataset = np.repeat(this_test_dataset[np.newaxis, :], vae_anomaly_detection_batch_size, axis=0)
    this_test_loader = DataLoader(this_test_dataset, batch_size=vae_anomaly_detection_batch_size)
    with torch.no_grad():
        for batch_idx, data in enumerate(this_test_loader):
            data = data.float()
            prob = model.reconstructed_probability(data.to(device)).mean()
    probNll = -math.log(prob)

    return probNll

def maskPredictors(cfWindowSize, Predictors, k=0, test = False):
    startMissingIndices = []
    observedData = []
    observedMask = []
    observedTp = []
    groundTruthMask = []

    if test == False:
        for i in range(len(Predictors)):
            thisStartMissingIndices = set()
            while len(thisStartMissingIndices) < cfWindowSize:
                observedData.append(Predictors[i])
                startIndex = np.random.randint(0, len(Predictors[i]) - cfWindowSize + 1)
                thisStartMissingIndices.add(startIndex)
                observedMask.append([1] * predictorTimesteps)
                thisGroundTruthMask = [1] * predictorTimesteps
                thisGroundTruthMask[startIndex: startIndex + cfWindowSize] = [0] * (cfWindowSize)
                groundTruthMask.append(thisGroundTruthMask)
                observedTp.append(np.arange(len(Predictors[i])).tolist())

            startMissingIndices.append(list(thisStartMissingIndices))

        return torch.from_numpy(np.array(observedTp)), torch.from_numpy(np.array(observedData)), torch.from_numpy(np.array(observedMask)), torch.from_numpy(np.array(groundTruthMask))

    elif test == True:
        probAndIndex = []
        for thisStartIndex in range(predictorTimesteps - cfWindowSize + 1):
            thisProb = checkSamplingDistribution(cfWindowSize, Predictors, thisStartIndex)
            #print(thisStartIndex, thisProb)
            probAndIndex.append([thisProb, thisStartIndex])

        probAndIndex = sorted(probAndIndex, key=lambda x: x[0])
        lowestProb, startIndex = probAndIndex[k][0], probAndIndex[k][1]
        observedMask.append([1] * predictorTimesteps)
        thisGroundTruthMask = [1] * predictorTimesteps
        thisGroundTruthMask[startIndex: startIndex + cfWindowSize] = [0] * (cfWindowSize)
        groundTruthMask.append(thisGroundTruthMask)
        observedTp.append(np.arange(len(Predictors)).tolist())
        observedData.append(Predictors)

        return lowestProb, startIndex, torch.from_numpy(np.array(observedTp)), torch.from_numpy(np.array(observedData)), torch.from_numpy(np.array(observedMask)), torch.from_numpy(np.array(groundTruthMask))

def main():
    resultCol = ['thisPredictor', 'bestCfPredictor', 'CfTargets', 'bestCfPrediction']
    resultDf = pd.DataFrame(columns=resultCol)

    dataProcessor = DataProcessor()
    dataProcessor.readFolder()
    dataProcessor.splitTrainValTest(totalTimesteps, trainTestRatio)
    dataProcessor.splitPredictorTarget(predictorTimesteps)

    if task == 'pre-train-deepExtrema':
        if forecastingModel == 'deepExtrema':
            print('training deepExtrema...')
            (num_epochs, batch_size, lr, n_hidden, n_layers, lambda_, lambda_2) = 3000, batchSize, 0.0005, 16, 2, 0.1, 0.5  # 'lambda_' is for evt loss. 'lambda_2' is for gev_loss
            implementDeepExtrema = ImplementDeepExtrema(batch_size=batch_size, lr=lr, n_hidden=n_hidden,
                                                        n_layers=n_layers, num_epochs=num_epochs)
            implementDeepExtrema.train(X_train=dataProcessor.trainPredictors, y_train=dataProcessor.trainTargets,
                                       X_val=dataProcessor.valPredictors, y_val=dataProcessor.valTargets,
                                       X_test=dataProcessor.testPredictors, y_test=dataProcessor.testTargets,
                                       lambda_=lambda_, lambda_2=lambda_2)
            print('training deepExtrema completed')

    elif task == 'pre-train-csdi':
        print('pre-training CSDI...')
        for cfWindowSize in cfWindowSizeList:
            print(cfWindowSize)
            trainPredictors = np.squeeze(np.array(dataProcessor.trainPredictors.cpu()), axis=2)
            valPredictors = np.squeeze(np.array(dataProcessor.valPredictors.cpu()), axis=2)
            testPredictors = np.squeeze(np.array(dataProcessor.testPredictors.cpu()), axis=2)
            observedTpTrain, observedDataTrain, observedMaskTrain, groundTruthMaskTrain = maskPredictors(cfWindowSize, trainPredictors)
            observedTpVal, observedDataVal, observedMaskVal, groundTruthMaskVal = maskPredictors(cfWindowSize, valPredictors)
            observedTpTest, observedDataTest, observedMaskTest, groundTruthMaskTest = maskPredictors(cfWindowSize, testPredictors)
            trainLoader = DataLoader(TensorDataset(observedTpTrain, observedDataTrain, observedMaskTrain, groundTruthMaskTrain), batch_size = config["train"]["batch_size"], shuffle = 1)
            valLoader = DataLoader(TensorDataset(observedTpVal, observedDataVal, observedMaskVal, groundTruthMaskVal), batch_size = config["train"]["batch_size"], shuffle = 0)
            testLoader = DataLoader(TensorDataset(observedTpTest, observedDataTest, observedMaskTest, groundTruthMaskTest),batch_size=config["train"]["batch_size"], shuffle=0)
            csdi = CSDI(cfWindowSize)
            csdiTrain(cfWindowSize, csdi, config["train"], trainLoader, valid_loader = valLoader, foldername = foldername,)
        print('pre-trained CSDI completed')

    elif task == 'pre-train-vae':
        print('pre-training VAE...')
        for cfWindowSize in cfWindowSizeList:
            trainVAE(cfWindowSize, foldername, dataProcessor.trainPredictors, dataProcessor.valPredictors, 200, vae_batch_size, 0.01)
        print('pre-trained VAE completed')

    elif task == 'pre-train-vae-anomaly-detection':
        print('pre-training VAE anomaly detection...')
        for cfWindowSize in cfWindowSizeList:
            print(cfWindowSize)
            train_dataset = np.squeeze(np.array(dataProcessor.trainPredictors.cpu()), axis=2)
            train_dataset = sliding_window_transform(train_dataset, cfWindowSize)
            train_loader = DataLoader(train_dataset, batch_size=vae_anomaly_detection_batch_size, shuffle=True)
            val_dataset = np.squeeze(np.array(dataProcessor.valPredictors.cpu()), axis=2)
            val_dataset = sliding_window_transform(val_dataset, cfWindowSize)
            val_loader = DataLoader(val_dataset, batch_size=vae_anomaly_detection_batch_size)
            checkpoint_callback = ModelCheckpoint(
                dirpath=foldername,
                filename=multiTimeSeries + '_' + str(vae_batch_size) + '_' +str(cfWindowSize) + '_' + X + '_vae_anomaly-detection',
                save_top_k=1,
                mode='min',
            )
            vaeAnomalyDetection =  VAEAnomalyTabular(cfWindowSize, 2, 2500, 0.005, 1000)
            trainer = pl.Trainer(max_epochs=15, callbacks=[checkpoint_callback])
            trainer.fit(vaeAnomalyDetection, train_loader, val_loader)
        print('pre-trained VAE anomaly detection completed')

    elif task == 'cf':
        max_iter, lr, pred_weight = 100, 0.01, 0.5
        cfLearner = CfLearner(max_iter, lr, pred_weight)
        deepExtremaModel = torch.load(modelSavePath).to(device)
        deepExtremaModel.eval()

        blockMaximaDistributionAnalysis = 0
        if blockMaximaDistributionAnalysis == 1:
            weekdayYu = []
            weekendYu = []
            weekdayReal = []
            weekendReal = []

        print('generating counterfactual block maxima...')
        for i in range(len(dataProcessor.allData)):
            thisPredictor = dataProcessor.allDataPredictors[i].expand(batchSize, -1, -1)
            thisTarget = dataProcessor.allDataTargets[i].expand(batchSize, 1)
            thisPrediction, y_u, thisCfTarget = cfLearner.getCfTarget(deepExtremaModel, thisPredictor, thisTarget, dataProcessor.allDataTargets)
            if thisPrediction > y_u:
                print(thisPrediction, y_u, thisCfTarget)
            dataProcessor.allDataPredictions.append(thisPrediction)
            dataProcessor.allDataCfTargets.append(thisCfTarget)
            dataProcessor.allDataYu.append(y_u)

            if blockMaximaDistributionAnalysis == 1 and dataset == 'dodgersLoopSensor':
                maxValue, maxValuePosi = torch.max(dataProcessor.allDataForecast[i], dim=0)
                maxValueDate = dataProcessor.allDataForecastYMS[i][maxValuePosi]
                weekday = datetime.datetime.strptime(maxValueDate, "%Y-%m-%d %H:%M:%S").weekday()
                if weekday == 5 or weekday == 6:
                    weekendYu.append(y_u.item())
                    weekendReal.append(dataProcessor.allDataTargets[i])
                else:
                    weekdayYu.append(y_u.item())
                    weekdayReal.append(dataProcessor.allDataTargets[i])

        if blockMaximaDistributionAnalysis == 1 and dataset == 'dodgersLoopSensor':
            weekdayYu = [x * dataProcessor.scalers['scale'] + dataProcessor.scalers['mean'] for x in weekdayYu]
            weekendYu = [x * dataProcessor.scalers['scale'] + dataProcessor.scalers['mean'] for x in weekendYu]
            weekdayReal = [x.cpu() * dataProcessor.scalers['scale'] + dataProcessor.scalers['mean'] for x in weekdayReal]
            weekendReal = [x.cpu() * dataProcessor.scalers['scale'] + dataProcessor.scalers['mean'] for x in weekendReal]

            data_list = np.unique(weekdayReal, return_counts=True)
            value_intervals = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400]
            hist, bin_edges = np.histogram(data_list, bins=value_intervals)
            total_count = sum(hist)
            probabilities = hist / total_count
            plt.figure(figsize=(10, 10))
            plt.bar(bin_edges[:-1], probabilities, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
            plt.xlabel('Maximum Daily Flow', fontsize=25)
            plt.ylabel('Probability', fontsize=25)
            plt.xticks(value_intervals, rotation = 45, fontsize=18)
            plt.yticks(fontsize=18)
            plt.title('Weekdays', fontsize = 30)
            plt.show()

            print(sum(weekdayReal)/len(weekdayReal))
            print(sum(weekendReal)/len(weekendReal))

        print('generating counterfactual block maxima completed...')

        print('generating counterfactual predictors...')
        if (forecastingModel == 'deepExtrema'):
            checkCount = 0

            if cfMethod == 'vae':
                for cfWindowSize in cfWindowSizeList:
                    totalEpoch = 0
                    totalHit = 0
                    totalWholeProb = []
                    totalRmse = []
                    totalProximity = []
                    totalSparsity = []
                    totalSampleProb = []
                    totalLeftEndPointSampleProb = []
                    totalRightEndPointSampleProb = []
                    totalLnorm = []
                    totalSmooth = []
                    totalConsecutiveness = []

                    print(cfWindowSize)
                    for i in range(len(dataProcessor.allData)):
                        if multiTimeSeries == 'multivariate':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][0][-1], '%Y-%m')
                        elif multiTimeSeries == 'univariate':
                            if dataset == 'Weather':
                                endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
                            elif dataset == 'GSOD':
                                endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1][0], '%Y-%m')
                            elif dataset == 'S&P500':
                                endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d')
                            elif dataset == 'dodgersLoopSensor':
                                endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S')

                        if (endDate > valTestSplitYear):
                            # (vae) from extreme to non-extreme
                            if (dataProcessor.allDataYu[i].item() < dataProcessor.allDataPredictions[i].item()) and cfSituation == 'e2ne':
                                thisPredictor = np.squeeze(np.array(dataProcessor.allDataPredictors[i].cpu()))
                                bestCfPredictor = None
                                bestCfPrediction = None
                                lowestPredictionError = 999

                                for k in range(kCsdi):
                                    # thisPredictor = np.repeat(thisPredictor[np.newaxis, :], config["train"]["batch_size"], axis=0)
                                    lowestProb, startIndex, observedTp, observedData, observedMask, groundTruthMask = maskPredictors(cfWindowSize, thisPredictor, test=True)
                                    vaeModel = torch.load(foldername + multiTimeSeries + '_' + str(vae_batch_size) + '_' + str(cfWindowSize) + '_' + X + "_vae_model.pth").to(device)
                                    vaeModel.eval()
                                    vaeInput = observedData[groundTruthMask == 0].to(device)

                                    for thisSamplingEpoch in range(200):
                                        vaeOutput = vaeModel(vaeInput)                                          # torch.tensor
                                        samples = observedData.to(device)
                                        samples[groundTruthMask == 0] = vaeOutput[0]
                                        thisCfPredictor = (samples.squeeze(1).to(device) * (1 - groundTruthMask.to(device)) + torch.from_numpy(thisPredictor).to(device) * groundTruthMask.to(device))[0]
                                        y_max = dataProcessor.allDataTargets.max()
                                        y_min = dataProcessor.allDataTargets.min()
                                        thisCfPredictor = thisCfPredictor.unsqueeze(1).expand(batchSize, -1, -1)
                                        zero_tensor = torch.tensor(0.0)
                                        mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor
                                        mu, sigma, xi_p, xi_n, thisCfPrediction = deepExtremaModel(thisCfPredictor.to(device), y_max,y_min, mu_fix, sigma_fix,xi_p_fix, xi_n_fix)
                                        if abs(thisCfPrediction[0].detach().cpu().numpy() - dataProcessor.allDataCfTargets[i].detach().cpu().numpy()) < lowestPredictionError:
                                            lowestPredictionError = abs(thisCfPrediction[0].detach().cpu().numpy() - dataProcessor.allDataCfTargets[i].detach().cpu().numpy())
                                            bestCfPredictor = thisCfPredictor
                                            bestCfPrediction = thisCfPrediction

                                        if dataProcessor.allDataYu[i].item() > bestCfPrediction[0].detach().cpu().numpy():
                                            break
                                    if dataProcessor.allDataYu[i].item() > bestCfPrediction[0].detach().cpu().numpy():
                                        totalHit += 1
                                        break

                                samples = bestCfPredictor[0].detach().cpu().numpy().flatten()
                                thisSampleProb = checkSamplingDistribution(cfWindowSize, samples, np.argmax(groundTruthMask == 0))
                                thisLeftEndPointSampleProb = checkSamplingDistribution(cfWindowSize, samples,np.argmax(groundTruthMask == 0) - 3)
                                thisRightEndPointSampleProb = checkSamplingDistribution(cfWindowSize, samples,np.argmax(groundTruthMask == 0) + 3)
                                thisWholeProb = checkWholeDistribution(samples)
                                thisRmse, thisProximity, thisSparsity, thisLnorm, thisSmooth, thisConsecutiveness = cfLearner.metricEval(thisPredictor,bestCfPredictor[0].detach().cpu().numpy().flatten(),
                                                                                                dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                                                                                bestCfPrediction[0].detach().cpu().numpy())

                                if thisWholeProb < 9.99:
                                    totalEpoch += 1
                                    totalWholeProb.append(thisWholeProb)
                                    totalRmse.append(thisRmse)
                                    totalProximity.append(thisProximity)
                                    totalSparsity.append(thisSparsity)
                                    totalSampleProb.append(thisSampleProb)
                                    totalLeftEndPointSampleProb.append(thisLeftEndPointSampleProb)
                                    totalRightEndPointSampleProb.append(thisRightEndPointSampleProb)
                                    totalLnorm.append(thisLnorm)
                                    totalSmooth.append(thisSmooth)
                                    totalConsecutiveness.append(thisConsecutiveness)

                                new_row_df = pd.DataFrame([[thisPredictor,
                                                                 bestCfPredictor[0].detach().cpu().numpy().flatten(),
                                                                 dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                                                 bestCfPrediction[0].detach().cpu().numpy()]],
                                                          columns=resultCol)
                                resultDf = pd.concat([resultDf, new_row_df], ignore_index=True)

                                cfLearner.plot(dataProcessor.scalers,
                                               dataProcessor.allDataPredictors[i],
                                               dataProcessor.allDataPredictorsYMS[i],
                                               bestCfPredictor[0],
                                               dataProcessor.allDataForecast[i],
                                               dataProcessor.allDataForecastYMS[i],
                                               dataProcessor.allDataPredictions[i],
                                               dataProcessor.allDataCfTargets[i],
                                               bestCfPrediction[0])

                    sumTotalRmse = sum(totalRmse)
                    if not isinstance(sumTotalRmse, float):
                        sumTotalRmse = sumTotalRmse[0]

                    print(totalEpoch,
                          round(sum(totalSparsity) / totalEpoch, 2),
                          round(sum(totalConsecutiveness) / totalEpoch, 2),
                          #round(sum(totalSmooth) / totalEpoch, 2),
                          #round(sum(totalLnorm) / totalEpoch, 2),
                          round(sum(totalProximity) / totalEpoch, 2),
                          round(sum(totalWholeProb) / totalEpoch, 2),
                          round(totalHit / totalEpoch, 2),
                          round(sumTotalRmse / totalEpoch, 2),
                          round(sum(totalSampleProb) / totalEpoch, 2),
                          round(sum(totalLeftEndPointSampleProb) / totalEpoch, 2),
                          round(sum(totalRightEndPointSampleProb) / totalEpoch, 2)
                          )

                    print(totalEpoch,round(math.sqrt(sum((x - sum(totalSparsity) / totalEpoch) ** 2 for x in totalSparsity) / (len(totalSparsity) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalConsecutiveness) / totalEpoch) ** 2 for x in totalConsecutiveness) / (len(totalConsecutiveness) - 1)), 2),
                          #round(math.sqrt(sum((x - sum(totalSmooth) / totalEpoch) ** 2 for x in totalSmooth) / (len(totalSmooth) - 1)), 2),
                          #round(math.sqrt(sum((x - sum(totalLnorm) / totalEpoch) ** 2 for x in totalLnorm) / (len(totalLnorm) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalProximity) / totalEpoch) ** 2 for x in totalProximity) / (len(totalProximity) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalWholeProb) / totalEpoch) ** 2 for x in totalWholeProb) / (len(totalWholeProb) - 1)), 2),
                          round(totalHit / totalEpoch, 2),
                          round(sumTotalRmse / totalEpoch, 2),
                          round(math.sqrt(sum((x - sum(totalSampleProb) / totalEpoch) ** 2 for x in totalSampleProb) / (len(totalSampleProb) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalLeftEndPointSampleProb) / totalEpoch) ** 2 for x in totalLeftEndPointSampleProb) / (len(totalLeftEndPointSampleProb) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalRightEndPointSampleProb) / totalEpoch) ** 2 for x in totalRightEndPointSampleProb) / (len(totalRightEndPointSampleProb) - 1)), 2),
                          )

                    resultDf.to_csv(dataset + '_' + cfMethod + '_' + str(cfWindowSize) + '.csv', index=False)

            elif cfMethod == 'csdi':
                for cfWindowSize in cfWindowSizeList:
                    totalEpoch = 0
                    totalHit = 0
                    totalWholeProb = []
                    totalRmse = []
                    totalProximity = []
                    totalSparsity = []
                    totalSampleProb = []
                    totalLeftEndPointSampleProb = []
                    totalRightEndPointSampleProb = []
                    totalLnorm = []
                    totalSmooth = []
                    totalConsecutiveness = []

                    print(cfWindowSize)
                    for i in range(len(dataProcessor.allData)):
                        if multiTimeSeries == 'multivariate':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][0][-1], '%Y-%m')
                        elif multiTimeSeries == 'univariate':
                            if dataset == 'Weather':
                                endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
                            elif dataset == 'GSOD':
                                endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1][0],'%Y-%m')
                            elif dataset == 'S&P500':
                                endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1],'%Y-%m-%d')
                            elif dataset == 'dodgersLoopSensor':
                                endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S')

                        if (endDate > valTestSplitYear):
                            # (csdi) from extreme to non-extreme
                            if (dataProcessor.allDataYu[i].item() < dataProcessor.allDataPredictions[i].item()) and cfSituation == 'e2ne':
                                # and dataProcessor.allDataPredictorsYMS[i][0][1] == "72222313899" and dataProcessor.allDataPredictorsYMS[i][-1][0] == '2023-05'

                                # thisPredictor = np.repeat(thisPredictor[np.newaxis, :], config["train"]["batch_size"], axis=0)
                                thisPredictor = np.squeeze(np.array(dataProcessor.allDataPredictors[i].cpu()))
                                bestCfPredictor = None
                                bestCfPrediction = None
                                lowestPredictionError = 999

                                for k in range(kCsdi):
                                    lowestProb, startIndex, observedTp, observedData, observedMask, groundTruthMask = maskPredictors(cfWindowSize, thisPredictor, k, test=True)
                                    testLoader = DataLoader(TensorDataset(observedTp, observedData, observedMask, groundTruthMask),batch_size=config["train"]["batch_size"], shuffle=0)
                                    csdiModel = torch.load(foldername + multiTimeSeries + '_' + str(vae_batch_size) + '_' + str(cfWindowSize) + '_' + X + "_csdi_model.pth").to(device)
                                    csdiModel.eval()

                                    for thisSamplingEpoch in range(300):
                                        samples, _, _, _, _ = csdiModel.evaluate(testLoader)
                                        thisCfPredictor = (samples.squeeze(1).to(device) * (1 - groundTruthMask.to(device)) + torch.from_numpy(thisPredictor).to(device) * groundTruthMask.to(device))[0]
                                        y_max = dataProcessor.allDataTargets.max()
                                        y_min = dataProcessor.allDataTargets.min()
                                        thisCfPredictor = thisCfPredictor.unsqueeze(1).expand(batchSize, -1, -1)
                                        zero_tensor = torch.tensor(0.0)
                                        mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor
                                        mu, sigma, xi_p, xi_n, thisCfPrediction = deepExtremaModel(thisCfPredictor.to(device), y_max,y_min, mu_fix, sigma_fix,xi_p_fix, xi_n_fix)
                                        if abs(thisCfPrediction[0].detach().cpu().numpy() - dataProcessor.allDataCfTargets[i].detach().cpu().numpy()) < lowestPredictionError:
                                            lowestPredictionError = abs(thisCfPrediction[0].detach().cpu().numpy() - dataProcessor.allDataCfTargets[i].detach().cpu().numpy())
                                            bestCfPredictor = thisCfPredictor
                                            bestCfPrediction = thisCfPrediction

                                        if dataProcessor.allDataYu[i].item() > bestCfPrediction[0].detach().cpu().numpy():
                                            break
                                    if dataProcessor.allDataYu[i].item() > bestCfPrediction[0].detach().cpu().numpy():
                                        totalHit += 1
                                        break

                                samples = bestCfPredictor[0].cpu().numpy().flatten()
                                thisSampleProb = checkSamplingDistribution(cfWindowSize, samples, np.argmax(groundTruthMask == 0))
                                thisLeftEndPointSampleProb = checkSamplingDistribution(cfWindowSize, samples,np.argmax(groundTruthMask == 0) - 3)
                                thisRightEndPointSampleProb = checkSamplingDistribution(cfWindowSize, samples,np.argmax(groundTruthMask == 0) + 3)
                                thisWholeProb = checkWholeDistribution(samples)
                                thisRmse, thisProximity, thisSparsity, thisLnorm, thisSmooth, thisConsecutiveness = cfLearner.metricEval(thisPredictor,
                                                                                                                    bestCfPredictor[0].detach().cpu().numpy().flatten(),
                                                                                                                    dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                                                                                                    bestCfPrediction[0].detach().cpu().numpy())

                                if thisWholeProb < 9.99:
                                    totalEpoch += 1
                                    totalWholeProb.append(thisWholeProb)
                                    totalRmse.append(thisRmse)
                                    totalProximity.append(thisProximity)
                                    totalSparsity.append(thisSparsity)
                                    totalSampleProb.append(thisSampleProb)
                                    totalLeftEndPointSampleProb.append(thisLeftEndPointSampleProb)
                                    totalRightEndPointSampleProb.append(thisRightEndPointSampleProb)
                                    totalLnorm.append(thisLnorm)
                                    totalSmooth.append(thisSmooth)
                                    totalConsecutiveness.append(thisConsecutiveness)

                                new_row_df = pd.DataFrame([[thisPredictor,
                                                                 bestCfPredictor[0].detach().cpu().numpy().flatten(),
                                                                 dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                                                 bestCfPrediction[0].detach().cpu().numpy()]],
                                                          columns=resultCol)
                                resultDf = pd.concat([resultDf, new_row_df], ignore_index=True)



                                cfLearner.plot(dataProcessor.scalers,
                                               dataProcessor.allDataPredictors[i],
                                               dataProcessor.allDataPredictorsYMS[i],
                                               bestCfPredictor[0],
                                               dataProcessor.allDataForecast[i],
                                               dataProcessor.allDataForecastYMS[i],
                                               dataProcessor.allDataPredictions[i],
                                               dataProcessor.allDataCfTargets[i],
                                               bestCfPrediction[0])

                    sumTotalRmse = sum(totalRmse)
                    if not isinstance(sumTotalRmse, float) or not isinstance(sumTotalRmse, int):
                        sumTotalRmse = sumTotalRmse[0]

                    print(totalEpoch,
                          round(sum(totalSparsity) / totalEpoch, 2),
                          round(sum(totalConsecutiveness) / totalEpoch, 2),
                          #round(sum(totalSmooth) / totalEpoch, 2),
                          #round(sum(totalLnorm) / totalEpoch, 2),
                          round(sum(totalProximity) / totalEpoch, 2),
                          round(sum(totalWholeProb) / totalEpoch, 2),
                          round(totalHit / totalEpoch, 2),
                          round(sumTotalRmse / totalEpoch, 2),
                          round(sum(totalSampleProb) / totalEpoch, 2),
                          round(sum(totalLeftEndPointSampleProb) / totalEpoch, 2),
                          round(sum(totalRightEndPointSampleProb) / totalEpoch, 2)
                          )

                    print(totalEpoch,
                          round(math.sqrt(sum((x - sum(totalSparsity) / totalEpoch) ** 2 for x in totalSparsity) / (len(totalSparsity) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalConsecutiveness) / totalEpoch) ** 2 for x in totalConsecutiveness) / (len(totalConsecutiveness) - 1)), 2),
                          #round(math.sqrt(sum((x - sum(totalSmooth) / totalEpoch) ** 2 for x in totalSmooth) / (len(totalSmooth) - 1)), 2),
                          #round(math.sqrt(sum((x - sum(totalLnorm) / totalEpoch) ** 2 for x in totalLnorm) / (len(totalLnorm) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalProximity) / totalEpoch) ** 2 for x in totalProximity) / (len(totalProximity) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalWholeProb) / totalEpoch) ** 2 for x in totalWholeProb) / (len(totalWholeProb) - 1)), 2),
                          round(totalHit / totalEpoch, 2),
                          round(sumTotalRmse / totalEpoch, 2),
                          round(math.sqrt(sum((x - sum(totalSampleProb) / totalEpoch) ** 2 for x in totalSampleProb) / (len(totalSampleProb) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalLeftEndPointSampleProb) / totalEpoch) ** 2 for x in totalLeftEndPointSampleProb) / (len(totalLeftEndPointSampleProb) - 1)), 2),
                          round(math.sqrt(sum((x - sum(totalRightEndPointSampleProb) / totalEpoch) ** 2 for x in totalRightEndPointSampleProb) / (len(totalRightEndPointSampleProb) - 1)), 2)
                          )

                    resultDf.to_csv(dataset + '_' + cfMethod + '_' + str(cfWindowSize) + '.csv', index=False)

            elif cfMethod == 'NGCF':
                totalEpoch = 0
                totalHit = 0
                totalWholeProb = []
                totalRmse = []
                totalProximity = []
                totalSparsity = []
                totalSampleProb = []
                totalLeftEndPointSampleProb = []
                totalRightEndPointSampleProb = []
                totalLnorm = []
                totalSmooth= []
                totalConsecutiveness = []

                for i in range(len(dataProcessor.allData)):
                    if dataset == 'Weather':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
                    elif dataset == 'GSOD':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1][0], '%Y-%m')
                    elif dataset == 'S&P500':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d')
                    elif dataset == 'dodgersLoopSensor':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S')

                    if (endDate > valTestSplitYear):
                        thisPredictor = dataProcessor.allDataPredictors[i]
                        NGCF = NativeGuide()
                        NGCF = NGCF.fit(dataProcessor.allDataPredictors, dataProcessor.allDataTargets, dataProcessor.allDataCfTargets[i])   # type: # <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'torch.Tensor'>

                        if dataProcessor.allDataYu[i].item() < dataProcessor.allDataPredictions[i].item() and cfSituation == 'e2ne':
                            beta = 0.0
                            min_error = 99999.0
                            bestCfPredictor = None
                            bestCfPrediction = None

                            for epoch in range(100):
                                thisCfPredictor = NGCF.transform(dataProcessor.allDataPredictors[i], beta, i)  # return numpy
                                thisCfPredictor = torch.tensor(np.array(thisCfPredictor).reshape(-1, 1), dtype=torch.float32)
                                thisCfPredictor = thisCfPredictor.expand(batchSize, -1, -1).to(device)
                                y_max = dataProcessor.allDataTargets.max()
                                y_min = dataProcessor.allDataTargets.min()
                                zero_tensor = torch.tensor(0.0)
                                mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor
                                mu, sigma, xi_p, xi_n, thisCfPrediction = deepExtremaModel(thisCfPredictor, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix)
                                beta += 0.01
                                error = abs(thisCfPrediction[0] - dataProcessor.allDataCfTargets[i])
                                if error < min_error:
                                    min_error = error
                                    bestCfPredictor = thisCfPredictor
                                    bestCfPrediction = thisCfPrediction[0]

                            thisWholeProb = checkWholeDistribution(bestCfPredictor[0].detach().cpu().numpy().flatten())
                            thisRmse, thisProximity, thisSparsity, thisLnorm, thisSmooth, thisConsecutiveness = cfLearner.metricEval(
                                thisPredictor.detach().cpu().numpy().flatten(),
                                bestCfPredictor[0].detach().cpu().numpy().flatten(),
                                dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                bestCfPrediction[0].detach().cpu().numpy())
                            if dataProcessor.allDataYu[i].item() > thisCfPrediction[0].detach().cpu().numpy():
                                totalHit += 1

                            totalEpoch += 1
                            totalWholeProb.append(thisWholeProb)
                            totalRmse.append(thisRmse)
                            totalProximity.append(thisProximity)
                            totalSparsity.append(thisSparsity)
                            totalLnorm.append(thisLnorm)
                            totalSmooth.append(thisSmooth)
                            totalConsecutiveness.append(thisConsecutiveness)

                            new_row_df = pd.DataFrame([[thisPredictor.detach().cpu().numpy().flatten(),
                                                             bestCfPredictor[0].detach().cpu().numpy().flatten(),
                                                             dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                                             bestCfPrediction[0].detach().cpu().numpy()]], columns=resultCol)
                            resultDf = pd.concat([resultDf, new_row_df], ignore_index=True)

                resultDf.to_csv(dataset + '_' + cfMethod + '.csv', index=False)

            elif cfMethod == 'baseNN':
                totalEpoch = 0
                totalHit = 0
                totalWholeProb = []
                totalRmse = []
                totalProximity = []
                totalSparsity = []
                totalSampleProb = []
                totalLeftEndPointSampleProb = []
                totalRightEndPointSampleProb = []
                totalLnorm = []
                totalSmooth = []
                totalConsecutiveness = []

                for i in range(len(dataProcessor.allData)):
                    if dataset == 'Weather':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
                    elif dataset == 'GSOD':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1][0], '%Y-%m')
                    elif dataset == 'S&P500':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d')
                    elif dataset == 'dodgersLoopSensor':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S')

                    if (endDate > valTestSplitYear):
                        thisPredictor = dataProcessor.allDataPredictors[i]
                        baseNN = BaseNN()
                        baseNN = baseNN.fit(dataProcessor.allDataPredictors, dataProcessor.allDataTargets)
                        thisCfPredictor = baseNN.transform(np.array(dataProcessor.allDataCfTargets[i].cpu()).reshape(1, -1))
                        thisCfPredictor = thisCfPredictor.expand(batchSize, -1, -1)
                        y_max = dataProcessor.allDataTargets.max()
                        y_min = dataProcessor.allDataTargets.min()
                        zero_tensor = torch.tensor(0.0)
                        mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor
                        mu, sigma, xi_p, xi_n, thisCfPrediction = deepExtremaModel(thisCfPredictor, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix)
                        thisWholeProb = checkWholeDistribution(thisCfPredictor[0].detach().cpu().numpy().flatten())
                        thisRmse, thisProximity, thisSparsity, thisLnorm, thisSmooth, thisConsecutiveness = cfLearner.metricEval(
                            thisPredictor.detach().cpu().numpy().flatten(),
                            thisCfPredictor[0].detach().cpu().numpy().flatten(),
                            dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                            thisCfPrediction[0].detach().cpu().numpy())

                        if dataProcessor.allDataYu[i].item() < dataProcessor.allDataPredictions[i].item() and cfSituation == 'e2ne' and thisWholeProb < 9.99:
                            totalEpoch += 1
                            if dataProcessor.allDataYu[i].item() > thisCfPrediction[0].detach().cpu().numpy():
                                totalHit += 1
                            totalWholeProb.append(thisWholeProb)
                            totalRmse.append(thisRmse)
                            totalProximity.append(thisProximity)
                            totalSparsity.append(thisSparsity)
                            totalLnorm.append(thisLnorm)
                            totalSmooth.append(thisSmooth)
                            totalConsecutiveness.append(thisConsecutiveness)

                            new_row_df = pd.DataFrame([[thisPredictor.detach().cpu().numpy().flatten(),
                                                        thisCfPredictor[0].detach().cpu().numpy().flatten(),
                                                        dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                                        thisCfPrediction[0].detach().cpu().numpy()]], columns=resultCol)
                            resultDf = pd.concat([resultDf, new_row_df], ignore_index=True)

                            cfLearner.plot(dataProcessor.scalers,
                                           thisPredictor,
                                           dataProcessor.allDataPredictorsYMS[i],
                                           thisCfPredictor[0],
                                           dataProcessor.allDataForecast[i],
                                           dataProcessor.allDataForecastYMS[i],
                                           dataProcessor.allDataPredictions[i],
                                           dataProcessor.allDataCfTargets[i],
                                           thisCfPrediction[0])

                resultDf.to_csv(dataset + '_' + cfMethod + '.csv', index=False)

            elif cfMethod == 'benchmark':
                totalEpoch = 0
                totalHit = 0
                totalWholeProb = []
                totalRmse = []
                totalProximity = []
                totalSparsity = []
                totalSampleProb = []
                totalLeftEndPointSampleProb = []
                totalRightEndPointSampleProb = []
                totalLnorm = []
                totalSmooth= []
                totalConsecutiveness = []

                for i in range(len(dataProcessor.allData)):
                    if multiTimeSeries == 'multivariate':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][0][-1], '%Y-%m')
                    elif multiTimeSeries == 'univariate':
                        if dataset == 'Weather':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
                        elif dataset == 'GSOD':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1][0], '%Y-%m')
                        elif dataset == 'S&P500':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d')
                        elif dataset == 'dodgersLoopSensor':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1],'%Y-%m-%d %H:%M:%S')

                    if (endDate >= valTestSplitYear):
                        # (benchmark) from extreme to non-extreme
                        if (dataProcessor.allDataYu[i].item() < dataProcessor.allDataPredictions[i].item()) and cfSituation == 'e2ne':
                            thisPredictor = dataProcessor.allDataPredictors[i].expand(batchSize, -1, -1)
                            thisCfTarget = dataProcessor.allDataCfTargets[i].expand(batchSize, 1)
                            thisCfPredictor, thisCfPrediction = cfLearner.getCfPredictors(deepExtremaModel,
                                                                                          thisPredictor, thisCfTarget,
                                                                                          dataProcessor.allDataTargets)
                            thisWholeProb = checkWholeDistribution(thisCfPredictor.detach().cpu().numpy().flatten())
                            thisRmse, thisProximity, thisSparsity, thisLnorm, thisSmooth, thisConsecutiveness = cfLearner.metricEval(
                                thisPredictor[0].detach().cpu().numpy().flatten(),
                                thisCfPredictor.detach().cpu().numpy().flatten(),
                                dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                thisCfPrediction[0].detach().cpu().numpy().flatten())
                            if dataProcessor.allDataYu[i].item() > thisCfPrediction[0].detach().cpu().numpy():
                                totalHit += 1

                            if thisWholeProb < 9.99:
                                totalEpoch += 1
                                totalWholeProb.append(thisWholeProb)
                                totalRmse.append(thisRmse)
                                totalProximity.append(thisProximity)
                                totalSparsity.append(thisSparsity)
                                totalLnorm.append(thisLnorm)
                                totalSmooth.append(thisSmooth)
                                totalConsecutiveness.append(thisConsecutiveness)

                            new_row_df = pd.DataFrame([[thisPredictor[0].detach().cpu().numpy().flatten(),
                                                             thisCfPredictor.detach().cpu().numpy().flatten(),
                                                             dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                                             thisCfPrediction[0].detach().cpu().numpy().flatten()]], columns=resultCol)
                            resultDf = pd.concat([resultDf, new_row_df], ignore_index=True)

                            cfLearner.plot(dataProcessor.scalers,
                                           thisPredictor[0],
                                           dataProcessor.allDataPredictorsYMS[i],
                                           thisCfPredictor,
                                           dataProcessor.allDataForecast[i],
                                           dataProcessor.allDataForecastYMS[i],
                                           dataProcessor.allDataPredictions[i],
                                           dataProcessor.allDataCfTargets[i],
                                           thisCfPrediction[0])

                resultDf.to_csv(dataset + '_' + benchmarkBaseline + '.csv', index=False)

            elif cfMethod == "SPARCE":
                totalEpoch = 0
                totalHit = 0
                totalWholeProb = []
                totalRmse = []
                totalProximity = []
                totalSparsity = []
                totalSampleProb = []
                totalLeftEndPointSampleProb = []
                totalRightEndPointSampleProb = []
                totalLnorm = []
                totalSmooth = []
                totalConsecutiveness = []

                args = parser.parse_args()
                random.seed(args.seed)

                Yu = [cuda_tensor.item() for cuda_tensor in dataProcessor.allDataYu]

                X_train = np.vstack((dataProcessor.trainPredictors.cpu(), dataProcessor.valPredictors.cpu(), dataProcessor.testPredictors.cpu()))
                y_train = np.vstack((dataProcessor.trainTargets.cpu(), dataProcessor.valTargets.cpu(), dataProcessor.testTargets.cpu()))
                y_train_class = y_train.flatten() - np.array(Yu[:len(y_train)])
                y_train_class = np.where(y_train_class > 0, 1, 0)

                X_train_target_samples, y_train_target_samples, X_train_generator_input, y_train_generator_input = split_target_and_input(X_train, y_train, y_train_class, args.target_class)

                j = 0
                for i in range(len(dataProcessor.allData)):
                    if multiTimeSeries == 'multivariate':
                        endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][0][-1], '%Y-%m')
                    elif multiTimeSeries == 'univariate':
                        if dataset == 'Weather':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
                        elif dataset == 'GSOD':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1][0], '%Y-%m')
                        elif dataset == 'S&P500':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d')
                        elif dataset == 'dodgersLoopSensor':
                            endDate = datetime.datetime.strptime(dataProcessor.allDataForecastYMS[i][-1], '%Y-%m-%d %H:%M:%S')

                    if (endDate >= valTestSplitYear):
                        if (dataProcessor.allDataYu[i].item() < dataProcessor.allDataPredictions[i].item()) and cfSituation == 'e2ne':
                            X_train_target_samples, y_train_target_samples = take_max_samples(args.seed, X_train_target_samples, y_train_target_samples, batchSize)

                            X_train_real_samples, y_train_real_samples = torch.from_numpy(X_train_target_samples), torch.from_numpy(y_train_target_samples)
                            X_train_generator_input = dataProcessor.allDataPredictors[i].expand(batchSize, -1, -1)
                            y_train_generator_input = dataProcessor.allDataCfTargets[i].expand(batchSize, 1)

                            # construct pytorch datasets
                            train_ds_target = TensorDataset(X_train_real_samples, y_train_real_samples)
                            train_ds_input = TensorDataset(X_train_generator_input, y_train_generator_input)

                            # pass datasets to dataloaders (wraps an iterable over the dataset)
                            train_dl_real_samples = DataLoader(train_ds_target, batchSize, shuffle=False)
                            train_dl_generator_input = DataLoader(train_ds_input, batchSize, shuffle=False)

                            # TODO: since our task is based on univariate time series.
                            num_features = 1
                            y_max = dataProcessor.allDataTargets.max()
                            y_min = dataProcessor.allDataTargets.min()
                            zero_tensor = torch.tensor(0.0)
                            mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor
                            model = CounterfactualTimeGAN(y_max, y_min,  mu_fix, sigma_fix, xi_p_fix, xi_n_fix, dataProcessor.allDataYu[i].item())
                            model.build_model(args=args, device=device, num_features=num_features, bidirectional=True,
                                              hidden_dim_generator=256, layer_dim_generator=2, hidden_dim_discriminator=16,
                                              layer_dim_discriminator=1, classifier_model_name="bidirectional_lstm_classifier")
                            resultDf = model.train(train_dl=train_dl_real_samples, generator_dl=train_dl_generator_input,max_samples=batchSize, max_batches=1)

                            index = (resultDf['predictions'] - dataProcessor.allDataCfTargets[i].detach().cpu().numpy()).abs().idxmin()
                            thisWholeProb = checkWholeDistribution(np.array(resultDf.iloc[index]['generated_sequences']))
                            # print(resultDf['predictions'])
                            thisRmse, thisProximity, thisSparsity, thisLnorm, thisSmooth, thisConsecutiveness = cfLearner.metricEval(
                                np.array(resultDf.iloc[index]['original_sequences']),
                                np.array(resultDf.iloc[index]['generated_sequences']),
                                dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                resultDf.iloc[index]['predictions'])
                            if dataProcessor.allDataYu[i].item() > resultDf.iloc[index]['predictions']:
                                totalHit += 1

                            print(totalEpoch, thisSparsity, thisConsecutiveness, thisProximity, thisWholeProb)

                            if thisWholeProb < 9.99:
                                totalEpoch += 1
                                totalWholeProb.append(thisWholeProb)
                                totalRmse.append(thisRmse)
                                totalProximity.append(thisProximity)
                                totalSparsity.append(thisSparsity)
                                totalLnorm.append(thisLnorm)
                                totalSmooth.append(thisSmooth)
                                totalConsecutiveness.append(thisConsecutiveness)


        sumTotalRmse = sum(totalRmse)
        if not isinstance(sumTotalRmse, float):
            sumTotalRmse = sumTotalRmse[0]

        print(totalEpoch,
              round(sum(totalSparsity) / totalEpoch, 2),
              round(sum(totalConsecutiveness) / totalEpoch, 2),
              #round(sum(totalSmooth) / totalEpoch, 2),
              #round(sum(totalLnorm) / totalEpoch, 2),
              round(sum(totalProximity) / totalEpoch, 2),
              round(sum(totalWholeProb) / totalEpoch, 2),
              round(totalHit / totalEpoch, 2),
              round(sumTotalRmse / totalEpoch, 2),
              round(sum(totalSampleProb) / totalEpoch, 2),
              round(sum(totalLeftEndPointSampleProb) / totalEpoch, 2),
              round(sum(totalRightEndPointSampleProb) / totalEpoch, 2)
              )

        print(totalEpoch,
              round(math.sqrt(sum((x - sum(totalSparsity) / totalEpoch) ** 2 for x in totalSparsity) / (len(totalSparsity) - 1)), 2),
              round(math.sqrt(sum((x - sum(totalConsecutiveness) / totalEpoch) ** 2 for x in totalConsecutiveness) / (len(totalConsecutiveness) - 1)), 2),
              #round(math.sqrt(sum((x - sum(totalSmooth) / totalEpoch) ** 2 for x in totalSmooth) / (len(totalSmooth) - 1)), 2),
              #round(math.sqrt(sum((x - sum(totalLnorm) / totalEpoch) ** 2 for x in totalLnorm) / (len(totalLnorm) - 1)), 2),
              round(math.sqrt(sum((x - sum(totalProximity) / totalEpoch) ** 2 for x in totalProximity) / (len(totalProximity) - 1)),2),
              round(math.sqrt(sum((x - sum(totalWholeProb) / totalEpoch) ** 2 for x in totalWholeProb) / (len(totalWholeProb) - 1)), 2),
              round(totalHit / totalEpoch, 2),
              round(sumTotalRmse / totalEpoch, 2),
              round(math.sqrt(sum((x - sum(totalSampleProb) / totalEpoch) ** 2 for x in totalSampleProb) / (len(totalSampleProb) - 1)), 2),
              round(math.sqrt(sum((x - sum(totalLeftEndPointSampleProb) / totalEpoch) ** 2 for x in totalLeftEndPointSampleProb) / (len(totalLeftEndPointSampleProb) - 1)), 2),
              round(math.sqrt(sum((x - sum(totalRightEndPointSampleProb) / totalEpoch) ** 2 for x in totalRightEndPointSampleProb) / (len(totalRightEndPointSampleProb) - 1)), 2)
            )

        print('generating counterfactual predictors completed')

    elif task == 'evaCsvResul':
        pass

def split_target_and_input(X, y, y_class, target_class):
    indices_target_class = [index for index, value in enumerate(y_class) if value == target_class]
    indices_non_target_class = [index for index, value in enumerate(y_class) if value != target_class]

    X_target_samples = X[indices_target_class]
    y_target_samples = y[indices_target_class]

    X_generator_input = X[indices_non_target_class]
    y_generator_input = y[indices_non_target_class]

    return X_target_samples, y_target_samples, X_generator_input, y_generator_input

def take_max_samples(seed, X, y, max_samples):
    random.seed(seed)
    # shuffle arrays, then take first max samples
    idx_list = list(range(X.shape[0]))
    random.shuffle(idx_list)

    X = X[idx_list]
    y = y[idx_list]

    X = X[:max_samples]
    y = y[:max_samples]

    return X, y

if __name__ == '__main__':
    print(device)
    print(torch.__version__)
    print(torch.version.cuda)
    main()
