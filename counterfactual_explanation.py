# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/13/2023
# ============================================================================

from pkg_manager import *
from para_manager import *

class CfLearner:
    def __init__(self, max_iter, lr, pred_weight):
        self.max_iter = max_iter
        self.lr = lr
        self.pred_weight = pred_weight

    def getCfTarget(self, model, thisPredictor, thisTarget, allDataTargets):
        test_loader = DataLoader(TensorDataset(thisPredictor, thisTarget), batch_size = batchSize)
        y_max = allDataTargets.max()
        y_min = allDataTargets.min()
        zero_tensor = torch.tensor(0.0)
        mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor

        model.eval()
        for k, (predictors, targets) in enumerate(test_loader):
            if (len(predictors) == batchSize):
                with torch.no_grad():
                    mu, sigma, xi_p, xi_n, prediction = model(predictors, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix)

                y_u = mu[0] + (sigma[0] / xi_n[0]) * ((-math.log(p)) ** (-xi_n[0]) - 1)
                z_max = mu[0] + (sigma[0] / xi_n[0]) * ((-math.log(0.99)) ** (-xi_n[0]) - 1)

                # from extreme to non-extreme
                if prediction[0] >= y_u:
                    lambdaCf = 0.01
                    if cfTargetType == 'expected_value':
                        denominator = 1 / (1 - math.exp(-lambdaCf * y_u))
                        cfTarget = (y_u - (1 / lambdaCf) + (math.exp(-lambdaCf * y_u) / lambdaCf)) * denominator
                    elif cfTargetType == 'prob_distribution':
                        pass
                    elif cfTargetType == 'GEV_distribution':
                        cfTarget = mu[0] + (sigma[0] / xi_n[0]) * ((-math.log(p_target_nex)) ** (-xi_n[0]) - 1)

                # from non-extreme to extreme
                elif prediction[0] < y_u:
                    lambdaCf = 1.5
                    if cfTargetType == 'expected_value':
                        denominator = 1 / (1- math.exp(lambdaCf * (y_u - z_max)))
                        cfTarget = (1 / lambdaCf + y_u - (z_max + 1 / lambdaCf) * math.exp(lambdaCf * (y_u - z_max))) * denominator
                    elif cfTargetType == 'prob_distribution':
                        pass
                    elif cfTargetType == 'GEV_distribution':
                        cfTarget = mu[0] + (sigma[0] / xi_n[0]) * ((-math.log(p_target_ex)) ** (-xi_n[0]) - 1)

        # print(thisTarget[0], prediction[0], y_u, cfTarget)

        return prediction[0], y_u, cfTarget

    def getCfPredictors(self, model, thisPredictor, thisCfTarget, allDataTargets):
        thisCfPredictor = thisPredictor.clone().requires_grad_()
        optimizer = optim.Adam([thisCfPredictor], lr = self.lr)

        if forecastingModel == 'deepExtrema':
            if benchmarkBaseline == 'wCF':
                y_max = allDataTargets.max()
                y_min = allDataTargets.min()

                model.train()
                for epoch in range(self.max_iter):
                    test_loader = DataLoader(TensorDataset(thisCfPredictor, thisCfTarget), batch_size = batchSize)
                    zero_tensor = torch.tensor(0.0)
                    mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor
                    for k, (Z_prime, y_cf) in enumerate(test_loader):
                        if (len(Z_prime) == batchSize):

                            # if multiTimeSeries == 'univariate':
                            #     Z_prime = torch.relu(Z_prime)
                            # elif multiTimeSeries == 'univariate':
                            #     pass

                            mu, sigma, xi_p, xi_n, y_pred = model(Z_prime, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix)
                            mse_loss = torch.nn.MSELoss(reduction='sum')
                            predLoss = mse_loss(y_pred, thisCfTarget)
                            worldDistance = torch.mean(torch.abs(thisPredictor - Z_prime))
                            train_loss = self.pred_weight * predLoss + worldDistance
                            optimizer.zero_grad()
                            train_loss.backward()
                            optimizer.step()

            elif benchmarkBaseline == 'ForecastCF':
                y_max = allDataTargets.max()
                y_min = allDataTargets.min()

                model.train()
                for epoch in range(self.max_iter):
                    test_loader = DataLoader(TensorDataset(thisCfPredictor, thisCfTarget), batch_size=batchSize)
                    zero_tensor = torch.tensor(0.0)
                    mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor
                    for k, (Z_prime, y_cf) in enumerate(test_loader):
                        if (len(Z_prime) == batchSize):
                            mu, sigma, xi_p, xi_n, y_pred = model(Z_prime, y_max, y_min, mu_fix, sigma_fix, xi_p_fix,xi_n_fix)
                            mse_loss = torch.nn.MSELoss(reduction='sum')
                            predLoss1 = mse_loss(y_pred, thisCfTarget - ToleranceOfForecastCF)
                            predLoss2 = mse_loss(y_pred, thisCfTarget + ToleranceOfForecastCF)
                            predLoss = predLoss1 + predLoss2

                            mask = 0 if ((y_pred[0] <= (thisCfTarget[0] + ToleranceOfForecastCF)) and (y_pred[0] >= (thisCfTarget[0] - ToleranceOfForecastCF))) else 1
                            train_loss = mask * predLoss
                            optimizer.zero_grad()
                            train_loss.backward()
                            optimizer.step()
                        if ((y_pred[0] <= (thisCfTarget[0] + ToleranceOfForecastCF)) and (y_pred[0] >= (thisCfTarget[0] - ToleranceOfForecastCF))):
                            break

        mu, sigma, xi_p, xi_n, thisCfPrecition = model(thisCfPredictor, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix)

        return thisCfPredictor[0], thisCfPrecition[0]

    def plot(self, scalers, thisPredictor, thisPredictorYMS, thisCfPredictor, thisForecast, thisForecastYMS, thisPrediction, thisCfTarget, thisCfPrediction,
             lowestProb = None, startIndex  = None, probSample  = None, probLeftEndPoint = None, probRightEndPoint = None):
        if multiTimeSeries == 'multivariate':
            for i in range(len(thisPredictor[-1])):
                thisyear, thisMonth = thisPredictorYMS[0, i].split('-')
                thisMonth = int(thisMonth)
                thisPredictor[-1][i] = thisPredictor[-1][i] * scalers[thisForecastYMS[-1][0]][thisMonth]['scale'] + scalers[thisForecastYMS[-1][0]][thisMonth]['mean']

            thisCfPredictor = thisCfPredictor.detach().numpy()
            for i in range(len(thisCfPredictor[-1])):
                thisyear, thisMonth = thisPredictorYMS[0, i].split('-')
                thisMonth = int(thisMonth)
                thisCfPredictor[-1][i] = thisCfPredictor[-1][i] * scalers[thisForecastYMS[-1][0]][thisMonth]['scale'] + scalers[thisForecastYMS[-1][0]][thisMonth]['mean']

            for i in range(len(thisForecast[-1])):
                thisyear, thisMonth = thisForecastYMS[0, i].split('-')
                thisMonth = int(thisMonth)
                thisForecast[-1][i] = thisForecast[-1][i] * scalers[thisForecastYMS[-1][0]][thisMonth]['scale'] + scalers[thisForecastYMS[-1][0]][thisMonth]['mean']

            totalYM = np.hstack((thisPredictorYMS[0, :], thisForecastYMS[0, :]))
            dates = [datetime.datetime.strptime(date_str, '%Y-%m') for date_str in totalYM]
            totalYM = np.array([date.strftime('%m-%Y') for date in dates])
            thisPredictorYMS = thisPredictorYMS[0, :]
            dates = [datetime.datetime.strptime(date_str, '%Y-%m') for date_str in thisPredictorYMS]
            thisPredictorYMS = np.array([date.strftime('%m-%Y') for date in dates])

            fig, axs = plt.subplots(len(F) + 1, 1, figsize=(10, 30))

            for i in range(len(F)):
                axs[i].plot(totalYM, np.hstack((thisPredictor[i].detach().numpy(), thisForecast[i].detach().numpy().flatten())), label = 'real' + str(F[i]), color = 'blue')
                axs[i].plot(thisPredictorYMS, thisCfPredictor[i], label ='counterfactual ' + str(F[i]), color='red', linestyle='--')
                axs[i].set_ylabel(F[i])
                axs[i].set_xticklabels([])
                axs[i].legend(loc = 'upper left', fontsize = 'small')

            axs[-1].plot(totalYM, np.hstack((thisPredictor[-1].detach().numpy(), thisForecast[-1].detach().numpy().flatten())), label='real X, Y', color='blue')
            axs[-1].plot(thisPredictorYMS, thisCfPredictor[-1], label='counterfactual X', color='red', linestyle='--')
            plt.axvline(x = len(thisPredictor[-1]) - 1, color='gray', linestyle='--')

            thisPrediction = thisPrediction.tolist()
            thisCfTarget = thisCfTarget.tolist()
            thisCfPrediction = thisCfPrediction.tolist()

            posi = len(thisPredictor[-1]) + np.argmax(thisForecast[-1].detach().numpy())
            blockMaximaMonth, blockMaximaYear = totalYM[posi].split('-')
            blockMaximaMonth = int(blockMaximaMonth)

            plt.scatter(totalYM[posi], thisPrediction * scalers[thisForecastYMS[-1][0]][blockMaximaMonth]['scale'] + scalers[thisForecastYMS[-1][0]][blockMaximaMonth]['mean'] , label="predicted block maxima", marker='o', color='green')
            plt.scatter(totalYM[posi], thisCfTarget * scalers[thisForecastYMS[-1][0]][blockMaximaMonth]['scale'] + scalers[thisForecastYMS[-1][0]][blockMaximaMonth]['mean'], label="counterfactual target block maxima", marker='o', color='red')
            plt.scatter(totalYM[posi], thisCfPrediction * scalers[thisForecastYMS[-1][0]][blockMaximaMonth]['scale'] + scalers[thisForecastYMS[-1][0]][blockMaximaMonth]['mean'], label="counterfactual predicted block maxima", marker='o',  facecolors='none', edgecolors='gray')

            plt.xticks(rotation = 45)
            plt.ylabel('Average ' + X)
            plt.title('Climate Station ' + str(thisForecastYMS[-1][0]))
            plt.legend(loc = 'upper left', fontsize = 'small')
            plt.show()

        elif multiTimeSeries == 'univariate' and dataset == 'GSOD':
            for i in range(len(thisPredictor)):
                thisyear, thisMonth = thisPredictorYMS[i, 0].split('-')
                thisMonth = int(thisMonth)
                thisPredictor[i] = thisPredictor[i].cpu() * scalers[thisForecastYMS[0, 1]][thisMonth]['scale'] + scalers[thisForecastYMS[0, 1]][thisMonth]['mean']

            thisCfPredictor = thisCfPredictor.detach().cpu().numpy()
            for i in range(len(thisCfPredictor)):
                thisyear, thisMonth = thisPredictorYMS[i, 0].split('-')
                thisMonth = int(thisMonth)
                thisCfPredictor[i] = thisCfPredictor[i] * scalers[thisForecastYMS[0, 1]][thisMonth]['scale'] + scalers[thisForecastYMS[0, 1]][thisMonth]['mean']

            for i in range(len(thisForecast)):
                thisyear, thisMonth = thisForecastYMS[i, 0].split('-')
                thisMonth = int(thisMonth)
                thisForecast[i] = thisForecast[i].cpu() * scalers[thisForecastYMS[0, 1]][thisMonth]['scale'] + scalers[thisForecastYMS[0, 1]][thisMonth]['mean']

            totalYM = np.hstack((thisPredictorYMS[:, 0], thisForecastYMS[:, 0]))
            dates = [datetime.datetime.strptime(date_str, '%Y-%m') for date_str in totalYM]
            totalYM = np.array([date.strftime('%m-%Y') for date in dates])
            thisPredictorYMS = thisPredictorYMS[:, 0]
            dates = [datetime.datetime.strptime(date_str, '%Y-%m') for date_str in thisPredictorYMS]
            thisPredictorYMS = np.array([date.strftime('%m-%Y') for date in dates])

            #if str(thisForecastYMS[0, 1]) in ['72222313899']:
            if str(thisForecastYMS[0, 1]) is not None:
                plt.figure(figsize=(12, 11))
                plt.plot(totalYM, np.vstack((thisPredictor.detach().cpu().numpy(), thisForecast.detach().cpu().numpy())), label= "original time series", color='blue', linewidth=3)
                plt.plot(thisPredictorYMS, thisCfPredictor, label= "counterfactual instance", color='red', linestyle='--', linewidth=3)
                plt.axvline(x = len(thisPredictor) - 1, color='gray', linestyle='--')
                plt.xticks(rotation = 45, fontsize=25)
                plt.ylabel('Average ' + X, fontsize = 45)
                plt.ylim(-0.1, 0.8)
                plt.title('DiffusionCF', fontsize =45)
                plt.gca().tick_params(axis='y', labelsize=25)
                plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(3))

                thisPrediction = thisPrediction.tolist()
                thisCfTarget = thisCfTarget.tolist()
                thisCfPrediction = thisCfPrediction.tolist()

                posi = len(thisPredictor) + np.argmax(thisForecast.detach().cpu().numpy())
                blockMaximaMonth, blockMaximaYear = totalYM[posi].split('-')
                blockMaximaMonth = int(blockMaximaMonth)
                plt.scatter(totalYM[posi], thisPrediction * scalers[thisForecastYMS[0, 1]][blockMaximaMonth]['scale'] + scalers[thisForecastYMS[0, 1]][blockMaximaMonth]['mean'], marker='o', color='blue', s=120)
                plt.scatter(totalYM[posi], thisCfTarget * scalers[thisForecastYMS[0, 1]][blockMaximaMonth]['scale'] + scalers[thisForecastYMS[0, 1]][blockMaximaMonth]['mean'], marker='o', color='red', s=120)
                # plt.scatter(totalYM[posi], thisCfPrediction * scalers[thisForecastYMS[0, 1]][blockMaximaMonth]['scale'] + scalers[thisForecastYMS[0, 1]][blockMaximaMonth]['mean'], label="counterfactual predicted block maxima", marker='o',  facecolors='none', edgecolors='gray')

                # print('Climate Station:', str(thisForecastYMS[0, 1]))

                # if lowestProb != None:
                #     plt.title('Climate Station ' + str(thisForecastYMS[0, 1]) + ' (' + str(round(lowestProb.item(), 2)) + '/' + str(round(probSample.item(), 2)) + '/' + str(round(probLeftEndPoint.item(), 2)) + '/' + str(round(probRightEndPoint.item(), 2))+')')
                # else:
                #     plt.title('Climate Station ' + str(thisForecastYMS[0, 1]))

                # plt.title('Climate Station ' + str(thisForecastYMS[0, 1]))

                plt.legend(loc = 'upper left', fontsize = 35)
                plt.show()

        elif multiTimeSeries == 'univariate' and dataset == 'dodgersLoopSensor':
            thisPredictor = thisPredictor.cpu() * scalers['scale'] + scalers['mean']
            thisCfPredictor = thisCfPredictor.detach().cpu().numpy()
            thisCfPredictor = thisCfPredictor * scalers['scale'] + scalers['mean']
            thisForecast = thisForecast.cpu() * scalers['scale'] + scalers['mean']

            totalYM = np.hstack((thisPredictorYMS[:], thisForecastYMS[:]))
            splitTime = datetime.datetime.strptime(totalYM[predictorTimesteps - 1], "%Y-%m-%d %H:%M:%S")

            # if splitTime.time() == datetime.datetime.min.time():
            if splitTime.hour == 00:
                dates = [date_str[5:16] for date_str in totalYM]
                totalYM = np.array([date for date in dates])
                thisPredictorYMS = thisPredictorYMS[:]
                dates = [date_str[5:16] for date_str in thisPredictorYMS]
                thisPredictorYMS = np.array([date for date in dates])

                plt.figure(figsize=(30, 17))
                plt.plot(totalYM, np.vstack((thisPredictor.detach().cpu().numpy(), thisForecast.detach().cpu().numpy())), label= "original time series", color='blue', linewidth=5)
                plt.plot(thisPredictorYMS, thisCfPredictor, label= "counterfactual instance", color='red', linestyle='--', linewidth=5)
                plt.axvline(x = len(thisPredictor) - 1, color='gray', linestyle='--')
                plt.xticks(rotation = 45, fontsize=25)
                plt.ylabel('Average ' + X, fontsize = 35)
                plt.title('DiffusionCF based on VAE', fontsize=50)
                plt.gca().tick_params(axis='y', labelsize=30)
                plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(3))

                thisPrediction = thisPrediction.tolist()
                thisCfTarget = thisCfTarget.tolist()
                thisCfPrediction = thisCfPrediction.tolist()

                posi = len(thisPredictor) + np.argmax(thisForecast.detach().cpu().numpy())
                plt.scatter(totalYM[posi], thisPrediction * scalers['scale'] + scalers['mean'] ,  marker='o', color='blue', s=120)
                plt.scatter(totalYM[posi], thisCfTarget * scalers['scale'] + scalers['mean'], marker='o', color='red', s=120)

                plt.legend(loc = 'upper left', fontsize = 35)
                plt.show()

    def metricEval(self, thisPredictor, thisCfPredictor, thisCfTarget, thisCfPrediction):                               # All inputs should be converted to numpy array of 1-dimension when using this API
        thisRmse = (thisCfTarget - thisCfPrediction)**2
        thisProximity = (1 / predictorTimesteps) * np.sum(np.abs(thisPredictor - thisCfPredictor))

        # similarityList = []
        # e = 0.05
        # for i in range(predictorTimesteps):
        #     similarity = 1 if abs(thisPredictor[i] - thisCfPredictor[i]) < e else 0
        #     similarityList.append(similarity)
        # thisCompactness = (1 / (predictorTimesteps + 1)) * sum(similarityList)

        thisSparsity = 0

        for i in range(predictorTimesteps):
           if abs(thisPredictor[i] - thisCfPredictor[i]) > 0:
                thisSparsity += 1
        thisSparsity = thisSparsity/predictorTimesteps


        thisLnorm = np.linalg.norm(thisPredictor - thisCfPredictor, ord = np.inf)

        differences = np.abs(np.array(thisPredictor) - np.array(thisCfPredictor))
        unique_values, counts = np.unique(differences, return_counts=True)
        total_count = len(differences)
        probabilities = counts / total_count
        # thisEntropy = -np.sum(probabilities * np.log2(probabilities))

        thisSmooth = 0
        for i in range(predictorTimesteps-1):
            thisSmooth += abs(abs(thisPredictor[i] - thisCfPredictor[i])-abs(thisPredictor[i+1] - thisCfPredictor[i+1]))
        thisSmooth = thisSmooth / (predictorTimesteps-1)


        totalOne = 0
        deltaX = abs(thisPredictor - thisCfPredictor)
        for i in range(len(deltaX)):
            if deltaX[i] > rho:
                deltaX[i] = 1
                totalOne += 1
            else:
                deltaX[i] = 0
        max_length = 0
        current_length = 0
        for num in deltaX:
            if num == 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        if totalOne == 0:
            thisConsecutiveness = 0
        else:
            thisConsecutiveness = max_length / totalOne

        return thisRmse, thisProximity, thisSparsity, thisLnorm, thisSmooth, thisConsecutiveness
