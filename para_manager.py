# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/10/2023
# ============================================================================

from pkg_manager import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
foldername = "D:\\researchCodes\\interpretableML\\code\\"

batchSize = 128     # for DeepExtrema
vae_batch_size = 128
vae_anomaly_detection_batch_size = 128

multiTimeSeries = 'univariate'  # 'univariate' or 'multivariate'
dataset = 'GSOD'         # todo: 'temp' as well as other ones with seasoning cycle should be standarlized by month/season. if with no seasoning cycle, we can do the current way
splitMethod = "byTime"      # 'byTime' or 'byDistribution'

if dataset == 'GSOD':
    area = 'Mobile_and_Pensacola_area'
    rootFolder = "D:\\researchCodes\\interpretableML\\data\\GSOD\\" + area
    climateIndexFolder = "D:\\researchCodes\\interpretableML\\data\\climateIndices\\"
    F = ['AO', 'NAO', 'NINO1+2', 'NINO3.4', 'NINO3', 'NINO4', 'PDO', 'PNA', 'SOI']      # ['AO', 'NAO', 'NINO1+2', 'NINO3.4', 'NINO3', 'NINO4', 'PDO', 'PNA', 'SOI']
    X = "PRCP"  # "PRCP", "TEMP", "GUST" (not good)
    if X == "PRCP":
        p = 0.51
        p_target_nex = 0.31
        p_target_ex = 0.71
    elif X == "TEMP":
        p = 0.49
        p_target_nex = 0.29
        p_target_ex = 0.69
    anomaly = 99.99  # 99.99 for "PRCP"; 999 for "GUST"; 150.0 for "TEMP"
    totalTimesteps = 24 + 6
    predictorTimesteps = 24
    cfWindowSizeList = [3, 6, 9, 12]  # for csdi, 'predictorTimesteps' is required.
    modelSavePath = multiTimeSeries + '_' + str(batchSize) + '_' + X + '_' + area + '_deepExtrema.pth'
    trainValSplitYear = datetime.datetime(2022, 1, 1)
    valTestSplitYear = datetime.datetime(2023, 1, 1)
elif dataset == 'Weather':
    rootFolder = "D:\\researchCodes\\interpretableML\\data\\weatherHistory.csv"
    X = "Temperature (C)"
    anomaly = 9999999999.99
    totalTimesteps = 16 + 8
    predictorTimesteps = 16
    cfWindowSizeList = [16]  # for csdi, 'predictorTimesteps' is required.
    modelSavePath = multiTimeSeries + '_' + str(batchSize) + '_' + X + '_' + dataset + '_deepExtrema.pth'
    trainValSplitYear = datetime.datetime(2015, 1, 1)
    valTestSplitYear = datetime.datetime(2016, 1, 1)
elif dataset == 'S&P500':
    rootFolder = "D:\\researchCodes\\interpretableML\\data\\S&P500.csv"
    X = "Close_Diff"
    p = 0.62
    p_target_nex = 0.32
    p_target_ex = 0.92
    anomaly = 9999999999
    totalTimesteps = 25 + 5
    predictorTimesteps = 25
    cfWindowSizeList = [5, 10, 15]  # for csdi, 'predictorTimesteps' is required.
    modelSavePath = multiTimeSeries + '_' + str(batchSize) + '_' + X + '_' + dataset + '_deepExtrema.pth'
    trainValSplitYear = datetime.datetime(2022, 1, 13)
    valTestSplitYear = datetime.datetime(2023, 1, 13)
elif dataset == 'dodgersLoopSensor':
    rootFolder = "D:\\researchCodes\\interpretableML\\data\\dodgersLoopSensor.csv"
    X = "Flow"
    p = 0.60
    p_target_nex = 0.30
    p_target_ex = 0.80
    anomaly = 9999999999
    totalTimesteps = 40 + 8
    predictorTimesteps = 40
    cfWindowSizeList = [5, 10, 15, 20]
    modelSavePath = multiTimeSeries + '_' + str(batchSize) + '_' + X + '_' + dataset + '_deepExtrema.pth'
    trainValSplitYear = datetime.datetime(2005, 8, 1)
    valTestSplitYear = datetime.datetime(2005, 9, 1)

task = 'cf'      # ['pre-train-deepExtrema', 'pre-train-csdi', 'pre-train-vae', 'pre-train-vae-anomaly-detection', 'cf', 'evaCsvResul']
cfMethod = 'baseNN'   # ['csdi', 'benchmark', 'vae', 'baseNN', 'NGCF', 'SPARCE']
kCsdi = 5
benchmarkBaseline = 'wCF'    # ['ForecastCF', 'wCF']
ToleranceOfForecastCF = 0.05
forecastingModel = 'deepExtrema'
rho = 0.1

cfTargetType = 'GEV_distribution'   # 'expected_value' or 'prob_distribution' or 'GEV_distribution'
cfSituation = 'e2ne'    # 'e2ne' or 'ne2e'

lambdaVAE = 0.00000001     # For GSOD data, the KL loss and the reconstructed loss is with a high gap, so we gotta set lambdaVAE very very small to weight the KL loss more and more!
lambdaVAEAnomaly = 1.0

# cfWindowSize = 5
# missingWindowNum = 5
config ={'train':{'epochs': 50,        # for csdi
                  'batch_size': 128,
                  'lr': 1.0e-3,
                  'itr_per_epoch': 1.0e+8
                  },
         'diffusion':{'layers': 4,
                      'channels': 64,
                      'nheads': 8,
                      'diffusion_embedding_dim': 128,
                      'beta_start': 0.0001,
                      'beta_end': 0.5,
                      'num_steps': 50,
                      'schedule': "quad",
                      'is_linear': False},
         'model':{'is_unconditional': 0,
                  'timeemb': 128,
                  'featureemb': 16,
                  'target_strategy': "random"}
         }

# for SPARCE
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--target_class", type=int, default=0)
parser.add_argument("--max_batches", type=int, default=500)
parser.add_argument("--dataset", type=str, default="motionsense", help="motionsense, simulated")
parser.add_argument("--batchsize", type=int, default=batchSize)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--save_indicator", type=bool, default=False, help="False or True")
parser.add_argument("--lambda1", type=float, default=1.0, help="Weight of adversarial loss")
parser.add_argument("--lambda2", type=float, default=1.0, help="Weight of classification loss")
parser.add_argument("--lambda3", type=float, default=1.0, help="Weight of similarity loss")
parser.add_argument("--lambda4", type=float, default=1.0, help="Weight of sparsity loss")
parser.add_argument("--lambda5", type=float, default=1.0, help="Weight of jerk loss")
parser.add_argument("--freeze_features", type=list, default=[])
parser.add_argument("--seed", type=int, default=123, help='random seed for splitting data')
parser.add_argument("--num_reps", type=int, default=1, help='number of repetitions of experiments')
parser.add_argument("--max_iter", type=int, default=10)
parser.add_argument("--init_lambda", type=float, default=1.0)
parser.add_argument("--approach", type=str, default="sparce")
parser.add_argument("--save", type=bool, default=False, help="save experiment file, originals and cfs")
parser.add_argument("--max_lambda_steps", type=int, default=5)
parser.add_argument("--lambda_increase", type=float, default=0.001)

allYears = datetime.datetime(1000, 1, 1)    # the earliest starting year is no eariler than 1500. Setting 1000 let us count all of them
trainTestRatio = 0.9
randomState = 66
