# Unraveling Block Maxima Forecasting Models with Counterfactual Explanation
**Authors:** Yue Deng, Asadullah Hill Galib, Pang-Ning Tan, Lifeng Luo

**Paper accepted to KDD 2024 Research Track:** https://dl.acm.org/doi/10.1145/3637528.3671923.

# Abstract
Disease surveillance, traffic management, and weather forecasting are some of the key applications that could benefit from block maxima forecasting of a time series as the extreme block maxima values often signify events of critical importance, such as disease outbreaks, traffic gridlock, and severe weather conditions. As the use of deep neural network models for block maxima forecasting increases, so does the need for explainable AI methods that could unravel the inner workings of such black box models. To fill this need, this paper presents a novel counterfactual explanation framework for block maxima forecasting models. Unlike existing methods, our proposed framework, DiffusionCF, combines deep anomaly detection with a conditional diffusion model to identify unusual patterns in the time series that could help explain the forecasted extreme block maxima. Experimental results on several real-world datasets demonstrate the superiority of DiffusionCF over other baseline methods when evaluated according to various metrics, particularly their informativeness and closeness.


<img src="https://github.com/yue2023cs/DiffusionCF/blob/main/DiffusionCF_architecture.png" width="666"/>

# Usage
The three **datasets** used in this work are: `S&P500.csv`, `Dogers/*`, and `Mobile_and_Pensacola_area/*`. All necessary **dependencies** to run the code are specified in `pkg_manager.py`. The key **hyperparameters** for both our method and the baseline methods are documented in `para_manager.py`.

`main.py` serves as the **driver** script for the entire codebase:
1. Loads the `dataProcessor` object from `data_processor.py` for data preprocessing, including the functions `readFolder`, `splitTrainValTest`, and `splitPredictorTarget`.
2. Supports pre-training tasks such as `pre-train-deepExtrema` (based on `deepExtrema.py`), `pre-train-csdi` (based on `CSDI_train.py` and `CSDI.py`), `pre-train-vae` (based on `VAE.py`), and `pre-train-vae-anomaly-detection` (based on `VAE_anomaly_detection.py`).
3. Implements the task of `cf`.

Most importantly, the `cf` task drives the core components for implementing counterfactual explanation approaches:
1. It loads the `cfLearner` object from `counterfactual_explanation.py`, which:
   - Constructs counterfactual targets using the `getCfTarget` function.
   - Learns counterfactual predictors for the baseline methods `wCF` and `ForecastCF` using the `getCfPredictors` function.
   - Visualizes results (e.g., as shown in Figure 4) using the `plot` function.
2. It implements our proposed method, `DiffusionCF`, when the imputation approach is set to `VAE` (`if cfMethod == 'vae'`) or `CSDI` (`elif cfMethod == 'csdi'`). Additionally, it supports several baseline methods:
   
   - `NGCF` (`elif cfMethod == 'NGCF'`, based on `NativeGuide.py`)
   - `baseNN` (`elif cfMethod == 'baseNN'`, based on `BaseNN.py`)
   - `wCF` and `ForecastCF` (`elif cfMethod == 'benchmark'`, based on `counterfactual_explanation.py`)
   - `SPARCE` (`elif cfMethod == 'SPARCE'`, based on `gan_models.py` and `loss_functions.py`)

   Each of these methods follows a similar pipeline consisting of the following steps:

   - Extract the `endDate` recorded in the dataset.
   - Select test samples using the condition `if (endDate > valTestSplitYear)`.
   - For each test sample, generate the counterfactual predictor `thisCfPredictor` based on the original predictor `thisPredictor` and target `thisCfTarget`, both retrieved from the `dataProcessor` object (which also serves as a container).
   - Use the loaded `deepExtremaModel` to compute the counterfactual prediction and associated parameters:
     ```python
     mu, sigma, xi_p, xi_n, thisCfPrediction = deepExtremaModel(thisCfPredictor.to(device), y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix)
     ```
   - During the iterations, save the `bestCfPredictor` and its corresponding `bestCfPrediction`.

  3. Calculate and save the evaluation metrics based on the implementation results for each sample:
     ```python
     thisRmse, thisProximity, thisSparsity, thisLnorm, thisSmooth, thisConsecutiveness = cfLearner.metricEval(thisPredictor,
                                                                                                              bestCfPredictor[0].detach().cpu().numpy().flatten(),
                                                                                                              dataProcessor.allDataCfTargets[i].detach().cpu().numpy(),
                                                                                                              bestCfPrediction[0].detach().cpu().numpy())
     ```
  4. Plot and save the implementation results for each sample:
     ```python
     cfLearner.plot(dataProcessor.scalers,
                    dataProcessor.allDataPredictors[i],
                    dataProcessor.allDataPredictorsYMS[i],
                    bestCfPredictor[0],
                    dataProcessor.allDataForecast[i],
                    dataProcessor.allDataForecastYMS[i],
                    dataProcessor.allDataPredictions[i],
                    dataProcessor.allDataCfTargets[i],
                    bestCfPrediction[0])
     ```
   
Finally, the average results across all samples are summarized.

# Citation
If you find our work relevant to your research, please cite:
```bibtex
@inproceedings{deng2024unraveling,
  title={Unraveling Block Maxima Forecasting Models with Counterfactual Explanation},
  author={Deng, Yue and Galib, Asadullah Hill and Tan, Pang-Ning and Luo, Lifeng},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={562--573},
  year={2024}
}
```
