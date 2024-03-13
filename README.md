# Predictive Modeling and Interpretability Techniques for SoC Performance with Machine Learning

## Abstract
This project presents a machine learning-based approach to predict System-on-Chip (SoC) performance using SPEC 2017 benchmark results. Our goal was to develop a high-accuracy, compact feedforward machine learning model that achieves a harmonious balance of accuracy, compactness, and low latency, with impressive R² scores of 0.98. Through the use of autoencoders, we analyzed and understood the decision-making processes of our models. Additionally, we employed feature selection and SHAP (SHapley Additive exPlanations) to enhance the transparency of our models.

## Introduction
In the realm of hardware design and deployment, performance benchmarks are essential for making informed decisions. Traditional benchmarking methods often fall short in predicting performance for unseen hardware configurations. This project seeks to overcome these limitations by employing a novel machine-learning approach for performance prediction, focusing on the processed SPEC CPU 2017 dataset.

## Dataset Processing
The SPEC 2017 benchmark dataset, preprocessed in previous work, serves as the foundation for our project. The preprocessing aimed to optimize the dataset for model training by cleaning, normalizing, and removing outliers to ensure the consistency and reliability of the performance predictions. This use of a preprocessed dataset allowed us to concentrate on the development and evaluation of our machine-learning models for SoC performance prediction.

For detailed information on the dataset's preprocessing methods and initial analysis, please refer to the original work by the authors in the publication/repository:
Cengiz, M., Forshaw, M., Atapour-Abarghouei, A., & McGough, A. S. (2023). Predicting the performance of a computing system with deep networks. In Proceedings of the 2023 ACM/SPEC International Conference on Performance Engineering (pp. 91–98).
https://github.com/cengizmehmet/BenchmarkNets

## Methodology
- **Dataset Processing:** Utilizes the preprocessed SPEC 2017 benchmark dataset for model training.
- **Model Design:** Develops a compact, accurate Multi-Layer Perceptron (MLP) model integrated into an autoencoder framework for interpretability.
- **Hyperparameter Tuning:** Explores various configurations to enhance the model's accuracy and efficiency.
- **Interpretability Techniques:** Applies SHAP and feature selection to elucidate the model's decision-making process, enhancing transparency and reliability.

## Results
Our project demonstrates that the proposed MLP model, despite its compact size, competes favorably with more complex models in terms of accuracy while offering significant advantages in efficiency. The application of SHAP and feature selection corroborates the insights provided by the autoencoder, offering a comprehensive view of the model's predictive dynamics.

## Conclusion
This work highlights the potential of machine learning models for SoC performance prediction and underscores the importance of interpretability in machine learning. By balancing predictive accuracy with interpretability techniques, it paves the way for more transparent and trustworthy AI solutions in the realm of hardware performance prediction.


## Dependencies
- Python 3.x
- TensorFlow
- Pandas
- Numpy
- SHAP
- Scikit-learn

## Authors
- Ghazi Ben Henia
- Abdoulaye Gamatié

## Acknowledgments
- LIRMM Univ. Montpellier, Montpellier, France

For more detailed information about the methodology, results, and conclusions, please refer to the [full paper](https://github.com/GhaziBenHenia/Predictive_Modeling_and_Interpretability_Techniques_for_SoC_Performance_with_Machine_Learning/blob/main/Predictive_Modeling_and_Interpretability_Techniques_for_SoC_Performance_with_Machine_Learning.ipynb).
