
# step 1 get data
python prepare_data.py test
python prepare_data.py train
python prepare_data.py extra

# step 2 run train test split & save the split result for classification
python classification_train_test_split.py

# step 3 commented out as this step trains the classification model using gpu
# python train_classification_model.py

# step 4 plot result into figures
python get_classification_performance_plot.py

# step 5 get sequence accuracy on classification model
python get_sequence_accuracy.py

# step 6 commented out as this step trains detection model using gpu.
# python train_detection_model.py