from A import *
from B import *

def main():
    """
    Main function to execute a series of model tests and print their outputs.
    This function performs the following steps:
    1. Initializes an empty list to store outputs.
    2. Appends the result of testing the 'ResNet18' model on the 'breastmnist' dataset.
    3. Appends the result of testing the 'ViT' model on the 'breastmnist' dataset.
    4. Appends the result of testing the 'ResNet18' model on the 'bloodmnist' dataset.
    5. Appends the result of testing the 'ViT' model on the 'bloodmnist' dataset.
    6. Prints each output in the list.
    Returns:
        int: Always returns 0.
    """
    output_list = []

    output_list.append(A_main.A_main_function(BATCH_SIZE=8,
                                            train_or_test='test', 
                                            train_model='ResNet18', 
                                            data_flag='breastmnist',
                                            model_path='ResNet18_2024-12-25 15:09:51.435892'))
    
    output_list.append(A_main.A_main_function(BATCH_SIZE=8,
                                            train_or_test='test', 
                                            train_model='ViT', 
                                            data_flag='breastmnist',
                                            model_path='ViT_2024-12-25 15:12:32.084710'))
    
    output_list.append(A_main.A_main_function(BATCH_SIZE=8,
                                            train_or_test='test', 
                                            train_model='SVM', 
                                            data_flag='breastmnist',
                                            model_path='SVM_2024-12-25 17:47:43.290416'))

    output_list.append(B_main.B_main_function(BATCH_SIZE=128,
                                            train_or_test='test', 
                                            train_model='ResNet18', 
                                            data_flag='bloodmnist',
                                            model_path='ResNet18_2024-12-25 15:20:01.738181'))
    
    output_list.append(B_main.B_main_function(BATCH_SIZE=128,
                                            train_or_test='test', 
                                            train_model='ViT', 
                                            data_flag='bloodmnist',
                                            model_path='ViT_2024-12-25 15:36:47.570820'))
    
    output_list.append(B_main.B_main_function(BATCH_SIZE=128,
                                            train_or_test='test', 
                                            train_model='RF', 
                                            data_flag='bloodmnist',
                                            model_path='RF_2024-12-25 18:10:31.138830'))
    
    for output in output_list:
        print(output)
    
    return 0

if __name__ == "__main__":
    main()