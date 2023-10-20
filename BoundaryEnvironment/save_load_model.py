import os
import torch
import torch.nn as nn


def save_model(model, model_path):
    # Get the current working directory

    # if not os.path.exists(model_path):
    #    print('File Path: ', model_path, ' does not exist')
    #    return 1

        
    # Save the model to the specified file
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")


def load_model( model_path):
    if model_path!=None:
        if not os.path.exists(model_path):
            print('File Path: ', model_path, ' does not exist')
            return 1
        print('model_loaded')
        return torch.load(model_path)
