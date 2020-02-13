from keras.models import load_model
import numpy as np

model=load_model("model_toy.h5")

def run_predictions(input_name):
    in_a = np.load(input_name+".npy")
    out_a = model.predict(np.transpose(in_a,(0,2,1)))
    np.save(input_name+"_out.npy",out_a)


run_predictions("test_toy")
run_predictions("test_toy_random")
