import h5py
from operator import itemgetter
from GP_controller import Controller
import tensorflow as tf
import os
import time
import numpy as np
import dill
from score_generator import homogeneity_evaluation


class RTController(Controller):
    def __init__(self, model, input_path):
        super().__init__(model, input_path)
        self.input_path = input_path
        self.model = model

    def gen_file_names(self):
        files = sorted(os.listdir(self.input_path))
        if len(files) < 2:
            print("not enough experiments")
            pass
        else:
            files = files[-2:]
            return files

    def gen_sequence(self, files):
        imgs = []
        for f in files:
            file_path = os.path.join(self.input_path, f)
            with h5py.File(file_path, 'r') as file:
                img = np.array(file['D2_dataNorm'], dtype=np.float64)
            img = np.expand_dims(img, axis=-1)
            imgs.append(img)
        input_sequence = np.expand_dims(np.stack(tuple(imgs)), axis=0)

        return input_sequence

    def encode_inputs(self, enc_input):
        input_sequence = self.model.encoder(enc_input)
        input_vector = self.model.flat(input_sequence)
        input_vector = self.model.dense1(input_vector)
        input_vector = self.model.bn1(input_vector)

        return input_vector.numpy()

    @staticmethod
    def load_hof(checkpoint):
        with open(checkpoint, "rb") as cp_file:
            cp = dill.load(cp_file)
        hof = cp["halloffame"] #halloffame
        return hof

    def evaluate(self, individual, input_vector):
        toolbox = self.toolbox_creator(self)
        func = toolbox.compile(expr=individual)
        tree_input = np.squeeze(input_vector)
        #tree_input = list(tree_input)
        fan_settings = func(*tree_input)
        fan_settings_vector = self.sigmoid(fan_settings)
        fan_settings = np.tile(fan_settings_vector, 100)
        prediction = self.decode_prediction(tree_input, fan_settings)
        std, hs, fl = homogeneity_evaluation(prediction.numpy(), fan_settings)

        return fan_settings_vector, 5*std + 5*hs + fl


def run_labview(model_path, input_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    controller = RTController(model, input_path)
    file_names = ["file1", "file2"]
    keep_going = True
    hof = controller.load_hof("GP_populations//medium_specialist_11.pkl")
    #tree = hof[5]
    counter = 1
    while keep_going:
        file_sequence = controller.gen_file_names()
        if file_sequence:
            if file_sequence == file_names:
                print("no new image")
                continue
            else:
                time.sleep(0.3)
                input_sequence = controller.gen_sequence(file_sequence)
                input_vector = controller.encode_inputs(input_sequence)
                #fan_settings = controller.evaluate(tree, input_vector)
                solutions = []
                scores = []
                for tree in hof:
                    fan_settings, score = controller.evaluate(tree, input_vector)
                    scores.append(score)
                    solutions.append(fan_settings)
                index = min(enumerate(scores), key=itemgetter(1))[0]
                fan_settings = np.multiply(solutions[index], 100).astype(int)
                #fan_settings = np.multiply(fan_settings, 100).astype(int)
                file_names = file_sequence
                np.save(f"fan_settings//{counter:03}.npy", fan_settings)
                counter = counter + 1
                yield fan_settings


if __name__ == "__main__":
    g = run_labview('TF_models//training_72//', "H://public//images//GP//medium_10//specialist_2//")
    for i in range(12):
        print(next(g))
