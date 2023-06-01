import os
import dill
import random
import numpy as np
import tensorflow as tf
import h5py as h5
from deap import creator, base, gp, tools, algorithms
import operator
from image_analysis import homogeneity_evaluation
from GP_eval import eaSimple_checkpointing
from GP_controller import Controller

class EvalController(Controller):
    def __init__(self, predictor, input_path):
        super(EvalController, self).__init__(predictor, input_path)
        self.model = predictor
        self.input_path = input_path
        
    def evolution(self, pop, toolbox):
        # only 3 solutions required:
        hof = tools.HallOfFame(10)
        pop, log, hof = eaSimple_checkpointing(pop, toolbox, 0.85, 0.05, 1, halloffame=hof, verbose=False, checkpoint='GP_checkpoints/training_5.pkl', frequency=1)
        return hof, pop


if __name__ == "__main__":
    model = tf.keras.models.load_model("checkpoints/training_72/", compile=False)
    model.summary()
    data_path = '/home/itsnas/ueuua/BA/dataset/train/ueuua_Cu_fans_100_20_60_HS_0_0_25.h5'
        
    controller = EvalController(model, data_path)
    toolbox = controller.toolbox_creator(controller)
    population = controller.population_initializer(300, toolbox)
    hof, pop = controller.evolution(population, toolbox)
    
    for tree in hof:
        std, hs, fl = controller.evaluate(tree)
        print("Score: ", std, hs, fl)
        
