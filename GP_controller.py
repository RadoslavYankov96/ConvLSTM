import numpy as np
import tensorflow as tf
import cv2
from deap import creator, base, gp, tools, algorithms
import operator
from image_analysis import statistical_homogeneity_score


class Controller:
    def __init__(self, predictor, input_path):
        self.model = predictor
        self.input_path = input_path

    def read_inputs(self):
        img = cv2.imread(self.input_path)
        return img

    def encode_inputs(self):
        imgs = self.read_inputs()
        input_sequence = self.model.encoder(imgs)
        input_vector = self.model.flat(input_sequence)
        return input_vector

    def prediction(self, imgs, fan_settings):
        prediction = self.model((imgs, fan_settings))
        return prediction

    @classmethod
    def create_primitives(cls) -> object:
        pset = gp.PrimitiveSet("MAIN", 40960)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(operator.neg, 1)
        return pset

    @classmethod
    def toolbox_creator(cls, self):
        pset = cls.create_primitives()

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -5.0))

        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self.evaluate)

        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))

        return toolbox

    def evaluate(self, individual):
        toolbox = self.toolbox_creator()
        func = toolbox.compile(expr=individual)
        tree_input = self.encode_inputs()
        fan_settings = func(tree_input)
        for i, fan in enumerate(fan_settings):
            fan_settings[i] = self.sigmoid(fan)
        prediction = self.prediction((tree_input, fan_settings))
        mean, std = statistical_homogeneity_score(prediction)

        return mean, std

    @staticmethod
    def population_initializer(n_individual, toolbox):
        pop = toolbox.population(n=n_individual)
        return pop

    @staticmethod
    def evolution(pop, toolbox):
        # only 1 solution is asked:
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.05, 10, halloffame=hof, verbose=False)
        return hof, pop

    @staticmethod
    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return 100*s


if __name__ == "__main__":
    model = tf.keras.models.load_model("saved_models//training_33//", compile=False)
