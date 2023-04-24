import os
import random
import numpy as np
import tensorflow as tf
import h5py as h5
from deap import creator, base, gp, tools, algorithms
import operator
from image_analysis import statistical_homogeneity_score
from GP_eval import eaSimple_checkpointing


class Controller:
    def __init__(self, predictor, input_path):
        self.model = predictor
        self.input_path = input_path

    def read_inputs(self):
        imgs = []
        
        with h5.File(self.input_path, 'r') as experiment:
            imgs.append(np.expand_dims(np.array(experiment['frame 6'], dtype=np.float32), axis=-1))
            imgs.append(np.expand_dims(np.array(experiment['frame 7'], dtype=np.float32), axis=-1))
   
        input_sequence = np.expand_dims(np.stack(tuple(imgs)), axis=0)

        return input_sequence

    def encode_inputs(self):
        imgs = self.read_inputs()
        input_sequence = self.model.encoder(imgs)
        input_vector = self.model.flat(input_sequence)
        
        return input_vector.numpy()

    def decode_prediction(self, input_vector, fan_settings):
        x = np.concatenate((input_vector, fan_settings))
        x = np.expand_dims(x, axis=0)
        x = self.model.dense1(x)
        x = self.model.bn1(x)
        x = self.model.dense4(x)
        x = self.model.bn4(x)
        x = self.model.rs(x)
        x = self.model.decoder(x)

        return x

    def prediction(self, imgs, fan_settings):
        prediction = self.model((imgs, fan_settings))
        return prediction

    @staticmethod
    def create_list(input_1, input_2, input_3):
        return [input_1, input_2, input_3]

    @staticmethod  # this ks a decorator.
    def protected_div(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1.0

    @staticmethod
    def inverse(x):
        try:
            return 1/x
        except ZeroDivisionError:
            return 1.0

    @staticmethod
    def ignore(x):
        return x*0

    @staticmethod
    def add_3(a, b, c):
        return a + b + c

    @staticmethod
    def mul_3(a, b, c):
        return a*b*c

    @staticmethod
    def greater(left, right):
        if left > right:
            return left
        else:
            return right

    @staticmethod
    def lesser(left, right):
        if left < right:
            return left
        else:
            return right

    @classmethod
    def create_primitives(cls, self) -> object:
        pset = gp.PrimitiveSetTyped("MAIN", [float]*5120, list)
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(operator.neg, [float], float)
        pset.addPrimitive(self.create_list, [float, float, float], list)
        '''pset.addPrimitive(self.protected_div, [float, float], float)
        pset.addPrimitive(self.add_3, [float, float, float], float)
        pset.addPrimitive(self.mul_3, [float, float, float], float)
        pset.addPrimitive(self.inverse, [float], float)
        pset.addPrimitive(self.greater, [float, float], float)
        pset.addPrimitive(self.lesser, [float, float], float)'''


        '''pset.addEphemeralConstant(f"random_parameter_{random.uniform(-3,3)}", lambda: random.uniform(-1, 1), float)
        pset.addEphemeralConstant(f"random_parameter_{random.uniform(-3,3)}", lambda: random.uniform(-1, 1), float)
        pset.addEphemeralConstant(f"random_parameter_{random.uniform(-3,3)}", lambda: random.uniform(-1, 1), float)
        pset.addEphemeralConstant(f"random_parameter_{random.uniform(-3, 3)}", lambda: random.uniform(-1, 1), float)
        pset.addEphemeralConstant(f"random_parameter_{random.uniform(-3, 3)}", lambda: random.uniform(-1, 1), float)
        pset.addEphemeralConstant(f"random_parameter_{random.uniform(-3, 3)}", lambda: random.uniform(-1, 1), float)'''

        return pset

    @classmethod
    def toolbox_creator(cls, self):
        pset = cls.create_primitives(self)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=10, max_=25)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self.evaluate)

        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=20)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))

        return toolbox

    def evaluate(self, individual):
        toolbox = self.toolbox_creator(self)
        func = toolbox.compile(expr=individual)
        tree_input = np.squeeze(self.encode_inputs())
        fan_settings = func(*tree_input)
        fan_settings = self.sigmoid(fan_settings)
        print(np.array(fan_settings).astype(int))
        prediction = self.decode_prediction(tree_input, fan_settings)
        mean, std = statistical_homogeneity_score(prediction.numpy())

        return std,

    @staticmethod
    def population_initializer(n_individual, toolbox):
        pop = toolbox.population(n=n_individual)
        return pop

    @staticmethod
    def evolution(pop, toolbox):
        # only 1 solution is asked:
        hof = tools.HallOfFame(10)
        pop, log, hof = eaSimple_checkpointing(pop, toolbox, 0.65, 0.05, 10, halloffame=hof, verbose=False, checkpoint='GP_checkpoints/first_training.pkl')
        return hof, pop

    @staticmethod
    def sigmoid(x):
        for i, element in enumerate(x):
            x[i] = 1 / (1 + np.exp(-element))
        return np.multiply(x, 100)


if __name__ == "__main__":
    model = tf.keras.models.load_model("checkpoints/training_44/", compile=False)
    model.summary()
    data_path = '/home/itsnas/ueuua/BA/dataset/train'
    for experiment in os.listdir(data_path):
        print(experiment)
        img_path = os.path.join(data_path, experiment)
        
        controller = Controller(model, img_path)
        toolbox = controller.toolbox_creator(controller)
        population = controller.population_initializer(200, toolbox)
        hof, pop = controller.evolution(population, toolbox)
        
        for individual in hof:
            std = controller.evaluate(individual)
            print(std)
