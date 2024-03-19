import random
import json


def exp(x):
    return 2.718281 ** x


class NeuralNetworkErrors:
    class ZeroInputNeuron(Exception):
        def __init__(self):
            super().__init__("Кол-во нейронов первого слоя не может быть менее 1")

    class ZeroOutputNeuron(Exception):
        def __init__(self):
            super().__init__("Кол-во нейронов последнего слоя не может быть менее 1")


class Activation:
    class Sigmoid:
        @staticmethod
        def function(x):
            return 1 / (1 + exp(-x))

        @staticmethod
        def derivative(x):
            return x * (1 - x)

        @staticmethod
        def generate_weights():
            return random.random() * 2 - 1

    class Tanh:
        @staticmethod
        def function(x):
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        @staticmethod
        def derivative(x):
            return 1 - x ** 2

        @staticmethod
        def generate_weights():
            return random.random() * 2 - 1

    class ReLU:
        @staticmethod
        def function(x):
            return max(0, x)

        @staticmethod
        def derivative(x):
            return 1 if x >= 0 else 0

        @staticmethod
        def generate_weights():
            return random.random()

    class LeakyReLU:
        @staticmethod
        def function(x):
            return max(0.01 * x, x)

        @staticmethod
        def derivative(x):
            return 1 if x >= 0 else 0.01

        @staticmethod
        def generate_weights():
            return random.random()


class Neuron:
    def __init__(self, is_bias=False):
        self.out = int(is_bias)
        self.delta = 0
        self.is_bias = is_bias

    def __str__(self):
        return f"Neuron out: {self.out} | delta: {self.delta} | bias: {self.is_bias}"


class NeuralNetwork:
    @staticmethod
    def create_by_structure(structure):
        layers = structure["l"]
        input_count = layers[0]
        hidden_layers = layers[1:-1]
        output_count = layers[-1]

        use_bias = structure.get("b")
        return NeuralNetwork(input_count, hidden_layers, output_count, use_bias=use_bias)

    @staticmethod
    def load(name="nn.sw"):
        with open(name, "r") as file:
            data = file.read()

        data = json.loads(data)
        neural_network = NeuralNetwork.create_by_structure(data["s"])
        neural_network.w = data["w"]
        return neural_network

    def __init__(self, input_count, hidden_layers_count, output_count, learn_rate=0.1, use_bias=True, moment=0,
                 activation=Activation.Sigmoid):

        self.activation = activation
        self.w = []
        self.last_delta_w = []

        if input_count < 1:
            raise NeuralNetworkErrors.ZeroInputNeuron

        if output_count < 1:
            raise NeuralNetworkErrors.ZeroOutputNeuron

        self.learn_rate = learn_rate
        self.moment = moment
        self.layers = [input_count] + hidden_layers_count + [output_count]
        self.use_bias = use_bias

        # Инициализация внутренних слоев
        hidden_layers = []
        for layer, count_in_layer in enumerate(hidden_layers_count):
            if count_in_layer < 1:
                print("[WARNING] Один из скрытых слоев имеет 0 нейронов")
                continue

            hidden_layers.append([])
            for _ in range(count_in_layer):
                hidden_layers[layer].append(Neuron())

            if self.use_bias:
                hidden_layers[layer].append(Neuron(True))

        # Массив нейронов
        self.array = [
            [Neuron() for _ in range(input_count)] +
            ([Neuron(True)] if self.use_bias else [])] + \
            hidden_layers + [[Neuron() for _ in range(output_count)]]

        # Установка начальных значений весов в соответствии с функцией активации
        self.reset_weights()

    def reset_weights(self):
        # Инициализация весов
        self.w = []
        for i in range(len(self.array) - 1):
            self.w.append([])
            for j in range(len(self.array[i])):
                self.w[i].append([])
                for k in range(len(self.array[i+1])):
                    self.w[i][j].append(self.activation.generate_weights())

        # Инициализация последнего изменения веса (для момента Нестерова)
        self.last_delta_w = []
        for i in range(len(self.array) - 1):
            self.last_delta_w.append([])
            for j in range(len(self.array[i])):
                self.last_delta_w[i].append([])
                for k in range(len(self.array[i+1])):
                    self.last_delta_w[i][j].append(0)

    def predict(self, inputs, softmax=False):
        # Проверка на соответствие кол-ва переданных входных значений с кол-вом входных нейронов

        if len(inputs) != len(self.array[0][:-1]):
            print(f"[WARNING] Кол-во переданных для обучения значений не совпадает с кол-вом входных нейронов." + \
                  f"Передано: {len(inputs)} / Кол-во выходных нейронов: {len(self.array[0][:-1])}")

        # Вставка входных данных
        for neurons, inp in zip(self.array[0][:-1], inputs):
            neurons.out = inp

        # Расчет суммы и активация
        for i, layer in enumerate(self.array[1:]): # i = i+1
            for j, neuron in enumerate(layer):
                if neuron.is_bias:
                    continue

                s = sum([lastNeuron.out * self.w[i][k][j] for k, lastNeuron in enumerate(self.array[i])])
                neuron.out = self.activation.function(s)

        # Нормализация выходов, возврат их процентного соотношения
        if softmax:
            s = sum([neuron.out for neuron in self.array[-1]])
            return [(neuron.out / s if s != 0 else 0)  for neuron in self.array[-1]]

        # Возврат не нормализованных выходных значений
        return [neuron.out for neuron in self.array[-1]]

    def learn_record(self, data, out):
        res = self.predict(data)
        errors = []

        # Если кол-во ожидаемых выходных значений не совпадает с фактическим кол-вом выходных нейронов
        if len(out) != len(res):
            print("[WARNING] Кол-во ожидаемых выходных значений не совпадает с кол-вом выходных нейронов")

        # Поиск градиентов последнего слоя
        for res_i, out_i, neuron_i in zip(res, out, self.array[-1]):
            errors.append((out_i - res_i) ** 2)
            neuron_i.delta = (out_i - res_i) * self.activation.derivative(res_i)

        # Поиск градиентов других слоев
        for i, _ in enumerate(self.array[1:-1]):
            i = len(self.array) - 2 - i
            for j, maij in enumerate(self.array[i]):
                s = 0
                for k, mik in enumerate(self.array[i+1]):
                    s += self.w[i][j][k] * mik.delta

                delta = self.activation.derivative(maij.out) * s
                maij.delta = delta

        # Коррекция весов
        for i, wi in enumerate(self.w):
            for j, wij in enumerate(wi):
                for k, wijk in enumerate(wij):
                    moment = self.last_delta_w[i][j][k] * self.moment
                    deltaW = self.learn_rate * self.array[i][j].out * self.array[i+1][k].delta + moment

                    self.w[i][j][k] += deltaW
                    self.last_delta_w[i][j][k] = deltaW

        return errors

    def learn_set(self, dataset):
        mse_list = []

        for one_set in dataset:
            mse_list += self.learn_record(one_set[0], one_set[1])

        return (1 / len(mse_list)) * sum(mse_list)

    def learn_by_iterations(self, dataset, iterations):
        """Обучение нейросети по кол-ву итераций"""
        return [self.learn_set(dataset) for _ in range(iterations)]

    def learn_by_error(self, dataset, max_error, max_iterations=10_000):
        """Обучение нейросети до определённого значения ошибки"""
        err = float("inf")
        errs = []

        while err > max_error:
            err = self.learn_set(dataset)
            errs.append([err])

            if len(errs) > max_iterations:
                break
        else:
            return errs

        print("[WARNING] Производится регенерация начальных весов. Достигнуто максимальное кол-во итераций.")

        self.reset_weights()
        return self.learn_by_error(dataset, max_error, max_iterations)

    def save(self, name="nn.sw"):
        """"Сохранение нейросети в файл"""
        structure = {
            "l": self.layers,
            "b": self.use_bias,
        }

        saving_data = {
            "s": structure,
            "w": self.w
        }

        with open(name, "w") as file:
            file.write(json.dumps(saving_data))
