import pygame.display

import NeuralNetwork


class Point:
    def __init__(self, pos, point_type):
        self.pos = pos
        self.type = point_type


class Pixel:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color


class App:
    def __init__(self):
        self.downscale_rate = 16
        self.size = (1280 // self.downscale_rate, 720 // self.downscale_rate)
        self.upscale_size = (1280, 720)
        self.screen = pygame.display.set_mode(self.upscale_size)
        self.surface = pygame.Surface(self.size)

        self.points = []
        self.enable = False

        self.nn = NeuralNetwork.NeuralNetwork(2, [10, 10], 1, 1, True, 0)

        self.predicted_pixels = [[Pixel((j, i), 0) for j in range(self.size[0])] for i in range(self.size[1])]

    def loop(self):
        while self.enable:
            self.events()
            self.draw()

    def add_point(self, pos, point_type):
        self.points.append(Point(pos, point_type))
        # self.learn()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.add_point(event.pos, 0)

                elif event.button == 3:
                    self.add_point(event.pos, 1)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.learn()

                elif event.key == pygame.K_2:
                    self.predict()

    def normalize_coord_points(self, pos):
        return [pos[0] / self.upscale_size[0], pos[1] / self.upscale_size[1]]

    def normalize_coord_pixels(self, pos):
        return [pos[0] / self.size[0], pos[1] / self.size[1]]

    def learn(self):
        for _ in range(10):
            for point in self.points:
                self.nn.learn_record(self.normalize_coord_points(point.pos), [point.type])

        print("Learned!")

        self.predict()

    def predict(self):
        for line in self.predicted_pixels:
            for pixel in line:
                pixel.color = self.nn.predict(self.normalize_coord_pixels(pixel.pos))[0]

        print("Predicted!")

        self.draw_predict()

    def draw_predict(self):
        for line in self.predicted_pixels:
            for pixel in line:
                self.surface.set_at(pixel.pos, (0, 0, int(255 * pixel.color)))

    def draw(self):
        self.learn()

        self.screen.fill((0, 0, 0))

        self.screen.blit(pygame.transform.scale(self.surface, self.upscale_size), (0, 0))

        for point in self.points:
            if point.type == 1:
                color = (255, 0, 0)

            else:
                color = (0, 255, 0)

            pygame.draw.circle(self.screen, color, point.pos, 10)

        pygame.display.update()

    def start(self):
        if self.enable:
            raise Exception("Already running")

        self.enable = True
        self.loop()

    def stop(self):
        self.enable = False


if __name__ == "__main__":
    app = App()
    app.start()
