class Ant:
    def __init__(self, start_city_index: int):
        self.start_city_index = start_city_index
        self.current_city_index = start_city_index
        self.visited_cities_indices = [start_city_index]

    def move(self, destination_city_index: int) -> None:
        self.current_city_index = destination_city_index
        self.visited_cities_indices.append(destination_city_index)

    def reset(self) -> None:
        self.current_city_index = self.start_city_index
        self.visited_cities_indices = [self.start_city_index]