class Player:
    kills = 0
    deaths = 0
    self = 0

    def __init__(self, name):
        self.name = name
        self.awards = []

    def print_information(self):
        print("Stats for player " + str(self.name))
        print("Awards:")
        for award in self.awards:
            print("  " + award)
        print("Kills: " + str(self.kills))
        print("Deaths: " + str(self.deaths))
        print("Self: " + str(self.self))
        print(" ")
