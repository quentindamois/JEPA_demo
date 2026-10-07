

class individual_tracker:
    def __init__(self):
        self.dict_loss = dict()

    def get(self, index):
        return self.dict_loss.get(index, 0)
    def inc(self, index, value):
        self.dict_loss[index] = self.dict_loss.get(index, 0) + value.item()
        return value
    
    def display_loss(self):
        return ", ".join([f"{item[0]}: {item[1]}" for item in self.dict_loss.items()])
    def res_loss(self):
        for key in self.dict_loss.keys():
            self.dict_loss[key] = 0