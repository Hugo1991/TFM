class node_values:
    def __init__(self, iden, value):
        self.iden = iden
        self.value = value

    def get_iden(self):
        return self.iden

    def set_iden(self, iden):
        self.iden = iden

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def __str__(self):
        iden = str(self.iden)
        value = str(self.value)
        return iden + ':' + value
