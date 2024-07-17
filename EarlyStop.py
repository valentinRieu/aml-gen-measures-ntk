class EarlyStop:
    def __init__(self, patience = 1, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_error = float('inf')

    
    def step(self, validation_error):
        if validation_error < self.min_validation_error:
            self.min_validation_error = validation_error
            self.counter = 0
            return False
        
        if validation_error > (self.min_validation_error + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False