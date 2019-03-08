class LogWriter():
    def __init__(self, log_file):
        self.log_file = log_file
        
    def write(self, text, verbose):
        print(text, file=self.log_file)
        self.log_file.flush()
        if verbose:
            print(text)