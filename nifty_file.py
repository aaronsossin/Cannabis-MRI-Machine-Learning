class nifty_file:
    def __init__(self):
        self.sub = None
        self.filenameBL = None
        self.pathBL = None
        self.filenameFU = None
        self.pathFU = None
        self.participant_info = None

    def set_participant_info(self, row):
        self.participant_info = row