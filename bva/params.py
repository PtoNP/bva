NB_FRAMES = 12

CLASSES = ['clear', 'drive', 'drop', 'full_smash', 'half_smash', 'lift',
       'lob', 'long_def', 'long_serve', 'net_kill', 'net_shot',
       'short_def', 'short_serve']

COURT_WIDTH = 670
COURT_HEIGHT = 1340

TRACKNET_ERROR_THRESHOLD = 100

FINAL_PREDICT_PROBA_THRESHOLD = 0.85
FINAL_PREDICT_MIN_FRAMES_BEFORE_NEXT_HIT = 8
MIN_FRAMES_FOR_HIT = 2
SHOW_INFO = True

HITNET_MODEL = "hitnet_mirror2"
