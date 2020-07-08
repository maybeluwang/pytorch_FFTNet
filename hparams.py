class hparams:
    seed = 0
    ################################
    # Audio                        #
    ################################
    num_mels = 80
    num_freq = 1025
    sample_rate = 22050
    frame_shift = 200
    frame_length = 800
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20
    gl_iters = 100
    power = 2
    seg_l = 16000
