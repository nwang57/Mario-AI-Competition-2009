
class Config:
    GAMMA = 0.99
    INITIAL_EPS = 0.5
    FINAL_EPS = 0.05
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    PRETRAIN_STEPS = 20000
    NUM_EPISODES = 5000
    MEMORY_SIZE = 50000
    BURN_IN_SIZE = 10000
    JE_IF_DIFFER = 0.8
    DEMO_SIZE = 100000
    PRIORITIZED = True
    UPDATE_TARGET_FREQ = 1000
    LAMBDAS = [0.0, 1.0, 1E-5]
    DEMO_BONUS = 1.0
    EVAL_FREQ = 100
    N_STEP = 3
    DEMO_FILE = "gg"
