
class Config:
    GAMMA = 0.9
    INITIAL_EPS = 0.5
    FINAL_EPS = 0.05
    BATCH_SIZE = 128
    LEARNING_RATE = 0.002
    PRETRAIN_STEPS = 500
    DEMO_MODE = True
    # NUM_EPISODES = 5000
    MEMORY_SIZE = 80000
    BURN_IN_SIZE = 15000
    JE_IF_DIFFER = 0.8
    DEMO_SIZE = 100000
    PRIORITIZED = True
    UPDATE_TARGET_FREQ = 5000
    LAMBDAS = [0.0, 1.0, 1E-5]
    DEMO_BONUS = 0.0
    EVAL_FREQ = 100
    N_STEP = 3
    DEMO_FILE = "/Users/nickwang/Documents/Programs/python/Mario-AI-Competition-2009/demo.npy"
