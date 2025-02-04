class Config:
    # Data parameters
    DATA_ROOT = "data"
    IMG_SIZE = (64, 512)  # Standard size for iris images
    BATCH_SIZE = 32
    
    # Model parameters
    EMBEDDING_DIM = 128
    MARGIN = 1.0  # Contrastive loss margin
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    TRAIN_SPLIT = 0.8
    
    # MBSIF parameters
    SCALES = [3, 5, 7]  # Filter scales
    NUM_FILTERS = 8 