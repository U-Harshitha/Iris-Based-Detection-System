class Config:
    # Data parameters
    DATA_ROOT = "data/eye_diseases"
    PROCESSED_DATA_ROOT = "data/processed"
    IMG_SIZE = (224, 224)  # Standard size for eye disease images
    BATCH_SIZE = 32
    
    # Model parameters
    EMBEDDING_DIM = 128
    NUM_CLASSES = None  # Will be set dynamically based on disease folders
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    TRAIN_SPLIT = 0.8
    VALIDATION_SPLIT = 0.1
    
    # MBSIF parameters
    SCALES = [3, 5, 7]  # Filter scales
    NUM_FILTERS = 8 