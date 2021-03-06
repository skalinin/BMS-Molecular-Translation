model_config = {
    "attention_dim": 256,
    "embed_dim": 256,
    "decoder_dim": 512,
    "dropout2d": 0.1,
    "dropout": 0.5,
    "encoder_clip_grad_norm": 5,
    "decoder_clip_grad_norm": 5,
    "image_height": 288,
    "image_width": 288,
    "train_dataset_size": 0.95,
    "paths": {
        "tokenizer": "/workdir/data/processed/tokenizer.pth",
        "train_csv": "/workdir/data/processed/train_labels_processed.pkl",
        "val_csv": "/workdir/data/processed/val_labels_processed.pkl",
        "submission_csv": "/workdir/data/bms-molecular-translation/sample_submission.csv"
     }
 }
