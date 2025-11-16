# ML Model Version Management

## Directory Structure

```
models/
├── production/          # Active production model
│   ├── model.pth
│   ├── encoders.pkl
│   └── metadata.json
├── staging/            # Testing environment for new models
│   ├── model.pth
│   ├── encoders.pkl
│   └── metadata.json
├── archive/            # Historical versions
│   ├── v1_2024-10-15/
│   ├── v2_2024-11-01/
│   ├── v3_2024-11-10/
│   └── v4_2024-11-12/
└── README.md
```

## Usage

### 1. List Available Versions
```bash
python scripts/switch_model_version.py --list
```

### 2. View Version Details
```bash
python scripts/switch_model_version.py --info production
python scripts/switch_model_version.py --info staging
python scripts/switch_model_version.py --info archive/v3_2024-11-10
```

### 3. Switch Model Version
```bash
# Dry run (preview changes)
python scripts/switch_model_version.py --env production --dry-run

# Apply changes
python scripts/switch_model_version.py --env production
python scripts/switch_model_version.py --env staging
python scripts/switch_model_version.py --env archive/v3_2024-11-10
```

## Environment Variables

Set `MODEL_ENVIRONMENT` in `.env` file:
```
MODEL_ENVIRONMENT=production  # Default
# Or: staging
# Or: archive/v3_2024-11-10
```

## Configuration in Code

The `config/settings.py` provides properties:
- `active_model_path`: Returns the path to the active model
- `active_encoder_path`: Returns the path to the active encoder
- `model_metadata_path`: Returns the path to metadata.json

## Deployment Workflow

1. **Development**: Train and validate new model
2. **Staging**: Deploy to `models/staging/` for testing
3. **Validation**: Test staging model thoroughly
4. **Archive Current**: Move current production to archive
5. **Promote**: Move staging model to production
6. **Update .env**: Set `MODEL_ENVIRONMENT=production`
7. **Restart**: Restart application

## Metadata Format

Each model version includes a `metadata.json`:

```json
{
  "version": "v4_no_leakage",
  "model_name": "model.pth",
  "encoder_name": "encoders.pkl",
  "created_date": "2024-11-15",
  "deployed_date": "2024-11-15",
  "description": "Model description",
  "metrics": {
    "final_val_loss": 89.43,
    "final_val_mae": 5.74,
    "training_epochs": 59,
    "best_epoch": 43
  },
  "model_architecture": {
    "type": "HybridRecommender",
    "features": ["face_shape", "gender", "age", "skin_tone", "user_preferences"],
    "embedding_dim": 384,
    "sentence_transformer": "paraphrase-multilingual-MiniLM-L12-v2"
  },
  "changelog": ["Change 1", "Change 2"],
  "status": "active"
}
```
