SARang/
├── README.md
├── LICENSE
├── Makefile
├── .gitignore
├── .env.example
├── CHANGELOG.md
├── docker-compose.yml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_CARD.md
│   ├── MODEL_CARD.md
│   ├── DECISION_LOG.md
│   ├── EVALUATION.md
│   ├── LIMITATIONS.md
│   ├── DEMO_SCRIPT.md
│   └── JUDGE_QNA.md
├── configs/
│   └── default.yaml
├── backend/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── common.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── preprocess.py
│   │   │   ├── inference.py
│   │   │   ├── insight_modes.py
│   │   │   ├── explain.py
│   │   │   ├── export.py
│   │   │   └── audit.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── sarangnet.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── files.py
│   ├── training/
│   │   ├── dataset.py
│   │   └── train.py
│   └── tests/
│       ├── __init__.py
│       ├── test_preprocess.py
│       └── test_api.py
├── frontend/
│   ├── package.json
│   ├── Dockerfile
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api.ts
│       ├── styles.css
│       └── components/
│           ├── FileDropzone.tsx
│           ├── BeforeAfterSlider.tsx
│           ├── ModeSelector.tsx
│           ├── ConfidenceOverlay.tsx
│           └── ExportPanel.tsx
├── scripts/
│   ├── make_demo_data.py
│   ├── prepare_sentinel_pairs.md
│   └── run_local_demo.sh
├── evaluation/
│   ├── evaluate.py
│   └── metrics_template.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_visualization.ipynb
│   └── 03_model_evaluation.ipynb
└── deployment/
    ├── production_notes.md
    └── offline_kiosk.md
