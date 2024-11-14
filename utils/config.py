MODEL_CONFIG = {
    'NHITS': {
        'start_padding_enabled': True,
        # 'accelerator': 'mps',
        'accelerator': 'cpu',
        'max_steps': 2000,
        'enable_checkpointing': True,
        'early_stop_patience_steps': 50,
    },
}
