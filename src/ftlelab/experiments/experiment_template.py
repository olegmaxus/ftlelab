

def run_experiment(exp_cfg):
    # 1. Data
    X_train, y_train, X_val, y_val, X_test, y_test = make_dataset_and_split(exp_cfg.data)

    # 2. Model
    model = make_model(**exp_cfg.model)

    # 3. Training
    trainer = Trainer(model, TrainConfig(**exp_cfg.train))
    trainer.train(train_loader, val_loader)
    best_model = trainer.model  # or reload best checkpoint

    # 4. Evaluation
    test_acc = evaluate_accuracy(best_model, X_test, y_test)

    # 5. FTLE
    ftle_field = ftle_field_for_model(
        best_model,
        layerspec=exp_cfg.ftle.layerspec,
        time_L=exp_cfg.ftle.time_L,
        grid_bounds=exp_cfg.ftle.bounds,
        nx=exp_cfg.ftle.nx,
        ny=exp_cfg.ftle.ny,
    )

    # 6. Save plots / metrics
    save_results_and_plots(exp_cfg, test_acc, ftle_field)