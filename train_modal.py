import modal

app = modal.App("visa-approval-training")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "pandas", "numpy", "scikit-learn", "imblearn", "xgboost", "catboost",
        "dill", "PyYAML", "neuro_mf", "evidently==0.2.8", "from_root",
        "scipy", "matplotlib", "seaborn",
    )
    .add_local_dir("visa_approval_prediction", remote_path="/root/project/visa_approval_prediction", copy=True)
    .add_local_dir("config", remote_path="/root/project/config", copy=True)
    .add_local_dir("notebook", remote_path="/root/project/notebook", copy=True)
    .add_local_file("setup.py", remote_path="/root/project/setup.py", copy=True)
    .run_commands("touch /root/project/.project-root && cd /root/project && pip install -e .")
)


@app.function(image=image, timeout=1800)
def train():
    import os
    os.chdir("/root/project")
    os.makedirs("model_registry", exist_ok=True)

    from visa_approval_prediction.pipline.training_pipeline import TrainPipeline

    pipeline = TrainPipeline()
    pipeline.run_pipeline()

    model_path = os.path.join("model_registry", "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        print(f"Training complete! Model size: {len(model_bytes) / 1024:.1f} KB")
        return model_bytes
    else:
        raise FileNotFoundError("Training finished but model_registry/model.pkl was not created. Model may not have been accepted.")


@app.local_entrypoint()
def main():
    import os

    print("Starting training on Modal...")
    model_bytes = train.remote()

    os.makedirs("model_registry", exist_ok=True)
    model_path = os.path.join("model_registry", "model.pkl")
    with open(model_path, "wb") as f:
        f.write(model_bytes)

    print(f"Model saved to {model_path}")
    print("You can now run: python app.py")
