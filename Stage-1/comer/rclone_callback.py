from pytorch_lightning.callbacks import Callback
import subprocess

class RcloneUploadCallback(Callback):
        def __init__(self, local_dir, remote_dir):
            super().__init__()
            self.local_dir = local_dir  # Directory to save local checkpoints
            self.remote_dir = remote_dir  # OneDrive remote directory

        def on_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch %5 ==0:
                print(f"Epoch {trainer.current_epoch} finished. Uploading checkpoints to OneDrive...")
                # Upload all files from the local directory to OneDrive
                command = f"rclone copy --verbose {self.local_dir} {self.remote_dir}"
                subprocess.run(command, shell=True, check=True)
                print(f"Upload completed for epoch {trainer.current_epoch}.")

        def on_train_end(self, trainer, pl_module):
            if trainer.current_epoch %5 ==0:
                print("Training complete. Final upload to OneDrive...")
                # Upload one last time when training finishes
                command = f"rclone copy --verbose {self.local_dir} {self.remote_dir}"
                subprocess.run(command, shell=True, check=True)
                print("Final upload completed.")