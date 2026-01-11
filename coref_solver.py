# coref_solver.py
from fastcoref import FCoref
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """
    Aggressively silences stdout and stderr, including C-level (TensorFlow) 
    and tqdm progress bars.
    """
    # Open a null file
    with open(os.devnull, "w") as devnull:
        # Save the actual file descriptors
        old_stdout_fd = sys.stdout.fileno()
        old_stderr_fd = sys.stderr.fileno()
        
        # Save copies of the original file descriptors
        saved_stdout_fd = os.dup(old_stdout_fd)
        saved_stderr_fd = os.dup(old_stderr_fd)
        
        try:
            # Redirect stdout/stderr to devnull at the OS level
            os.dup2(devnull.fileno(), old_stdout_fd)
            os.dup2(devnull.fileno(), old_stderr_fd)
            yield
        finally:
            # Restore the original file descriptors
            os.dup2(saved_stdout_fd, old_stdout_fd)
            os.dup2(saved_stderr_fd, old_stderr_fd)
            
            # Close the saved copies
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

class CorefResolver:
    def __init__(self, device='cpu'):
        # We silence the initialization too to hide TensorFlow warnings
        with suppress_output():
            self.model = FCoref(device=device)

    def resolve(self, text: str):
        # We silence the prediction to hide "Map/Inference" bars
        with suppress_output():
            preds = self.model.predict(
                texts=[text],
                is_split_into_words=False
            )
        return preds[0].get_clusters(as_strings=False)