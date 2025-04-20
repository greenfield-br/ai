import numpy as np
import pathlib

def split_mnist():
    # Load the original MNIST dataset
    data_path = f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz"
    with np.load(data_path) as f:
        images, labels = f["x_train"], f["y_train"]
    
    # Ensure we have 60,000 samples
    assert images.shape[0] == 60000 and labels.shape[0] == 60000, "Unexpected MNIST dataset size"
    
    # Shuffle the dataset
    indices = np.random.permutation(images.shape[0])
    images = images[indices]
    labels = labels[indices]
    
    # Split into training (50,000) and evaluation (10,000)
    train_images = images[:50000]
    train_labels = labels[:50000]
    eval_images = images[50000:]
    eval_labels = labels[50000:]
    
    # Save training set
    train_path = f"{pathlib.Path(__file__).parent.absolute()}/data/mnist_train.npz"
    np.savez(train_path, x_train=train_images, y_train=train_labels)
    print(f"Saved training set (50,000 samples) to {train_path}")
    
    # Save evaluation set
    eval_path = f"{pathlib.Path(__file__).parent.absolute()}/data/mnist_eval.npz"
    np.savez(eval_path, x_eval=eval_images, y_eval=eval_labels)
    print(f"Saved evaluation set (10,000 samples) to {eval_path}")

if __name__ == "__main__":
    try:
        split_mnist()
    except FileNotFoundError as e:
        print(f"Error: Could not load mnist.npz: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
