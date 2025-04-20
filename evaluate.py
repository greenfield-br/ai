import numpy as np
import pathlib
import sys
from scipy.special import softmax
from numpy import array, argmax

class ANN:
    def __init__(self, _in, _out, _KN):
        retCode = 1
        if not isinstance(_in, int) or not isinstance(_out, int) or not isinstance(_KN, list) or not all(isinstance(i, int) for i in _KN):
            retCode = 0
        arrKN = np.array(_KN)
        if arrKN.ndim != 1 or arrKN.shape[0] < 1 or any(_e < 1 for _e in arrKN):
            retCode = 0
        if retCode != 1:
            print("Error: Invalid ANN initialization parameters", file=sys.stderr)
            return
        self.KN = _KN + [_out]
        self.lst = None

    def load_model(self, model_path):
        try:
            self.lst = np.load(model_path, allow_pickle=True).tolist()
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            return False
        return True

    def f(self, _z, _layer_type='hidden'):
        if _layer_type == 'output':
            return softmax(_z)
        return np.maximum(0, _z)

    def Y(self, _IN):
        arrIN = np.array(_IN)
        if arrIN.ndim != 1 or arrIN.shape[0] < 1:
            print("Error: Invalid input shape", file=sys.stderr)
            return 0
        lstY = []
        lstZ = []
        lenKN = len(self.KN)
        Kin = arrIN
        for counter1 in range(lenKN):
            _coefK = self.lst[counter1]
            _z = _coefK[:, :-1] @ Kin + _coefK[:, -1]
            _y = self.f(_z, _layer_type='output' if counter1 == lenKN-1 else 'hidden')
            lstZ.append(_z)
            lstY.append(_y)
            Kin = _y
        return lstZ, lstY

def get_mnist_eval():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist_eval.npz") as f:
        images, labels = f["x_eval"], f["y_eval"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

def evaluate_model(model_path):
    # Initialize ANN
    ann = ANN(784, 10, [256])
    if not ann.load_model(model_path):
        return
    
    # Load evaluation dataset
    try:
        images, labels = get_mnist_eval()
        print(f"Loaded {len(images)} evaluation images and {len(labels)} labels", flush=True)
    except FileNotFoundError as e:
        print(f"Error: Could not load mnist_eval.npz: {e}", file=sys.stderr)
        return
    
    # Evaluate
    nr_correct = 0
    for img, lbl in zip(images, labels):
        _, y = ann.Y(img)
        y_pred = np.array(y[-1])
        nr_correct += int(argmax(y_pred) == argmax(lbl))
    
    accuracy = (nr_correct / len(images)) * 100
    print(f"Evaluation Accuracy: {accuracy:.2f}% ({nr_correct}/{len(images)} correct)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ANN on MNIST evaluation set")
    parser.add_argument("model_path", type=str, help="Path to the trained model (.npy file)")
    args = parser.parse_args()
    
    evaluate_model(args.model_path)
