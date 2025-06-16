from train_model import train_model

if __name__ == "__main__":
    test_accuracy = train_model()
    print(f"Model training completed with test accuracy: {test_accuracy * 100:.2f}%")
