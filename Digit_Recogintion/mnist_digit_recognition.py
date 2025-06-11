import warnings
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# TensorFlow import with error handling
def check_and_install_tensorflow():
    """Check TensorFlow installation and provide solutions"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} is available!")
        return True
    except ImportError as e:
        print("❌ TensorFlow import failed!")
        print(f"Error: {e}")
        print("\n🔧 Possible solutions:")
        print("1. Install TensorFlow CPU version:")
        print("   pip install tensorflow-cpu==2.15.0")
        print("\n2. Or install regular TensorFlow:")
        print("   pip install tensorflow==2.15.0")
        print("\n3. Update pip and try again:")
        print("   python -m pip install --upgrade pip")
        print("   pip install tensorflow")
        
        # Ask user if they want to try installing now
        choice = input("\nWould you like to try installing TensorFlow now? (y/n): ").lower()
        if choice == 'y':
            return install_tensorflow()
        return False

def install_tensorflow():
    """Try to install TensorFlow automatically"""
    print("🔄 Attempting to install TensorFlow...")
    
    # Try different installation methods
    install_commands = [
        [sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.15.0"],
        [sys.executable, "-m", "pip", "install", "tensorflow==2.15.0"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
    ]
    
    for i, cmd in enumerate(install_commands):
        try:
            print(f"Trying installation method {i+1}...")
            subprocess.check_call(cmd)
            
            # Test if tensorflow works now
            try:
                import tensorflow as tf
                print(f"✅ TensorFlow {tf.__version__} installed successfully!")
                return True
            except ImportError:
                continue
                
        except subprocess.CalledProcessError:
            print(f"❌ Installation method {i+1} failed")
            continue
    
    print("❌ All installation attempts failed.")
    print("Please install TensorFlow manually and try again.")
    return False

# Check TensorFlow before importing other modules
print("🔍 Checking TensorFlow installation...")
if not check_and_install_tensorflow():
    print("❌ Cannot proceed without TensorFlow. Exiting...")
    sys.exit(1)

# Now import TensorFlow and other required modules
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("✅ All TensorFlow modules imported successfully!")
except ImportError as e:
    print(f"❌ TensorFlow module import failed: {e}")
    sys.exit(1)

# Import other required packages
try:
    from sklearn.metrics import classification_report, confusion_matrix
    print("✅ All other packages imported successfully!")
except ImportError as e:
    print(f"❌ Package import failed: {e}")
    print("Please install missing packages:")
    print("pip install numpy matplotlib seaborn scikit-learn")
    sys.exit(1)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MNISTDigitRecognizer:
    def __init__(self):
        """Initialize the MNIST Digit Recognizer"""
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        print("🤖 MNISTDigitRecognizer initialized successfully!")
        
    def load_and_preprocess_data(self):
        """Load and preprocess the MNIST dataset"""
        print("📥 Loading MNIST dataset...")
        
        try:
            # Load the data
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            
            print(f"✅ Data loaded successfully!")
            print(f"Training data shape: {x_train.shape}")
            print(f"Training labels shape: {y_train.shape}")
            print(f"Test data shape: {x_test.shape}")
            print(f"Test labels shape: {y_test.shape}")
            
            # Normalize pixel values to [0, 1]
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            
            # Reshape data to add channel dimension (for CNN)
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
            
            # Convert labels to categorical (one-hot encoding)
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)
            
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            
            print("✅ Data preprocessing completed!")
            return x_train, y_train, x_test, y_test
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None, None, None, None
    
    def visualize_data(self):
        """Visualize sample images from the dataset"""
        if self.x_train is None:
            print("❌ No data to visualize. Load data first!")
            return
        
        try:
            # Convert back to original labels for visualization
            original_labels = np.argmax(self.y_train, axis=1)
            
            plt.figure(figsize=(12, 8))
            for i in range(25):
                plt.subplot(5, 5, i+1)
                plt.imshow(self.x_train[i].reshape(28, 28), cmap='gray')
                plt.title(f'Label: {original_labels[i]}')
                plt.axis('off')
            plt.suptitle('Sample MNIST Digits', fontsize=16)
            plt.tight_layout()
            plt.show()
            
            # Display class distribution
            unique, counts = np.unique(original_labels, return_counts=True)
            plt.figure(figsize=(10, 6))
            plt.bar(unique, counts)
            plt.title('Distribution of Digits in Training Set')
            plt.xlabel('Digit')
            plt.ylabel('Count')
            plt.show()
            
            print("✅ Data visualization completed!")
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
    
    def build_cnn_model(self):
        """Build the CNN model architecture"""
        print("🏗️ Building CNN model...")
        
        try:
            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            print("✅ CNN model built and compiled successfully!")
            return model
            
        except Exception as e:
            print(f"❌ Model building failed: {e}")
            return None
    
    def display_model_summary(self):
        """Display model architecture summary"""
        if self.model:
            print("\n📋 Model Architecture Summary:")
            self.model.summary()
        else:
            print("❌ No model to display. Build model first!")
    
    def train_model(self, epochs=20, batch_size=128, validation_split=0.1):
        """Train the CNN model with proper data handling"""
        if self.model is None:
            print("❌ No model to train. Build model first!")
            return None
        
        if self.x_train is None:
            print("❌ No training data. Load data first!")
            return None
        
        print(f"🚀 Training model for {epochs} epochs...")
        print("This may take 5-10 minutes depending on your hardware...")
        
        try:
            # Split data manually for validation
            from sklearn.model_selection import train_test_split
            
            x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
                self.x_train, self.y_train, 
                test_size=validation_split, 
                random_state=42, 
                stratify=np.argmax(self.y_train, axis=1)
            )
            
            # Create data augmentation generator for training data only
            datagen = ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1
            )
            
            # Fit the generator on training data
            datagen.fit(x_train_split)
            
            # Create training generator
            train_generator = datagen.flow(
                x_train_split, y_train_split,
                batch_size=batch_size
            )
            
            # Calculate steps per epoch
            steps_per_epoch = len(x_train_split) // batch_size
            
            # Add callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=0.0001
                )
            ]
            
            # Train the model
            history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=(x_val_split, y_val_split),
                callbacks=callbacks,
                verbose=1
            )
            
            self.history = history
            print("✅ Model training completed successfully!")
            return history
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        if self.history is None:
            print("❌ No training history to plot. Train model first!")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot accuracy
            ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)
            
            # Plot loss
            ax2.plot(self.history.history['loss'], label='Training Loss')
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Training history plotted successfully!")
            
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        if self.model is None:
            print("❌ No model to evaluate. Train model first!")
            return None, None
        
        if self.x_test is None:
            print("❌ No test data. Load data first!")
            return None, None
        
        print("📊 Evaluating model on test data...")
        
        try:
            # Evaluate on test data
            test_loss, test_accuracy = self.model.evaluate(
                self.x_test, self.y_test, verbose=0
            )
            
            print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"✅ Test Loss: {test_loss:.4f}")
            
            # Generate predictions for detailed analysis
            predictions = self.model.predict(self.x_test, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(self.y_test, axis=1)
            
            # Classification report
            print("\n📋 Classification Report:")
            print(classification_report(true_classes, predicted_classes))
            
            # Confusion matrix
            cm = confusion_matrix(true_classes, predicted_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
            
            return test_accuracy, test_loss
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None, None
    
    def save_model(self, filepath='mnist_digit_model.h5'):
        """Save the trained model"""
        if self.model:
            try:
                self.model.save(filepath)
                print(f"✅ Model saved successfully as {filepath}")
            except Exception as e:
                print(f"❌ Failed to save model: {e}")
        else:
            print("❌ No model to save. Train model first!")

def main():
    """Main function to run the complete pipeline"""
    print("=" * 80)
    print("🔢 HANDWRITTEN DIGIT RECOGNITION - MNIST DATASET")
    print(f"👤 User: DevanshSrajput")
    print(f"📅 Date: 2025-06-11 13:06:16 UTC")
    print("=" * 80)
    
    try:
        # Initialize the recognizer
        recognizer = MNISTDigitRecognizer()
        
        # Step 1: Load and preprocess data
        print("\n📥 STEP 1: Loading and preprocessing data...")
        x_train, y_train, x_test, y_test = recognizer.load_and_preprocess_data()
        
        if x_train is None:
            print("❌ Failed to load data. Exiting...")
            return
        
        # Step 2: Visualize data
        print("\n👀 STEP 2: Visualizing dataset...")
        recognizer.visualize_data()
        
        # Step 3: Build model
        print("\n🏗️ STEP 3: Building CNN model...")
        model = recognizer.build_cnn_model()
        
        if model is None:
            print("❌ Failed to build model. Exiting...")
            return
        
        recognizer.display_model_summary()
        
        # Step 4: Train model
        print("\n🚀 STEP 4: Training model...")
        history = recognizer.train_model(epochs=20, batch_size=128)
        
        if history is None:
            print("❌ Training failed. Exiting...")
            return
        
        # Step 5: Plot training history
        print("\n📊 STEP 5: Plotting training history...")
        recognizer.plot_training_history()
        
        # Step 6: Evaluate model
        print("\n📈 STEP 6: Evaluating model...")
        test_accuracy, test_loss = recognizer.evaluate_model()
        
        if test_accuracy is None:
            print("❌ Evaluation failed. Exiting...")
            return
        
        # Step 7: Save model
        print("\n💾 STEP 7: Saving model...")
        recognizer.save_model()
        
        # Final results
        print("\n" + "=" * 80)
        print("🎉 FINAL RESULTS")
        print("=" * 80)
        print(f"✅ Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        if test_accuracy > 0.98:
            print("🏆 Excellent! Your model achieved >98% accuracy!")
        else:
            print("👍 Good performance! Consider training for more epochs.")
        
        print("\n🎯 Next steps:")
        print("1. Run 'python digit_drawing_ui.py' for desktop app")
        print("2. Use the saved model for your own applications")
        
    except Exception as e:
        print(f"❌ Unexpected error in main pipeline: {e}")
        print("Please check your Python environment and try again.")

if __name__ == "__main__":
    main()