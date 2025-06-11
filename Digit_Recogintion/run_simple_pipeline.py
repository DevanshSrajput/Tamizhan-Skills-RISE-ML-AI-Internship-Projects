import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are available"""
    required_packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'cv2': 'OpenCV'
    }
    
    missing_packages = []
    available_packages = []
    
    print("🔍 Checking dependencies...")
    for package, name in required_packages.items():
        try:
            if package == 'PIL':
                import PIL
                available_packages.append(f"✅ {name}")
            elif package == 'cv2':
                import cv2
                available_packages.append(f"✅ {name}")
            elif package == 'sklearn':
                import sklearn
                available_packages.append(f"✅ {name}")
            else:
                __import__(package)
                available_packages.append(f"✅ {name}")
        except ImportError:
            missing_packages.append(package)
    
    # Print results
    for pkg in available_packages:
        print(pkg)
    
    if missing_packages:
        print("\n❌ Missing packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n✅ All required dependencies are available!")
    return True

def install_requirements():
    """Install requirements from requirements.txt"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        print("Creating requirements.txt...")
        
        requirements = """tensorflow>=2.12.0,<2.16.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.0.0
opencv-python>=4.5.0"""
        
        with open("requirements.txt", "w") as f:
            f.write(requirements)
        print("✅ requirements.txt created!")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_model_exists():
    """Check if trained model exists"""
    model_files = ['mnist_digit_model.h5', 'best_mnist_model.h5']
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ Found existing model: {model_file}")
            return True
    
    print("❌ No trained model found")
    return False

def train_model():
    """Train the MNIST model by running the main script"""
    print("\n" + "="*60)
    print("🤖 TRAINING THE MNIST MODEL")
    print("="*60)
    print("This will take 5-10 minutes depending on your hardware...")
    print("The script will show training progress and save the model.")
    
    try:
        # Import and run the main training script
        from mnist_digit_recognition import main as train_main
        train_main()
        print("✅ Model training completed successfully!")
        return True
    except ImportError:
        print("❌ Could not import mnist_digit_recognition.py")
        print("Please ensure the file exists in the current directory.")
        return False
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return False

def launch_tkinter_app():
    """Launch the Tkinter drawing application"""
    print("\n" + "="*60)
    print("🎨 LAUNCHING TKINTER DRAWING APPLICATION")
    print("="*60)
    print("Instructions:")
    print("- Draw a digit (0-9) in the white canvas")
    print("- Click 'Predict Digit' to see the prediction")
    print("- Click 'Clear Canvas' to start over")
    print("- Close the window when done")
    
    try:
        from digit_drawing_ui import main as ui_main
        ui_main()
        print("✅ Tkinter application closed successfully")
        return True
    except ImportError:
        print("❌ Could not import digit_drawing_ui.py")
        print("Please ensure the file exists in the current directory.")
        return False
    except Exception as e:
        print(f"❌ Failed to launch Tkinter app: {e}")
        return False

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\n" + "="*60)
    print("🧪 RUNNING QUICK TEST")
    print("="*60)
    
    try:
        # Test imports
        from mnist_digit_recognition import MNISTDigitRecognizer
        
        # Create recognizer instance
        recognizer = MNISTDigitRecognizer()
        
        # Test data loading
        print("Testing data loading...")
        x_train, y_train, x_test, y_test = recognizer.load_and_preprocess_data()
        print(f"✅ Data loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples")
        
        # Test model building
        print("Testing model building...")
        model = recognizer.build_cnn_model()
        print("✅ Model built successfully")
        
        print("\n🎉 All tests passed! The system is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("🔢 MNIST DIGIT RECOGNITION - MAIN MENU")
    print("="*60)
    print("Choose an option:")
    print("1. 🧪 Run Quick Test (verify setup)")
    print("2. 🤖 Train Model (if not already trained)")
    print("3. 🎨 Launch Tkinter Drawing App")
    print("4. 🔄 Train New Model (overwrite existing)")
    print("5. ❌ Exit")
    print("="*60)

def main():
    """Main pipeline function"""
    print("🚀 MNIST DIGIT RECOGNITION - SIMPLIFIED PIPELINE")
    print("Current User:", "DevanshSrajput")
    print("Current Time:", "2025-06-11 13:02:57 UTC")
    print("="*60)
    
    # Step 1: Check and install dependencies
    print("📋 Step 1: Checking dependencies...")
    if not check_dependencies():
        print("Installing missing dependencies...")
        if not install_requirements():
            print("❌ Failed to install dependencies. Please install manually.")
            return
    
    # Step 2: Check if model exists
    print("\n🤖 Step 2: Checking for trained model...")
    model_files = ['mnist_digit_model.h5', 'best_mnist_model.h5']
    model_exists = any(Path(f).exists() for f in model_files)
    
    if not model_exists:
        print("No trained model found. Training new model...")
        if not train_model():
            print("❌ Model training failed!")
            return
    else:
        print("✅ Trained model found!")
    
    # Step 3: Launch application menu
    while True:
        show_menu()
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            run_quick_test()
            
        elif choice == "2":
            if model_exists:
                retrain = input("Model already exists. Retrain anyway? (y/n): ").lower()
                if retrain == 'y':
                    train_model()
            else:
                train_model()
                
        elif choice == "3":
            launch_tkinter_app()
            
        elif choice == "4":
            train_model()
            
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()