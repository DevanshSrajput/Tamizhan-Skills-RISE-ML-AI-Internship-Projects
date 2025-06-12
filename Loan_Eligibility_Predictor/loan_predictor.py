import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import warnings
import joblib
warnings.filterwarnings('ignore')

class LoanEligibilityPredictor:
    """
    A comprehensive machine learning model for predicting loan eligibility
    using Logistic Regression and Random Forest algorithms.
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.logistic_model = LogisticRegression(random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.best_model = None
        self.best_model_name = None  # Added to track which model is best
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic loan application data for training and testing.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        np.random.seed(42)
        
        # Generate synthetic data
        data = {
            'Age': np.random.randint(18, 70, n_samples),
            'Income': np.random.normal(50000, 25000, n_samples),
            'Credit_Score': np.random.randint(300, 850, n_samples),
            'Employment_Years': np.random.randint(0, 40, n_samples),
            'Loan_Amount': np.random.normal(200000, 100000, n_samples),
            'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'Employment_Status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
            'Property_Area': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], n_samples)
        }
        
        # Ensure positive values
        data['Income'] = np.abs(data['Income'])
        data['Loan_Amount'] = np.abs(data['Loan_Amount'])
        
        df = pd.DataFrame(data)
        
        # Create loan approval logic based on realistic criteria
        def determine_loan_approval(row):
            score = 0
            
            # Age factor (25-55 is ideal)
            if 25 <= row['Age'] <= 55:
                score += 20
            elif row['Age'] < 25 or row['Age'] > 55:
                score += 10
                
            # Income factor
            if row['Income'] > 60000:
                score += 25
            elif row['Income'] > 40000:
                score += 15
            else:
                score += 5
                
            # Credit score factor
            if row['Credit_Score'] >= 750:
                score += 30
            elif row['Credit_Score'] >= 650:
                score += 20
            elif row['Credit_Score'] >= 550:
                score += 10
            else:
                score += 0
                
            # Employment years
            if row['Employment_Years'] >= 5:
                score += 15
            elif row['Employment_Years'] >= 2:
                score += 10
            else:
                score += 5
                
            # Loan amount to income ratio
            if row['Loan_Amount'] / row['Income'] <= 3:
                score += 10
            elif row['Loan_Amount'] / row['Income'] <= 5:
                score += 5
            else:
                score -= 5
                
            # Add some randomness
            score += np.random.randint(-10, 11)
            
            return 1 if score >= 70 else 0
        
        df['Loan_Approved'] = df.apply(determine_loan_approval, axis=1)
        
        print(f"Generated {n_samples} synthetic loan applications")
        print(f"Approval rate: {df['Loan_Approved'].mean():.2%}")
        
        return df
    
    def load_and_explore_data(self, data=None):
        """
        Load and perform exploratory data analysis on the dataset.
        
        Args:
            data (pd.DataFrame): Dataset to analyze. If None, generates synthetic data.
        """
        if data is None:
            self.data = self.generate_synthetic_data()
        else:
            self.data = data.copy()
            
        print("Dataset Shape:", self.data.shape)
        print("\nDataset Info:")
        print(self.data.info())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nTarget Variable Distribution:")
        print(self.data['Loan_Approved'].value_counts())
        
        # Create visualizations
        try:
            self.create_exploratory_plots()
        except Exception as e:
            print(f"Warning: Could not create plots - {e}")
        
    def create_exploratory_plots(self):
        """Create exploratory data analysis plots."""
        plt.figure(figsize=(15, 12))
        
        # Distribution of target variable
        plt.subplot(2, 3, 1)
        self.data['Loan_Approved'].value_counts().plot(kind='bar')
        plt.title('Loan Approval Distribution')
        plt.xlabel('Loan Approved (0: No, 1: Yes)')
        plt.ylabel('Count')
        
        # Age distribution by loan approval
        plt.subplot(2, 3, 2)
        sns.boxplot(data=self.data, x='Loan_Approved', y='Age')
        plt.title('Age Distribution by Loan Approval')
        
        # Income distribution by loan approval
        plt.subplot(2, 3, 3)
        sns.boxplot(data=self.data, x='Loan_Approved', y='Income')
        plt.title('Income Distribution by Loan Approval')
        
        # Credit score distribution by loan approval
        plt.subplot(2, 3, 4)
        sns.boxplot(data=self.data, x='Loan_Approved', y='Credit_Score')
        plt.title('Credit Score Distribution by Loan Approval')
        
        # Education level distribution
        plt.subplot(2, 3, 5)
        education_approval = pd.crosstab(self.data['Education'], self.data['Loan_Approved'])
        education_approval.plot(kind='bar', stacked=True)
        plt.title('Education Level vs Loan Approval')
        plt.xticks(rotation=45)
        
        # Employment status distribution
        plt.subplot(2, 3, 6)
        employment_approval = pd.crosstab(self.data['Employment_Status'], self.data['Loan_Approved'])
        employment_approval.plot(kind='bar', stacked=True)
        plt.title('Employment Status vs Loan Approval')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('loan_eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """
        Preprocess the data by handling missing values, encoding categorical variables,
        and scaling numerical features.
        """
        # Handle missing values (fill with mode for categorical, median for numerical)
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            else:
                self.data[column].fillna(self.data[column].median(), inplace=True)
        
        # Separate features and target
        X = self.data.drop('Loan_Approved', axis=1)
        y = self.data['Loan_Approved']
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            self.label_encoders[column] = le
            
        # Store feature names in consistent order
        self.feature_names = X.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preprocessing completed!")
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
    def train_models(self):
        """Train both Logistic Regression and Random Forest models."""
        print("Training models...")
        
        # Train Logistic Regression
        self.logistic_model.fit(self.X_train_scaled, self.y_train)
        
        # Train Random Forest
        self.rf_model.fit(self.X_train, self.y_train)
        
        print("Models trained successfully!")
        
    def evaluate_models(self):
        """
        Evaluate both models using various metrics and determine the best model.
        
        Returns:
            dict: Dictionary containing evaluation results for both models
        """
        results = {}
        
        # Logistic Regression predictions
        lr_train_pred = self.logistic_model.predict(self.X_train_scaled)
        lr_test_pred = self.logistic_model.predict(self.X_test_scaled)
        lr_test_proba = self.logistic_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Random Forest predictions
        rf_train_pred = self.rf_model.predict(self.X_train)
        rf_test_pred = self.rf_model.predict(self.X_test)
        rf_test_proba = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics for Logistic Regression
        lr_train_acc = accuracy_score(self.y_train, lr_train_pred)
        lr_test_acc = accuracy_score(self.y_test, lr_test_pred)
        lr_cv_scores = cross_val_score(self.logistic_model, self.X_train_scaled, self.y_train, cv=5)
        
        # Calculate metrics for Random Forest
        rf_train_acc = accuracy_score(self.y_train, rf_train_pred)
        rf_test_acc = accuracy_score(self.y_test, rf_test_pred)
        rf_cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5)
        
        # Store results
        results['Logistic Regression'] = {
            'train_accuracy': lr_train_acc,
            'test_accuracy': lr_test_acc,
            'cv_mean': lr_cv_scores.mean(),
            'cv_std': lr_cv_scores.std(),
            'predictions': lr_test_pred,
            'probabilities': lr_test_proba
        }
        
        results['Random Forest'] = {
            'train_accuracy': rf_train_acc,
            'test_accuracy': rf_test_acc,
            'cv_mean': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std(),
            'predictions': rf_test_pred,
            'probabilities': rf_test_proba
        }
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
        
        # Determine best model
        if results['Random Forest']['test_accuracy'] > results['Logistic Regression']['test_accuracy']:
            self.best_model = self.rf_model
            self.best_model_name = 'Random Forest'
            best_predictions = rf_test_pred
            best_probabilities = rf_test_proba
        else:
            self.best_model = self.logistic_model
            self.best_model_name = 'Logistic Regression'  # Fixed typo
            best_predictions = lr_test_pred
            best_probabilities = lr_test_proba
            
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Test Accuracy: {max(results['Random Forest']['test_accuracy'], results['Logistic Regression']['test_accuracy']):.4f}")
        
        # Create evaluation plots
        try:
            self.create_evaluation_plots(results, best_predictions, best_probabilities)
        except Exception as e:
            print(f"Warning: Could not create evaluation plots - {e}")
        
        return results
    
    def create_evaluation_plots(self, results, best_predictions, best_probabilities):
        """Create evaluation plots including confusion matrices and ROC curves."""
        plt.figure(figsize=(15, 10))
        
        # Confusion matrices
        plt.subplot(2, 3, 1)
        cm_lr = confusion_matrix(self.y_test, results['Logistic Regression']['predictions'])
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
        plt.title('Logistic Regression\nConfusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.subplot(2, 3, 2)
        cm_rf = confusion_matrix(self.y_test, results['Random Forest']['predictions'])
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
        plt.title('Random Forest\nConfusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # ROC Curves
        plt.subplot(2, 3, 3)
        
        # Logistic Regression ROC
        fpr_lr, tpr_lr, _ = roc_curve(self.y_test, results['Logistic Regression']['probabilities'])
        auc_lr = auc(fpr_lr, tpr_lr)
        plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
        
        # Random Forest ROC
        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, results['Random Forest']['probabilities'])
        auc_rf = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        # Model comparison
        plt.subplot(2, 3, 4)
        models = ['Logistic Regression', 'Random Forest']
        test_accuracies = [results['Logistic Regression']['test_accuracy'], 
                          results['Random Forest']['test_accuracy']]
        
        bars = plt.bar(models, test_accuracies, color=['blue', 'green'], alpha=0.7)
        plt.title('Model Comparison - Test Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, test_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Feature importance (Random Forest)
        plt.subplot(2, 3, 5)
        feature_importance = self.rf_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
        
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
        plt.title('Top 10 Feature Importance\n(Random Forest)')
        plt.xlabel('Importance')
        
        # Cross-validation scores comparison
        plt.subplot(2, 3, 6)
        cv_means = [results['Logistic Regression']['cv_mean'], results['Random Forest']['cv_mean']]
        cv_stds = [results['Logistic Regression']['cv_std'], results['Random Forest']['cv_std']]
        
        plt.bar(models, cv_means, yerr=cv_stds, capsize=5, color=['blue', 'green'], alpha=0.7)
        plt.title('Cross-Validation Scores')
        plt.ylabel('CV Score')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict_single_application(self, application_data):
        """
        Predict loan eligibility for a single application.
        
        Args:
            application_data (dict): Dictionary containing application details
            
        Returns:
            tuple: (prediction, probability)
        """
        # Convert to DataFrame with consistent column order
        df = pd.DataFrame([application_data])
        
        # Reorder columns to match training data
        df = df.reindex(columns=self.feature_names)
        
        # Encode categorical variables
        for column in df.columns:
            if column in self.label_encoders:
                if application_data[column] in self.label_encoders[column].classes_:
                    df[column] = self.label_encoders[column].transform([application_data[column]])[0]
                else:
                    # Handle unseen category by using the most frequent class
                    df[column] = 0
        
        # Make prediction based on best model type
        if self.best_model_name == 'Logistic Regression':
            # Scale the data for logistic regression
            df_scaled = self.scaler.transform(df)
            prediction = self.best_model.predict(df_scaled)[0]
            probability = self.best_model.predict_proba(df_scaled)[0][1]
        else:
            # Random Forest doesn't need scaling
            prediction = self.best_model.predict(df)[0]
            probability = self.best_model.predict_proba(df)[0][1]
            
        return prediction, probability
    
    def save_model(self, filename='loan_predictor_model.pkl'):
        """Save the trained model and preprocessors."""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
        
    def load_model(self, filename='loan_predictor_model.pkl'):
        """Load a pre-trained model and preprocessors."""
        model_data = joblib.load(filename)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data.get('best_model_name', 'Unknown')
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filename}")

def main():
    """
    Main function to run the complete loan eligibility prediction pipeline.
    """
    print("="*60)
    print("LOAN ELIGIBILITY PREDICTOR")
    print("="*60)
    
    # Initialize the predictor
    predictor = LoanEligibilityPredictor()
    
    # Load and explore data
    print("\n1. Loading and exploring data...")
    predictor.load_and_explore_data()
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    predictor.preprocess_data()
    
    # Train models
    print("\n3. Training models...")
    predictor.train_models()
    
    # Evaluate models
    print("\n4. Evaluating models...")
    results = predictor.evaluate_models()
    
    # Save the model
    print("\n5. Saving model...")
    predictor.save_model()
    
    # Example prediction
    print("\n6. Example prediction...")
    sample_application = {
        'Age': 35,
        'Income': 65000,
        'Credit_Score': 720,
        'Employment_Years': 8,
        'Loan_Amount': 250000,
        'Education': 'Bachelor',
        'Marital_Status': 'Married',
        'Employment_Status': 'Employed',
        'Property_Area': 'Urban'
    }
    
    prediction, probability = predictor.predict_single_application(sample_application)
    print(f"Sample Application: {sample_application}")
    print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
    print(f"Confidence: {probability:.2%}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()