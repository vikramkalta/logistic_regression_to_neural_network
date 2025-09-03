"""
Comprehensive Logistic Regression Tutorial with Iris Dataset
===========================================================

This tutorial demonstrates:
1. Dataset exploration and visualization
2. Why logistic regression is appropriate for this problem
3. Step-by-step model training and evaluation
4. Visualization of results and decision boundaries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IrisLogisticRegressionTutorial:
    """
    A comprehensive tutorial class for logistic regression using the Iris dataset.
    """
    
    def __init__(self):
        """Initialize the tutorial by loading the Iris dataset."""
        print("🌸 Loading Iris Dataset...")
        print("=" * 50)
        
        # Load the famous Iris dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names
        
        # Create a DataFrame for easier manipulation
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['species'] = pd.Categorical.from_codes(self.y, self.target_names)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Features: {list(self.feature_names)}")
        print(f"Target classes: {list(self.target_names)}")
        
    def explore_dataset(self):
        """Comprehensive dataset exploration."""
        print("\n🔍 Dataset Exploration")
        print("=" * 50)
        
        # Basic information
        print("📊 Basic Dataset Information:")
        print(f"Number of samples: {len(self.df)}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of classes: {len(self.target_names)}")
        
        # Display first few rows
        print("\n📋 First 5 rows:")
        print(self.df.head())
        
        # Statistical summary
        print("\n📈 Statistical Summary:")
        print(self.df.describe())
        
        # Check for missing values
        print("\n❓ Missing values:")
        missing_values = self.df.isnull().sum()
        print(missing_values)
        
        # Class distribution
        print("\n🏷️ Class Distribution:")
        class_counts = self.df['species'].value_counts()
        print(class_counts)
        
        return class_counts
    
    def visualize_dataset(self):
        """Create comprehensive visualizations of the dataset."""
        print("\n📊 Creating Dataset Visualizations...")
        print("=" * 50)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Class distribution
        plt.subplot(3, 3, 1)
        class_counts = self.df['species'].value_counts()
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        
        # 2. Feature distributions
        for i, feature in enumerate(self.feature_names):
            plt.subplot(3, 3, i + 2)
            for species in self.target_names:
                data = self.df[self.df['species'] == species][feature]
                plt.hist(data, alpha=0.7, label=species, bins=15)
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {feature}')
            plt.legend()
        
        # 3. Correlation heatmap
        plt.subplot(3, 3, 6)
        correlation_matrix = self.df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        # 4. Pairplot (separate figure for clarity)
        plt.tight_layout()
        plt.show()
        
        # Create pairplot
        print("Creating pairplot...")
        plt.figure(figsize=(12, 10))
        sns.pairplot(self.df, hue='species', diag_kind='hist')
        plt.suptitle('Pairplot of Iris Features by Species', y=1.02, fontsize=16)
        plt.show()
        
        # Box plots for each feature
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_names):
            sns.boxplot(data=self.df, x='species', y=feature, ax=axes[i])
            axes[i].set_title(f'Box Plot: {feature}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def explain_logistic_regression_choice(self):
        """Explain why logistic regression is appropriate for this problem."""
        print("\n🤔 Why Logistic Regression for Iris Classification?")
        print("=" * 60)
        
        reasons = [
            "1. 🎯 CLASSIFICATION PROBLEM: We're predicting discrete categories (species), not continuous values",
            "2. 📊 MULTIPLE CLASSES: Logistic regression can handle multi-class problems (3 species)",
            "3. 📏 LINEAR SEPARABILITY: Iris features show good linear separability between classes",
            "4. 🔢 NUMERICAL FEATURES: All features are continuous numerical values (perfect for logistic regression)",
            "5. 📈 PROBABILISTIC OUTPUT: We get probability estimates for each class prediction",
            "6. 🚀 INTERPRETABILITY: Coefficients tell us feature importance and direction of influence",
            "7. ⚡ EFFICIENCY: Fast training and prediction, good for small to medium datasets",
            "8. 🎲 NO ASSUMPTIONS: Doesn't assume normal distribution of features (unlike some other methods)"
        ]
        
        for reason in reasons:
            print(reason)
        
        print("\n💡 Key Insight:")
        print("Logistic regression uses the sigmoid function to map any real number to a probability")
        print("between 0 and 1, making it perfect for classification tasks!")
        
        # Show sigmoid function
        x = np.linspace(-10, 10, 100)
        sigmoid = 1 / (1 + np.exp(-x))
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, sigmoid, 'b-', linewidth=3, label='Sigmoid Function')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
        plt.xlabel('Input (z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ)')
        plt.ylabel('Probability P(y=1)')
        plt.title('Sigmoid Function: The Heart of Logistic Regression')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    def prepare_data(self):
        """Prepare data for modeling."""
        print("\n🔧 Preparing Data for Modeling...")
        print("=" * 50)
        
        # For binary classification demonstration, let's first do setosa vs others
        print("Step 1: Binary Classification (Setosa vs Others)")
        y_binary = (self.y == 0).astype(int)  # 1 for setosa, 0 for others
        
        # Split the data
        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
            self.X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )
        
        # Scale the features
        scaler_bin = StandardScaler()
        X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
        X_test_bin_scaled = scaler_bin.transform(X_test_bin)
        
        print(f"Binary - Training set: {X_train_bin.shape}, Test set: {X_test_bin.shape}")
        
        # For multi-class classification
        print("\nStep 2: Multi-class Classification (All 3 species)")
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Scale the features
        scaler_multi = StandardScaler()
        X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
        X_test_multi_scaled = scaler_multi.transform(X_test_multi)
        
        print(f"Multi-class - Training set: {X_train_multi.shape}, Test set: {X_test_multi.shape}")
        
        # Store for later use
        self.binary_data = {
            'X_train': X_train_bin_scaled, 'X_test': X_test_bin_scaled,
            'y_train': y_train_bin, 'y_test': y_test_bin, 'scaler': scaler_bin
        }
        
        self.multi_data = {
            'X_train': X_train_multi_scaled, 'X_test': X_test_multi_scaled,
            'y_train': y_train_multi, 'y_test': y_test_multi, 'scaler': scaler_multi
        }
        
        print("✅ Data preparation complete!")
    
    def train_binary_model(self):
        """Train and evaluate binary logistic regression model."""
        print("\n🎯 Training Binary Logistic Regression (Setosa vs Others)")
        print("=" * 60)
        
        # Create and train the model
        model_binary = LogisticRegression(random_state=42)
        model_binary.fit(self.binary_data['X_train'], self.binary_data['y_train'])
        
        # Make predictions
        y_pred_binary = model_binary.predict(self.binary_data['X_test'])
        y_pred_proba_binary = model_binary.predict_proba(self.binary_data['X_test'])[:, 1]
        
        # Evaluate the model
        accuracy = accuracy_score(self.binary_data['y_test'], y_pred_binary)
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(self.binary_data['y_test'], y_pred_binary)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        # Classification Report
        print(f"\n📋 Classification Report:")
        print(classification_report(self.binary_data['y_test'], y_pred_binary, 
                                  target_names=['Not Setosa', 'Setosa']))
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.binary_data['y_test'], y_pred_proba_binary)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        
        # Feature Importance
        feature_importance = abs(model_binary.coef_[0])
        axes[2].barh(self.feature_names, feature_importance)
        axes[2].set_title('Feature Importance (|Coefficients|)')
        axes[2].set_xlabel('Absolute Coefficient Value')
        
        plt.tight_layout()
        plt.show()
        
        self.model_binary = model_binary
        return model_binary
    
    def train_multiclass_model(self):
        """Train and evaluate multi-class logistic regression model."""
        print("\n🎯 Training Multi-class Logistic Regression (All 3 Species)")
        print("=" * 60)
        
        # Create and train the model
        model_multi = LogisticRegression(random_state=42, max_iter=200)
        model_multi.fit(self.multi_data['X_train'], self.multi_data['y_train'])
        
        # Make predictions
        y_pred_multi = model_multi.predict(self.multi_data['X_test'])
        y_pred_proba_multi = model_multi.predict_proba(self.multi_data['X_test'])
        
        # Evaluate the model
        accuracy = accuracy_score(self.multi_data['y_test'], y_pred_multi)
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(self.multi_data['y_test'], y_pred_multi)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        # Classification Report
        print(f"\n📋 Classification Report:")
        print(classification_report(self.multi_data['y_test'], y_pred_multi, 
                                  target_names=self.target_names))
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        axes[0,0].set_xticklabels(self.target_names, rotation=45)
        axes[0,0].set_yticklabels(self.target_names, rotation=0)
        
        # Feature Importance for each class
        feature_importance = abs(model_multi.coef_)
        im = axes[0,1].imshow(feature_importance, cmap='viridis', aspect='auto')
        axes[0,1].set_title('Feature Importance by Class')
        axes[0,1].set_xlabel('Features')
        axes[0,1].set_ylabel('Classes')
        axes[0,1].set_xticks(range(len(self.feature_names)))
        axes[0,1].set_xticklabels(self.feature_names, rotation=45)
        axes[0,1].set_yticks(range(len(self.target_names)))
        axes[0,1].set_yticklabels(self.target_names)
        plt.colorbar(im, ax=axes[0,1])
        
        # Prediction probabilities for test set
        axes[1,0].hist(y_pred_proba_multi.max(axis=1), bins=20, alpha=0.7)
        axes[1,0].set_title('Distribution of Maximum Prediction Probabilities')
        axes[1,0].set_xlabel('Max Probability')
        axes[1,0].set_ylabel('Frequency')
        
        # Feature importance bar plot (average across classes)
        avg_importance = np.mean(abs(model_multi.coef_), axis=0)
        axes[1,1].barh(self.feature_names, avg_importance)
        axes[1,1].set_title('Average Feature Importance Across Classes')
        axes[1,1].set_xlabel('Average |Coefficient|')
        
        plt.tight_layout()
        plt.show()
        
        self.model_multi = model_multi
        return model_multi
    
    def visualize_decision_boundaries(self):
        """Visualize decision boundaries for 2D projections."""
        print("\n🎨 Visualizing Decision Boundaries...")
        print("=" * 50)
        
        # Use the two most important features for visualization
        # Let's use petal length and petal width (features 2 and 3)
        X_2d = self.X[:, [2, 3]]  # Petal length and width
        
        # Split and scale 2D data
        X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
            X_2d, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        scaler_2d = StandardScaler()
        X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
        X_test_2d_scaled = scaler_2d.transform(X_test_2d)
        
        # Train model on 2D data
        model_2d = LogisticRegression(random_state=42)
        model_2d.fit(X_train_2d_scaled, y_train_2d)
        
        # Create a mesh for decision boundary
        h = 0.02
        x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
        y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundaries
        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot the data points
        scatter = plt.scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1], 
                            c=y_train_2d, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.xlabel('Petal Length (scaled)')
        plt.ylabel('Petal Width (scaled)')
        plt.title('Logistic Regression Decision Boundaries\n(Petal Length vs Petal Width)')
        
        # Add legend
        for i, species in enumerate(self.target_names):
            plt.scatter([], [], c=plt.cm.RdYlBu(i/2), label=species, edgecolors='black')
        plt.legend()
        
        plt.show()
        
        # Accuracy on 2D model
        y_pred_2d = model_2d.predict(X_test_2d_scaled)
        accuracy_2d = accuracy_score(y_test_2d, y_pred_2d)
        print(f"🎯 2D Model Accuracy: {accuracy_2d:.4f} ({accuracy_2d*100:.2f}%)")
    
    def model_interpretation(self):
        """Interpret the trained models."""
        print("\n🔍 Model Interpretation")
        print("=" * 50)
        
        if hasattr(self, 'model_multi'):
            print("Multi-class Model Coefficients:")
            print("-" * 40)
            
            coef_df = pd.DataFrame(
                self.model_multi.coef_,
                columns=self.feature_names,
                index=self.target_names
            )
            print(coef_df.round(4))
            
            print(f"\nIntercepts: {self.model_multi.intercept_.round(4)}")
            
            print("\n💡 Interpretation:")
            print("- Positive coefficients increase the probability of that class")
            print("- Negative coefficients decrease the probability of that class")
            print("- Larger absolute values indicate more important features")
    
    def run_complete_tutorial(self):
        """Run the complete tutorial."""
        print("🌸 IRIS LOGISTIC REGRESSION TUTORIAL")
        print("=" * 60)
        print("Welcome to the comprehensive Iris dataset logistic regression tutorial!")
        print("We'll explore the data, understand why logistic regression fits,")
        print("and train models with detailed visualizations.\n")
        
        # Step 1: Explore the dataset
        self.explore_dataset()
        
        # Step 2: Visualize the dataset
        self.visualize_dataset()
        
        # Step 3: Explain why logistic regression
        self.explain_logistic_regression_choice()
        
        # Step 4: Prepare data
        self.prepare_data()
        
        # Step 5: Train binary model
        self.train_binary_model()
        
        # Step 6: Train multi-class model
        self.train_multiclass_model()
        
        # Step 7: Visualize decision boundaries
        self.visualize_decision_boundaries()
        
        # Step 8: Model interpretation
        self.model_interpretation()
        
        print("\n🎉 Tutorial Complete!")
        print("=" * 50)
        print("You've successfully:")
        print("✅ Explored the Iris dataset comprehensively")
        print("✅ Understood why logistic regression is appropriate")
        print("✅ Trained both binary and multi-class models")
        print("✅ Visualized results and decision boundaries")
        print("✅ Interpreted model coefficients")
        print("\nGreat job! You now understand logistic regression deeply! 🚀")


def main():
    """Main function to run the tutorial."""
    # Create and run the tutorial
    tutorial = IrisLogisticRegressionTutorial()
    tutorial.run_complete_tutorial()


if __name__ == "__main__":
    main()
