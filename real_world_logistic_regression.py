"""
Real-World Logistic Regression: Customer Purchase Prediction
===========================================================

This script demonstrates a complete machine learning pipeline using logistic regression
to predict whether a customer will make a purchase based on their characteristics.

Steps covered:
1. Data Generation (simulating real-world customer data)
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Model Training
5. Model Evaluation
6. Results Interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerPurchasePrediction:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def generate_realistic_data(self, n_samples=1000):
        """
        Step 1: Generate realistic customer data
        
        Features:
        - Age: Customer age
        - Income: Annual income
        - Time_on_site: Minutes spent on website
        - Previous_purchases: Number of previous purchases
        - Email_opens: Number of marketing emails opened
        - Device_type: Mobile, Desktop, Tablet
        - Campaign_type: Email, Social, Search, Direct
        """
        print("🔄 Step 1: Generating realistic customer data...")
        
        np.random.seed(42)
        
        # Generate features with realistic distributions
        age = np.random.normal(35, 12, n_samples).clip(18, 80)
        income = np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000)
        time_on_site = np.random.exponential(5, n_samples).clip(0.5, 60)
        previous_purchases = np.random.poisson(2, n_samples).clip(0, 20)
        email_opens = np.random.poisson(3, n_samples).clip(0, 50)
        
        # Categorical features
        device_types = np.random.choice(['Mobile', 'Desktop', 'Tablet'], 
                                      n_samples, p=[0.6, 0.3, 0.1])
        campaign_types = np.random.choice(['Email', 'Social', 'Search', 'Direct'], 
                                        n_samples, p=[0.4, 0.3, 0.2, 0.1])
        
        # Create target variable with realistic relationships
        # Higher probability of purchase for:
        # - Higher income, more time on site, more previous purchases, more email opens
        purchase_probability = (
            0.1 +  # Base probability
            0.3 * (income > 50000) +  # Higher income
            0.2 * (time_on_site > 10) +  # More time on site
            0.3 * (previous_purchases > 1) +  # Repeat customers
            0.1 * (email_opens > 5) +  # Engaged with emails
            0.1 * (device_types == 'Desktop') +  # Desktop users
            0.05 * (age > 30) * (age < 50)  # Prime age group
        )
        
        # Add some noise and ensure probabilities are valid
        purchase_probability = np.clip(purchase_probability + np.random.normal(0, 0.1, n_samples), 0, 1)
        
        # Generate binary target
        purchased = np.random.binomial(1, purchase_probability, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'income': income,
            'time_on_site': time_on_site,
            'previous_purchases': previous_purchases,
            'email_opens': email_opens,
            'device_type': device_types,
            'campaign_type': campaign_types,
            'purchased': purchased
        })
        
        print(f"✅ Generated {n_samples} customer records")
        print(f"📊 Purchase rate: {data['purchased'].mean():.1%}")
        
        return data
    
    def explore_data(self, data):
        """
        Step 2: Exploratory Data Analysis
        """
        print("\n🔍 Step 2: Exploratory Data Analysis...")
        
        # Basic info
        print(f"\nDataset shape: {data.shape}")
        print(f"\nData types:\n{data.dtypes}")
        print(f"\nMissing values:\n{data.isnull().sum()}")
        
        # Summary statistics
        print(f"\nSummary statistics:")
        print(data.describe())
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Customer Data Exploration', fontsize=16, fontweight='bold')
        
        # Age distribution by purchase
        sns.histplot(data=data, x='age', hue='purchased', bins=20, ax=axes[0,0])
        axes[0,0].set_title('Age Distribution by Purchase')
        
        # Income distribution by purchase
        sns.histplot(data=data, x='income', hue='purchased', bins=20, ax=axes[0,1])
        axes[0,1].set_title('Income Distribution by Purchase')
        
        # Time on site by purchase
        sns.boxplot(data=data, x='purchased', y='time_on_site', ax=axes[0,2])
        axes[0,2].set_title('Time on Site by Purchase')
        
        # Device type purchase rates
        purchase_by_device = data.groupby('device_type')['purchased'].mean()
        purchase_by_device.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Purchase Rate by Device Type')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Campaign type purchase rates
        purchase_by_campaign = data.groupby('campaign_type')['purchased'].mean()
        purchase_by_campaign.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Purchase Rate by Campaign Type')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('/Users/vikram/Desktop/projects/logistic_regression_to_neural_network/eda_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ EDA complete - visualizations saved as 'eda_analysis.png'")
    
    def preprocess_data(self, data):
        """
        Step 3: Data Preprocessing
        """
        print("\n🔧 Step 3: Data Preprocessing...")
        
        # Separate features and target
        X = data.drop('purchased', axis=1)
        y = data['purchased']
        
        # Handle categorical variables
        categorical_cols = ['device_type', 'campaign_type']
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            print(f"📝 Encoded {col}: {list(le.classes_)}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Preprocessing complete")
        print(f"📋 Features: {self.feature_names}")
        
        return X, y
    
    def split_and_scale_data(self, X, y, test_size=0.2):
        """
        Step 4: Split data and scale features
        """
        print(f"\n📊 Step 4: Splitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Data split complete")
        print(f"📈 Training set: {X_train_scaled.shape[0]} samples")
        print(f"📉 Test set: {X_test_scaled.shape[0]} samples")
        print(f"⚖️ Train purchase rate: {y_train.mean():.1%}")
        print(f"⚖️ Test purchase rate: {y_test.mean():.1%}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Step 5: Train the logistic regression model
        """
        print("\n🚀 Step 5: Training Logistic Regression Model...")
        
        # Create and train the model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train, y_train)
        
        print("✅ Model training complete!")
        
        # Display model coefficients
        print(f"\n📊 Model Coefficients:")
        for feature, coef in zip(self.feature_names, self.model.coef_[0]):
            print(f"  {feature}: {coef:.4f}")
        print(f"  Intercept: {self.model.intercept_[0]:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        Step 6: Evaluate the model
        """
        print("\n📈 Step 6: Model Evaluation...")
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        auc_score = roc_auc_score(y_test, y_test_proba)
        
        print(f"🎯 Training Accuracy: {train_accuracy:.3f}")
        print(f"🎯 Test Accuracy: {test_accuracy:.3f}")
        print(f"📊 AUC Score: {auc_score:.3f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # Create evaluation plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True)
        
        # Feature Importance
        feature_importance = abs(self.model.coef_[0])
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        axes[2].barh(feature_df['feature'], feature_df['importance'])
        axes[2].set_title('Feature Importance (|Coefficients|)')
        axes[2].set_xlabel('Absolute Coefficient Value')
        
        plt.tight_layout()
        plt.savefig('/Users/vikram/Desktop/projects/logistic_regression_to_neural_network/model_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Model evaluation complete - results saved as 'model_evaluation.png'")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'auc_score': auc_score,
            'predictions': y_test_pred,
            'probabilities': y_test_proba
        }
    
    def interpret_results(self, results):
        """
        Step 7: Interpret and explain results
        """
        print("\n🧠 Step 7: Results Interpretation...")
        
        print(f"\n📊 Model Performance Summary:")
        print(f"  • The model achieved {results['test_accuracy']:.1%} accuracy on unseen data")
        print(f"  • AUC score of {results['auc_score']:.3f} indicates {'excellent' if results['auc_score'] > 0.8 else 'good' if results['auc_score'] > 0.7 else 'fair'} predictive power")
        
        # Interpret coefficients
        print(f"\n🔍 Key Insights from Model Coefficients:")
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        for _, row in coef_df.head(3).iterrows():
            direction = "increases" if row['coefficient'] > 0 else "decreases"
            print(f"  • {row['feature']}: {direction} purchase probability (coef: {row['coefficient']:.3f})")
        
        print(f"\n💡 Business Recommendations:")
        print(f"  • Focus marketing efforts on high-impact features")
        print(f"  • Consider A/B testing campaigns based on model insights")
        print(f"  • Monitor model performance over time and retrain as needed")
        
    def run_complete_pipeline(self, n_samples=1000):
        """
        Run the complete machine learning pipeline
        """
        print("🚀 Starting Complete Logistic Regression Pipeline")
        print("=" * 60)
        
        # Step 1: Generate data
        data = self.generate_realistic_data(n_samples)
        
        # Step 2: Explore data
        self.explore_data(data)
        
        # Step 3: Preprocess data
        X, y = self.preprocess_data(data)
        
        # Step 4: Split and scale data
        X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)
        
        # Step 5: Train model
        self.train_model(X_train, y_train)
        
        # Step 6: Evaluate model
        # results = self.evaluate_model(X_train, X_test, y_train, y_test)
        
        # Step 7: Interpret results
        # self.interpret_results(results)
        
        print("\n🎉 Pipeline Complete!")
        print("=" * 60)
        
        return data, results

def main():
    """
    Main function to run the customer purchase prediction pipeline
    """
    # Create the prediction system
    predictor = CustomerPurchasePrediction()
    
    # Run the complete pipeline
    data, results = predictor.run_complete_pipeline(n_samples=1500)
    
    print(f"\n🎯 Final Results:")
    print(f"  • Dataset: {len(data)} customers")
    print(f"  • Test Accuracy: {results['test_accuracy']:.1%}")
    print(f"  • AUC Score: {results['auc_score']:.3f}")
    
    # Example prediction for a new customer
    print(f"\n🔮 Example: Predicting for a new customer...")
    
    # Create a sample customer (you would normally get this from user input)
    new_customer = np.array([[35, 65000, 15, 3, 8, 1, 0]])  # Desktop user from email campaign
    
    # Scale the features
    new_customer_scaled = predictor.scaler.transform(new_customer)
    
    # Make prediction
    prediction = predictor.model.predict(new_customer_scaled)[0]
    probability = predictor.model.predict_proba(new_customer_scaled)[0, 1]
    
    print(f"  • Prediction: {'Will Purchase' if prediction == 1 else 'Will Not Purchase'}")
    print(f"  • Confidence: {probability:.1%}")

if __name__ == "__main__":
    main()
