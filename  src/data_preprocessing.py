import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """Load the dataset."""
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and encoding categorical variables."""
    
    # Clean column names (strip spaces and lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Handle missing values (for simplicity, we fill NaNs with the mean of the column)
    df.fillna(df.mean(), inplace=True)
    
    # If there are categorical columns, encode them (e.g., 'club' column)
    label_encoder = LabelEncoder()
    if 'club' in df.columns:
        df['club'] = label_encoder.fit_transform(df['club'])
    
    return df

def scale_data(df, columns_to_scale):
    """Scale numeric columns to a standard range."""
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df

def split_data(df, target_column, test_size=0.2):
    """Split data into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = "/Users/jhansi/Documents/FootballPlayerPositionPrediction/data/player_stats.csv"
    
    # Load and clean the data
    df = load_data(file_path)
    df = clean_data(df)
    
    # Example: scale numeric columns like 'age', 'height', 'weight', etc.
    columns_to_scale = ['age', 'height', 'weight', 'ballcontrol', 'dribbling', 'aggression']
    df = scale_data(df, columns_to_scale)
    
    # Split data
    target_column = 'age'  # Example target column (change as needed)
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    print("Data processing completed. The cleaned and scaled data is ready for clustering or further analysis.")
