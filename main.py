import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import time
from imblearn.over_sampling import SMOTE
file_path = "/Users/jhansi/Documents/Football Player Position Prediction and Performance Analysis/data/player_stats.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")
sns.set(style="whitegrid", palette="muted")
sns.set(style="whitegrid", palette="muted")
df.columns = df.columns.str.strip().str.lower()
if 'player' in df.columns:  
    df['player'] = df['player'].str.strip() 
print("Cleaned Column Names:")
print(df.columns)
print(df.head())

if 'nationality' in df.columns:
    country_col = 'nationality'
elif 'country' in df.columns:
    country_col = 'country'
else:
    raise ValueError("Column for nationality or country not found in the dataset.")
target_countries = [
    "England", "Germany", "Spain", "France", "Argentina", "Brazil",
    "Italy", "Netherlands", "Sweden", "Republic of Ireland", "China PR",
    "United States", "Norway", "Saudi Arabia", "Poland", "Romania", "Portugal",
    "Denmark", "Korea Republic", "Australia", "Belgium", "Austria", "Scotland",
    "Turkey", "Colombia", "Uruguay", "Switzerland", "Paraguay", "India", "Chile"
]
filtered_data = df[df[country_col].isin(target_countries)]
country_counts = filtered_data[country_col].value_counts()
country_counts = country_counts.reindex(target_countries)
country_df = pd.DataFrame({
    'Country': country_counts.index,
    'Players': country_counts.values
})
plt.figure(figsize=(14, 8))
sns.barplot(x='Country', y='Players', data=country_df, hue='Country', palette='coolwarm')
for index, row in country_df.iterrows():
    plt.text(index, row.Players + 1, str(int(row.Players)), ha='center', fontsize=9)
plt.title('Number of Players by Country', fontsize=16, weight='bold')
plt.xlabel('Country', fontsize=14)
plt.ylabel('Sum', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['age'], bins=20, kde=True, color="skyblue")
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.subplot(1, 3, 2)
sns.histplot(df['height'], bins=20, kde=True, color="orange")
plt.title('Height Distribution')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.subplot(1, 3, 3)
sns.histplot(df['weight'], bins=20, kde=True, color="green")
plt.title('Weight Distribution')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

top_clubs = df['club'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_clubs.values, y=top_clubs.index, hue=top_clubs.index, palette="viridis") 
plt.title("Top 10 Clubs by Player Count")
plt.xlabel("Number of Players")
plt.ylabel("Club")
plt.tight_layout()
plt.show()
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=df['ball_control'],  
    y=df['dribbling'],     
    z=df['aggression'],    
    mode='markers',
    marker=dict(
        size=6, 
        color=df['aggression'], 
        colorscale='Viridis', 
        opacity=0.8
    ),
    text=[f"Player: {player}" for player in df['player']] 
))
field_length = 120
field_width = 80
for y in np.linspace(0, field_length, 13): 
    fig.add_trace(go.Scatter3d(
        x=[0, field_width],
        y=[y, y],
        z=[0, 0],  
        mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))

for x in np.linspace(0, field_width, 9): 
    fig.add_trace(go.Scatter3d(
        x=[x, x],
        y=[0, field_length],
        z=[0, 0],  
        mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))
goal_width = field_width * 0.2
goal_depth = 6  
fig.add_trace(go.Scatter3d(
    x=[(field_width - goal_width) / 2, (field_width + goal_width) / 2],
    y=[0, 0],
    z=[0, 0],
    mode='lines',
    line=dict(color='red', width=3),
    showlegend=False
))
fig.add_trace(go.Scatter3d(
    x=[(field_width - goal_width) / 2, (field_width + goal_width) / 2],
    y=[field_length, field_length],
    z=[0, 0],
    mode='lines',
    line=dict(color='red', width=3),
    showlegend=False
))
fig.add_trace(go.Scatter3d(
    x=[(field_width - goal_width) / 2, (field_width - goal_width) / 2],
    y=[0, goal_depth],
    z=[0, 0],
    mode='lines',
    line=dict(color='red', width=3),
    showlegend=False
))
fig.add_trace(go.Scatter3d(
    x=[(field_width + goal_width) / 2, (field_width + goal_width) / 2],
    y=[0, goal_depth],
    z=[0, 0],
    mode='lines',
    line=dict(color='red', width=3),
    showlegend=False
))
for x in np.linspace(0, field_width, 4):
    for y in np.linspace(0, field_length, 4):
        fig.add_trace(go.Scatter3d(
            x=[x, x],
            y=[y, y],
            z=[0, 100], 
            mode='lines',
            line=dict(color='white', width=1, dash='dot'),
            showlegend=False
        ))
fig.update_layout(
    scene=dict(
        xaxis=dict(title="Ball Control", range=[0, field_width]),
        yaxis=dict(title="Dribbling", range=[0, field_length]),
        zaxis=dict(title="Aggression", range=[0, 100]),
        aspectmode="manual",
        aspectratio=dict(x=4, y=6, z=1.5),  
        xaxis_showspikes=False,
        yaxis_showspikes=False,
        zaxis_showspikes=False,
        bgcolor="green" 
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    title=dict(
        text="3D Football Field: Ball Control, Dribbling, Aggression",
        font=dict(size=16)
    )
)
fig.show()

fig = go.Figure(data=[go.Scatter3d(
    x=df['ball_control'],
    y=df['dribbling'],
    z=df['aggression'],
    mode='markers',
    marker=dict(
        size=6,
        color=df['aggression'],
        colorscale='Rainbow',  
        opacity=0.8
    ),
    text=[f"Player: {player}" for player in df['player']]
)])

fig.update_layout(
    scene=dict(
        xaxis=dict(title=dict(text="Ball Control", font=dict(size=12))),
        yaxis=dict(title=dict(text="Dribbling", font=dict(size=12))),
        zaxis=dict(title=dict(text="Aggression", font=dict(size=12))),
    ),
    title=dict(text="3D Scatter Plot for Ball Control, Dribbling, and Aggression",
               font=dict(size=16)),
    margin=dict(l=0, r=0, b=0, t=40) 
)

fig.show()

fig = go.Figure(data=[go.Scatter3d(
    x=df['slide_tackle'],
    y=df['stand_tackle'],
    z=df['interceptions'],
    mode='markers',
    marker=dict(
        size=6,
        color=df['interceptions'],  
        colorscale='Rainbow',  
        opacity=0.8
    ),
    text=[f"Player: {player}" for player in df['player']]
)])

fig.update_layout(
    scene=dict(
        xaxis=dict(title=dict(text="Slide Tackle", font=dict(size=12))),
        yaxis=dict(title=dict(text="Stand Tackle", font=dict(size=12))),
        zaxis=dict(title=dict(text="Interceptions", font=dict(size=12))),
    ),
    title=dict(text="3D Scatter Plot for Slide Tackle, Stand Tackle, and Interceptions",
               font=dict(size=16)),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()

fig = go.Figure(data=[go.Scatter3d(
    x=df['short_pass'],
    y=df['long_pass'],
    z=df['shot_power'],
    mode='markers',
    marker=dict(
        size=6,
        color=df['shot_power'],  
        colorscale='Rainbow',  
        opacity=0.8
    ),
    text=[f"Player: {player}" for player in df['player']]
)])

fig.update_layout(
    scene=dict(
        xaxis=dict(title=dict(text="Short Pass", font=dict(size=12))),
        yaxis=dict(title=dict(text="Long Pass", font=dict(size=12))),
        zaxis=dict(title=dict(text="Shot Power", font=dict(size=12))),
    ),
    title=dict(text="3D Scatter Plot for Short Pass, Long Pass, and Shot Power",
               font=dict(size=16)),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()

fig = go.Figure(data=[go.Scatter3d(
    x=df['gk_positioning'],
    y=df['gk_diving'],
    z=df['gk_handling'],
    mode='markers',
    marker=dict(
        size=6,
        color=df['gk_handling'],  
        colorscale='Rainbow',  
        opacity=0.8
    ),
    text=[f"Player: {player}" for player in df['player']]
)])

fig.update_layout(
    scene=dict(
        xaxis=dict(title=dict(text="GK Positioning", font=dict(size=12))),
        yaxis=dict(title=dict(text="GK Diving", font=dict(size=12))),
        zaxis=dict(title=dict(text="GK Handling", font=dict(size=12))),
    ),
    title=dict(text="3D Scatter Plot for GK Positioning, Diving, and Handling",
               font=dict(size=16)),
    margin=dict(l=0, r=0, b=0, t=40) 
)

fig.show()

numerical_df = df.select_dtypes(include=['float64', 'int64'])
correlation = numerical_df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".1f", annot_kws={'size': 8}, cbar_kws={'shrink': 0.8}, linewidths=0.5)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("Handling missing values...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        print(f"Imputing missing values in column: {col}")
        if df[col].isnull().all():
            df[col] = 0 
        else:
            df[col] = df[col].fillna(df[col].median())
print(f"Missing values handled. Current dataset size: {df.shape}")
target_column = 'ball_control' 
if target_column not in df.columns:
    raise ValueError(f"'{target_column}' column is missing from the dataset.")

if not pd.api.types.is_numeric_dtype(df[target_column]):
    print(f"Converting '{target_column}' column to numeric...")
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    if df[target_column].isnull().any():
        raise ValueError(f"Non-convertible values found in '{target_column}' column after conversion.")
print(f"'{target_column}' column successfully processed and cleaned.")
min_samples = 10
rare_classes = df[target_column].value_counts()[df[target_column].value_counts() < min_samples].index
if len(rare_classes) > 0:
    print(f"Removing rare classes with fewer than {min_samples} samples: {list(rare_classes)}")
    df = df[~df[target_column].isin(rare_classes)]
    print(f"Dataset size after removing rare classes: {df.shape}")
print("Target distribution after removing rare classes:")
print(df[target_column].value_counts())
categorical_columns = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_columns:
    df.loc[:, col] = encoder.fit_transform(df[col])
X = df.drop(columns=[target_column])
y = df[target_column]
print("Checking for NaN or infinite values in the dataset...")
X = X.replace([np.inf, -np.inf], np.nan) 
X = X.fillna(0) 
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
except ValueError as e:
    print(f"Error during train-test split: {e}")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling complete.")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

print("Starting hyperparameter tuning...")
start_time = time.time()
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=10, 
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
end_time = time.time()
print(f"Hyperparameter tuning completed in {end_time - start_time:.2f} seconds.")
best_model = random_search.best_estimator_
print("Best parameters found:", random_search.best_params_)
best_model.fit(X_train, y_train)
print("Model training complete.")
y_pred = best_model.predict(X_test)
print("Prediction complete.")
conf_matrix = confusion_matrix(y_test, y_pred)
top_n = 10  
class_counts = pd.Series(y_test).value_counts()
top_classes = class_counts[:top_n].index
filtered_indices = [i for i, label in enumerate(best_model.classes_) if label in top_classes]
filtered_conf_matrix = conf_matrix[np.ix_(filtered_indices, filtered_indices)]
plt.figure(figsize=(12, 10))
sns.heatmap(
    filtered_conf_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=top_classes,
    yticklabels=top_classes,
    cbar_kws={'shrink': 0.8},
    linewidths=2,
    linecolor='white',
    annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'}
)

plt.title(f'Normalized Confusion Matrix (Top {top_n} Classes)', fontsize=18, weight='bold', pad=20)
plt.xlabel('Predicted Labels', fontsize=14, labelpad=10)
plt.ylabel('True Labels', fontsize=14, labelpad=10)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()

performance_columns = [
    'ball_control', 'dribbling', 'aggression', 'slide_tackle', 'stand_tackle',
    'interceptions', 'short_pass', 'long_pass', 'shot_power', 'gk_positioning',
    'gk_diving', 'gk_handling'
]
missing_cols = [col for col in performance_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in the dataset: {missing_cols}")
if 'player' not in df.columns:
    raise ValueError("'player' column is missing in the dataset.")

df['performance_avg'] = df[performance_columns].mean(axis=1)

df['performance_avg'] = df['performance_avg'].round(2)
sorted_df = df[['player', 'performance_avg']].sort_values(by='performance_avg', ascending=False)

plt.figure(figsize=(16, 8))  
sns.barplot(
    x='performance_avg',
    y='player',
    data=sorted_df.head(15),
    palette='Set2',  
    ci=None
)
plt.title('Top 15 Players Based on Performance Metrics Average', fontsize=18, weight='bold')
plt.xlabel('Average Performance Score', fontsize=14)
plt.ylabel('Player', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


