import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data_path = '../data/player_stats.csv'  # Adjust path if needed
df = pd.read_csv(data_path)

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

print("\nDataset Head:")
print(df.head())

# Visualization 1: 3D scatter plot for short_pass, long_pass, and crossing
fig = px.scatter_3d(
    df, 
    x='short_pass', 
    y='long_pass', 
    z='crossing', 
    color='value', 
    color_continuous_scale='Viridis',
    title='3D Scatter Plot: Passing and Crossing'
)
fig.show()

# Visualization 2: 3D scatter plot for shot_power, long_shots, and finishing
fig = px.scatter_3d(
    df, 
    x='shot_power', 
    y='long_shots', 
    z='finishing', 
    color='value', 
    color_continuous_scale='Viridis',
    title='3D Scatter Plot: Shooting Attributes'
)
fig.show()

# Visualization 3: Pairplot for goalkeeper-related attributes
columns_to_visualize = ['gk_positioning', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes']

data = df[columns_to_visualize].dropna()
if not data.empty:
    sns.pairplot(data)
    plt.suptitle('Goalkeeper Attributes Pairplot', y=1.02)
    plt.show()
else:
    print("No data available for goalkeeper attributes.")

# Visualization 4: Heatmap of correlations for goalkeeper attributes
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap: Goalkeeper Attributes')
plt.show()

# Visualization 5: Distribution histograms
histogram_labels = list(df.columns)
values_to_remove = ['player', 'country', 'club', 'value']
histogram_labels = [col for col in histogram_labels if col not in values_to_remove]

colors = (
    px.colors.qualitative.Plotly +
    px.colors.qualitative.Dark24 +
    px.colors.qualitative.Light24 +
    px.colors.qualitative.Pastel1 +
    px.colors.qualitative.Pastel2
)

for i, label in enumerate(histogram_labels):
    fig = px.histogram(
        df, 
        x=label, 
        title=f'{label} Distribution',
        color_discrete_sequence=[colors[i % len(colors)]]
    )
    fig.show()
