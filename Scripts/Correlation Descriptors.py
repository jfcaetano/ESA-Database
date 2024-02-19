### Mol Correlation
### JFCAETANO 2024
### MIT Licence


def find_best_correlated_descriptors(data, target_descriptor, exclude_cols):
    """
    Finds and returns the descriptors that are most correlated with the target descriptor, excluding specified columns.

    :param data: Pandas DataFrame containing the dataset.
    :param target_descriptor: String name of the target descriptor.
    :param exclude_cols: List of column names to be excluded from the analysis.
    :return: Pandas Series with descriptors sorted by their correlation to the target.
    """
    # Exclude specified columns
    data_filtered = data.drop(columns=exclude_cols, errors='ignore')

    # Calculate correlation matrix
    correlation_matrix = data_filtered.corr()

    # Extract correlations with the target descriptor
    target_correlations = correlation_matrix[target_descriptor]

    # Remove the target descriptor from the series
    target_correlations = target_correlations.drop(target_descriptor, errors='ignore')

    # Sort by absolute correlation values in descending order
    best_correlations = target_correlations.abs().sort_values(ascending=False)

    return best_correlations

# Example usage
# Load your dataset
df = pd.read_csv('Database_ESA.csv')  # Replace with your dataset file path

# Specify your target descriptor
target_descriptor = 'Yield'  # Replace with your target descriptor's column name

# Columns to exclude
exclude_cols = ['Solvent', 'Cat_Structure', 'Catalyst', 'Substrate', 'Ligand', 'Oxidant', 'EE', 'Configuration', 'Entry']

# Find best correlated descriptors
best_correlated_descriptors = find_best_correlated_descriptors(df, target_descriptor, exclude_cols)
print(best_correlated_descriptors)
best_correlated_descriptors.to_csv('Correlated Descriptors.csv')
