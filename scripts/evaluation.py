import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import kruskal, chi2_contingency
import numpy as np
import statsmodels.formula.api as smf

def preprocess_data(impact_matrix: pd.DataFrame, residual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the impact matrix and residual data for further analysis.

    Args:
        impact_matrix (pd.DataFrame): The impact matrix with dates as index.
        residual_df (pd.DataFrame): The DataFrame containing residual data.

    Returns:
        pd.DataFrame: The merged and preprocessed data.
    """
    non_zero_impact = impact_matrix.stack().reset_index()
    non_zero_impact.columns = ['date', 'company', 'impact_score']
    non_zero_impact = non_zero_impact[non_zero_impact['impact_score'] != 0]

    non_zero_impact['date'] = pd.to_datetime(non_zero_impact['date'], format='%Y-%m-%d')
    residual_df['YYYYMMDD'] = pd.to_datetime(residual_df['YYYYMMDD'], format='%Y-%m-%d')

    merged_df = pd.merge(non_zero_impact, residual_df, left_on=['date', 'company'], right_on=['YYYYMMDD', 'Ticker'])
    merged_df = merged_df.dropna(subset=['impact_score', '__1_CAMP', '__2_fama3', '__3_fama5'])
    merged_df['impact_score'] = pd.to_numeric(merged_df['impact_score'], errors='coerce')
    merged_df['__1_CAMP'] = pd.to_numeric(merged_df['__1_CAMP'], errors='coerce')
    merged_df.to_csv('merged_data.csv')
    return merged_df

def polynomial_regression(y: pd.Series, X: pd.Series, degree: int, filename: str):
    """
    Performs polynomial regression and saves the model summary.

    Args:
        y (pd.Series): The dependent variable.
        X (pd.Series): The independent variable.
        degree (int): The degree of the polynomial.
        filename (str): Filename to save the regression results.
    """
    X_poly = np.column_stack([X ** i for i in range(1, degree + 1)])
    X_poly = sm.add_constant(X_poly)

    model = sm.OLS(y, X_poly).fit()

    with open(filename, 'w') as f:
        f.write(model.summary().as_text())
    print(f"Polynomial regression (degree {degree}) results saved to {filename}")

def anova_analysis(merged_df: pd.DataFrame, column_name: str, filename: str):
    """
    Performs one-way ANOVA and saves the results.

    Args:
        merged_df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name to analyze.
        filename (str): Filename to save the ANOVA results.
    """
    value_counts = merged_df['impact_score'].value_counts()
    valid_groups = [i for i in merged_df['impact_score'].unique() if value_counts.get(i, 0) > 1]

    with open(filename, 'w') as f:
        if len(valid_groups) < 2:
            f.write("Not enough valid groups to perform ANOVA.")
        else:
            groups = [merged_df[merged_df['impact_score'] == i][column_name].dropna() for i in valid_groups]
            f_stat, p_value = stats.f_oneway(*groups)

            f.write(f'F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}\n')
            if p_value < 0.05:
                f.write(f"Different impact levels have significant effects on {column_name} (p < 0.05)\n")
            else:
                f.write(f"Different impact levels do not have significant effects on {column_name} (p >= 0.05)\n")
    print(f"ANOVA results saved to {filename}")

def kruskal_wallis_test(merged_df: pd.DataFrame, column_name: str, filename: str):
    """
    Performs Kruskal-Wallis test and saves the results.

    Args:
        merged_df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name to analyze.
        filename (str): Filename to save the Kruskal-Wallis test results.
    """
    value_counts = merged_df['impact_score'].value_counts()
    valid_groups = [i for i in merged_df['impact_score'].unique() if value_counts.get(i, 0) > 1]

    with open(filename, 'w') as f:
        if len(valid_groups) < 2:
            f.write("Not enough valid groups to perform Kruskal-Wallis test.")
        else:
            groups = [merged_df[merged_df['impact_score'] == i][column_name].dropna() for i in valid_groups]
            h_stat, p_value = kruskal(*groups)

            f.write(f'H-statistic: {h_stat:.4f}, p-value: {p_value:.4f}\n')
            if p_value < 0.05:
                f.write(f"Different impact levels have significant effects on {column_name} (p < 0.05)\n")
            else:
                f.write(f"Different impact levels do not have significant effects on {column_name} (p >= 0.05)\n")
    print(f"Kruskal-Wallis test results saved to {filename}")

def categorical_regression(merged_df: pd.DataFrame, column_name: str, filename: str):
    """
    Performs categorical regression and saves the model summary.

    Args:
        merged_df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name to analyze.
        filename (str): Filename to save the regression results.
    """
    merged_df['impact_score'] = merged_df['impact_score'].astype('category')
    formula = f'{column_name} ~ C(impact_score)'
    model = smf.ols(formula=formula, data=merged_df).fit()

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(model.summary().as_text())
        f.write(f"\nExplained variance R²: {model.rsquared:.4f}")
    print(f"Categorical regression results saved to {filename}")

def chi_square_test(merged_df: pd.DataFrame, filename: str):
    """
    Performs chi-square test and saves the results.

    Args:
        merged_df (pd.DataFrame): The DataFrame containing the data.
        filename (str): Filename to save the chi-square test results.
    """
    contingency_table = pd.crosstab(merged_df['impact_score'], merged_df['company'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f'Chi-square statistic: {chi2:.4f}, p-value: {p_value:.4f}\n')
        if p_value < 0.05:
            f.write("Significant differences in impact score distribution across companies (p < 0.05)\n")
        else:
            f.write("No significant differences in impact score distribution across companies (p >= 0.05)\n")
    print(f"Chi-square test results saved to {filename}")

def calculate_eta_squared(merged_df: pd.DataFrame, column_name: str, filename: str):
    """
    Calculates eta squared (η²) from one-way ANOVA and saves the results.

    Args:
        merged_df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name to analyze.
        filename (str): Filename to save the eta squared results.
    """
    value_counts = merged_df['impact_score'].value_counts()
    valid_groups = [i for i in merged_df['impact_score'].unique() if value_counts.get(i, 0) > 1]

    with open(filename, 'w', encoding='utf-8') as f:
        if len(valid_groups) < 2:
            f.write("Not enough valid groups to perform ANOVA, cannot calculate eta squared.")
        else:
            groups = [merged_df[merged_df['impact_score'] == i][column_name].dropna() for i in valid_groups]
            f_stat, p_value = stats.f_oneway(*groups)

            grand_mean = merged_df[column_name].mean()
            ss_between = sum(len(group) * ((group.mean() - grand_mean) ** 2) for group in groups)
            ss_total = sum((merged_df[column_name] - grand_mean) ** 2)

            eta_squared = ss_between / ss_total
            f.write(f'Eta squared (η²): {eta_squared:.4f}\n')
    print(f"Eta squared calculation results saved to {filename}")

# Main processing
impact_matrix = pd.read_csv('gpt35/output.csv', index_col=0)
residual_df = pd.read_csv('../timeseries.csv')
model = 'gpt35/'

# Preprocess data
merged_df = preprocess_data(impact_matrix, residual_df)

# Polynomial regression analysis
polynomial_regression(merged_df['__1_CAMP'], merged_df['impact_score'], 1, model+'capm_poly_regression_results.txt')
polynomial_regression(merged_df['__2_fama3'], merged_df['impact_score'], 1, model+'fama3_poly_regression_results.txt')
polynomial_regression(merged_df['__3_fama5'], merged_df['impact_score'], 1, model+'fama5_poly_regression_results.txt')

# ANOVA analysis
anova_analysis(merged_df, '__2_fama3', model+'anova_fama3_results.txt')

# Kruskal-Wallis test
kruskal_wallis_test(merged_df, '__2_fama3', model+'kruskal_wallis_fama3_results.txt')

# Categorical regression
categorical_regression(merged_df, '__2_fama3', model+'categorical_regression_fama3_results.txt')

# Chi-square test
chi_square_test(merged_df, model+'chi_square_test_results.txt')

# Calculate eta squared
calculate_eta_squared(merged_df, '__2_fama3', model+'eta_squared_fama3_results.txt')

