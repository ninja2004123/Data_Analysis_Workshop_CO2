import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colour import Color
from scipy import stats




def add_missing_one_year(df):
  ''' # function for adding rows from the column '1 year ago' where the calculated
        dates do not coencide with the existing rows - gives us more accurate data in terms of trends and movement'''

  df = df.sort_index()
  new_rows = []

  for current_date, row in df.iterrows():
      one_year_ago_date = current_date - pd.DateOffset(years=1)

      if one_year_ago_date not in df.index:
        new_row = {
                'median_CO2': row['1 year ago'],
                '1 year ago': None,  # This would be blank for the new row
                '10 years ago': None,  # This would be blank for the new row
            }
        new_rows.append((one_year_ago_date, new_row))

  new_rows_df = pd.DataFrame([r[1] for r in new_rows], index=[r[0] for r in new_rows])
  df = pd.concat([df, new_rows_df]).sort_index()

  return df


def add_missing_ten_years(df):
  '''inserting new rows with data from '10 years ago' column' to give us a richer timeframe'''
  df = df.sort_index()

  placeholder_mask = (df['10 years ago'] == -999)

  shifted_average = df['median_CO2'].copy()
  shifted_average.index = shifted_average.index - pd.DateOffset(years=10)

  aligned_shifted_average = shifted_average.reindex(df.index)
  df.loc[placeholder_mask, '10 years ago'] = aligned_shifted_average[placeholder_mask]
  df.loc[placeholder_mask, ['median_CO2', '10 years ago']]

  return df

def coerce_into_full_datetime(df):
  ''' coercing datetime indexed column from separate non numeric columns '''

  df['year'] = pd.to_numeric(df['year'], errors='coerce')
  df['month'] = pd.to_numeric(df['month'], errors='coerce')
  df['day'] = pd.to_numeric(df['day'], errors='coerce')
  df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')

  df.set_index('datetime', inplace=True)
  df.drop(columns = ['year', 'month', 'day'], inplace = True)

  return df


def plot_column(df, column_name, colour):
  '''simple plot of column over TIME'''

  plt.figure(figsize=(17, 6))
  plt.plot(df.index, df[column_name], label=column_name, color=colour)
  plt.grid(
    color="gray",      
    linestyle="--",   
    linewidth=0.5,   
    alpha=0.7)
  plt.title(f'{column_name} Over Time')
  plt.xlabel('Date')
  plt.ylabel(column_name)
  plt.legend()
  plt.show()

def plot_entire_df(df):
  '''median_CO2Plot chosen columns from a dataframe over time'''

  columns_to_plot = ['median_CO2', 'temperature_2m (°C)', 'relative_humidity_2m (%)', 'dew_point_2m (°C)',
                      'precipitation (mm)', 'pressure_msl (hPa)',
                      'et0_fao_evapotranspiration (mm)', 'wind_speed_10m (km/h)',
                      'soil_temperature_0_to_7cm (°C)', 'surface_pressure (hPa)']

  plt.figure(figsize=(17, 26))

  for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(10, 1, i)
    plt.plot(df.index, df[col], label=col, color='b')  # Plot each column
    plt.xlabel("Datetime")
    plt.ylabel("levels")
    plt.title(f"{col} Over Time")
    plt.legend()
    plt.grid(True)

  plt.tight_layout()  # Adjust layout to prevent overlapping
  plt.show()

def plot_increase_five_years(df, column, gas, color1, color2):
    '''function for plotting the increase of levels through the five_years'''
    df_viz = df.reset_index()  # Reset the index if needed
    df_viz.rename(columns={'index': 'datetime'}, inplace=True)

    df_viz['datetime'] = pd.to_datetime(df_viz['datetime'], errors='coerce')
    df_viz['five_year'] = (df_viz['datetime'].dt.year // 5) * 5

    # Group by five_year and calculate the mean CO2 levels for each five_year
    five_year_avg_gas = df_viz.groupby('five_year')[column].mean().reset_index()


    plt.figure(figsize=(16, 6))
    colors = list(Color(color1).range_to(Color(color2), 10))
    colors = [str(color) for color in colors]
    bars = plt.bar(five_year_avg_gas['five_year'], five_year_avg_gas[column], color=colors, width=4, edgecolor='black')

    plt.title(f'Layered Level Chart of {gas} Levels by Five Years', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(f'{gas} Levels (ppm)', fontsize=12)
    plt.xticks(five_year_avg_gas['five_year'], labels=[f"{int(year)}s" for year in five_year_avg_gas['five_year']])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    legend_labels = [f"{int(year)}s" for year in five_year_avg_gas['five_year']]
    plt.legend(bars, legend_labels, title='Five Years', loc='upper left')

    plt.tight_layout()
    plt.show()




def summary_stats(df):
    results = []
    
    for col in df.columns:
        # Skip non-numeric columns
        if not np.issubdtype(df[col].dtype, np.number):
            continue
            
        col_data = df[col].dropna()
        
        # Calculate statistics
        stats_dict = {
            'column': col,
            'mean': np.mean(col_data),
            'median': np.median(col_data),
            'mode': stats.mode(col_data, keepdims=True).mode[0],  # Returns most frequent value
            'std': np.std(col_data, ddof=1),  # Sample standard deviation
            'var': np.var(col_data, ddof=1),  # Sample variance
            'range': np.ptp(col_data),  # Peak-to-peak (max - min)
            'IQR': stats.iqr(col_data)  # Interquartile range
        }
        results.append(stats_dict)
    
    # Create DataFrame from results
    stats_df = pd.DataFrame(results)
    
    # Set column order
    stats_df = stats_df[['column', 'mean', 'median', 'mode', 'std', 'var', 'range', 'IQR']]
    
    return stats_df



def create_heatmaps(*dfs):
    if len(dfs) == 1:
        size = (12, 12)
    else:
        size = (25, 12)

    fig, axes = plt.subplots(1, len(dfs), figsize=size)

    if (len(dfs) == 1):
        axes = [axes]

    for i, correlation_info in enumerate(dfs):
        corr, name = correlation_info

        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[i])
        axes[i].set_title(name)
    

    plt.tight_layout()
    plt.show()


def plot_rolling_correlations(df):
    ''' plotting rolling corelations between the CO2 column and other columns'''

    rolling_corr_30 = df.rolling(window=30).corr(df["median_CO2"])
    rolling_corr_365 = df.rolling(window=365).corr(df["median_CO2"])

    # Plot individual rolling correlations for both window sizes
    for col in df.columns:
        if col != "median_CO2":
            plt.figure(figsize=(16, 5))
            plt.plot(rolling_corr_30.index, rolling_corr_30[col], label=f"30-day Corr with {col}", color="lightblue", alpha=0.7)
            plt.plot(rolling_corr_365.index, rolling_corr_365[col], label=f"365-day Corr with {col}", color="darkred", linestyle="dashed", alpha=0.99)  
            plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            plt.title(f"Rolling Correlation (30-day & 365-day) between CO2 and {col}")
            plt.xlabel("Date")
            plt.ylabel("Correlation")
            plt.legend()
            plt.show()


def plot_lagged_correlations(df, target_column, max_lag_days, significance_threshold=0.05):
    """Plots lagged correlations between the target column and other columns in the DataFrame."""
    
    lagged_correlation_results = pd.DataFrame(index=range(1, max_lag_days + 1))
    for col in df.columns:
        if col != target_column:
            correlations = [
                df[target_column].corr(df[col].shift(lag)) for lag in range(1, max_lag_days + 1)
            ]
            lagged_correlation_results[col] = correlations

    plt.figure(figsize=(16, 6))
    for col in lagged_correlation_results.columns:
        plt.plot(lagged_correlation_results.index, lagged_correlation_results[col], label=col)

    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(y=significance_threshold, color="red", linestyle="dashed", label=f"Significance Threshold (±{significance_threshold})")
    plt.axhline(y=-significance_threshold, color="red", linestyle="dashed")
    plt.title(f"Lagged Correlation Analysis (Up to {max_lag_days} Days): {target_column} vs Predictors")
    plt.xlabel("Lag (Days)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()


def interpret_p_value(p_values):
    ''' Add an interpretation column based on p-values'''

    if all(p < 0.01 for p in p_values):  # Strong causality across all lags
        return "Strong causality (p < 0.01)"
    elif any(p < 0.05 for p in p_values):  # Moderate causality at some lags
        return "Moderate to low causality (p < 0.05 at some lags)"
    else:
        return "No significant causality"


def plot_feature_importances(features, importances_list, model_titles, feature_colors, layout=(2, 2), figsize=(12, 10)):
    """Plots feature importance for multiple models in a grid layout with consistent colors for features."""
    
    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)

    for i, ax in enumerate(axes.ravel()):
        y_pos = np.arange(len(features))
        colors = [feature_colors[feature] for feature in features]
        ax.barh(y_pos, importances_list[i], align="center", alpha=0.7, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Importance (%)")
        ax.set_title(model_titles[i])
        ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()




class GasDistributionVisualizer:
    def __init__(self, data, stats_df):
        self.data = data
        self.stats_df = stats_df
        self.colors = ['red', 'blue', 'green', 'magenta']
    
    def _create_figure(self):
        """Create the figure and axes for the subplots"""
        num_columns = len(self.data.columns)
        rows = (num_columns + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
        return fig, axes.flatten()
    
    def _plot_histogram_kde(self, ax, data, color):
        """Plot histogram and KDE for the given data"""
        data.plot(kind="hist", bins=30, alpha=0.6, density=True,
                 ax=ax, label="Histogram", color=color)
        data.plot(kind="kde", ax=ax, label="KDE")
    
    def _plot_stat_lines(self, ax, stats):
        """Plot vertical lines for mean, median, and mode"""
        ax.axvline(stats['mean'], color='black', linestyle='-', linewidth=1.5, label='Mean')
        ax.axvline(stats['median'], color='gray', linestyle='--', linewidth=1.5, label='Median')
        ax.axvline(stats['mode'], color='purple', linestyle=':', linewidth=1.5, label='Mode')
    
    def _add_stats_text(self, ax, stats):
        """Add statistics as text box to the plot"""
        stats_text = (
            f"Mean: {stats['mean']:.2f}\n"
            f"Median: {stats['median']:.2f}\n"
            f"Mode: {stats['mode']:.2f}\n"
            f"Std: {stats['std']:.2f}\n"
            f"Var: {stats['var']:.2f}\n"
            f"Range: {stats['range']:.2f}\n"
            f"IQR: {stats['IQR']:.2f}"
        )
        ax.text(0.025, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    def _cleanup_unused_axes(self, fig, axes, used_count):
        """Remove any unused subplots"""
        for j in range(used_count, len(axes)):
            fig.delaxes(axes[j])
    
    def visualize(self):
        """Main method to create the visualization"""
        fig, axes = self._create_figure()
        
        for i, col in enumerate(self.data.columns):
            ax = axes[i]
            data = self.data[col].dropna()
            color = self.colors[i % len(self.colors)]
            
            # Get stats for current column
            stats = self.stats_df[self.stats_df['column'] == col].iloc[0]
            
            # Create the plots
            self._plot_histogram_kde(ax, data, color)
            self._plot_stat_lines(ax, stats)
            self._add_stats_text(ax, stats)
            
            # Set titles and labels
            ax.set_title(f"{col} distribution")
            ax.set_xlabel(col)
            ax.legend()
        
        # Cleanup and show
        self._cleanup_unused_axes(fig, axes, i + 1)
        plt.tight_layout()
        plt.show()



class WeatherPairPlotter:
    def __init__(self, df, columns, title):
        """
        Initialize the WeatherPairPlotter with data, columns, and title.
        
        Parameters:
        - df: pandas.DataFrame containing the weather data
        - columns: list of column names to include in the pairplot
        - title: str title for the plot
        """
        self.df = df
        self.columns = columns
        self.title = title
        self.colors = None
        self.grid = None
        
    def select_columns(self):
        """Select specified columns from dataframe."""
        return self.df[self.columns]
    
    def setup_colors(self, palette_name="husl"):
        """Create color mapping for each column."""
        palette = sns.color_palette(palette_name, len(self.columns))
        self.colors = dict(zip(self.columns, palette))
    
    def create_pairgrid(self, corner=True, diag_sharey=False):
        """Initialize PairGrid for pairwise relationships."""
        self.grid = sns.PairGrid(self.select_columns(), corner=corner, diag_sharey=diag_sharey)
    
    @staticmethod
    def hist_kde_plot(x, **kwargs):
        """Combined histogram and KDE plot for diagonal."""
        sns.histplot(x, kde=True, stat="density", alpha=0.8, **kwargs)
        sns.kdeplot(x, fill=True, alpha=0.3, **kwargs)
    
    def style_diagonal_plots(self, stats_df):
        """Style diagonal plots with custom colors and statistics."""
        for ax, var in zip(self.grid.diag_axes, self.columns):
            color = self.colors[var]
            
            # Recolor bars and lines
            for patch in ax.patches:
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            lines = ax.get_lines()
            if lines:
                lines[0].set_color(color)
                lines[0].set_alpha(1)
            
            for coll in ax.collections:
                coll.set_facecolor(color)
                coll.set_alpha(0.3)
            
            # Add statistical lines
            col_stats = stats_df[stats_df["column"] == var].iloc[0]
            ax.axvline(col_stats["mean"], color="black", linestyle="-", linewidth=1.5, label="Mean")
            ax.axvline(col_stats["median"], color="gray", linestyle="--", linewidth=1.5, label="Median")
            ax.axvline(col_stats["mode"], color="purple", linestyle=":", linewidth=1.5, label="Mode")
            
            ax.legend(loc="upper left", fontsize="x-small")
    
    def plot_off_diagonal(self):
        """Plot regression plots for off-diagonal elements."""
        df_selected = self.select_columns()
        for i, y_var in enumerate(self.columns[1:], start=1):
            for j, x_var in enumerate(self.columns[:i]):
                ax = self.grid.axes[i, j]
                sns.regplot(
                    x=x_var, y=y_var, data=df_selected,
                    scatter_kws={"s": 20, "alpha": 0.7, "color": self.colors[y_var]},
                    line_kws={"color": "red", "lw": 2},
                    ax=ax,
                    truncate=False
                )
    
    @staticmethod
    def summary_stats(df):
        """Calculate summary statistics for each column."""
        stats = []
        for col in df.columns:
            stats.append({
                "column": col,
                "mean": df[col].mean(),
                "median": df[col].median(),
                "mode": df[col].mode()[0] 
            })
        return pd.DataFrame(stats)
    
    def create_pairplot(self):
        """Main method to create the complete pairplot."""
        self.setup_colors()
        self.create_pairgrid()
        
        self.grid.map_diag(self.hist_kde_plot)
        
        stats_df = self.summary_stats(self.select_columns())
        
        self.style_diagonal_plots(stats_df)
        
        self.plot_off_diagonal()
        
        self.grid.figure.suptitle(self.title, y=1.02, fontsize=16, fontweight="bold")
        self.grid.tight_layout()
        
        return self.grid
    
    def show(self):
        """Display the plot."""
        plt.show()