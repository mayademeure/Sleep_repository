import argparse
from src.data_preprocessing import DatasetManager
from src.visualization import Visualizer
from src.analysis import Analysis

# Set up argument parser for command-line options
parser = argparse.ArgumentParser(description="Run specific parts of the program.")
parser.add_argument("--run", type=str, choices=["all", "analysis", "visualization"], default="all",
                    help="Specify which part to run: 'all', 'analysis', or 'visualization'")
parser.add_argument("--viz", type=str, choices=["all", "heatmap", "histograms", "bar_charts", "scatter",  "BMI_and_age", "bp_boxplots", "BMI_and_sleep"], default="all",
                    help="Specify which visualization to run: 'all', 'heatmap', 'histograms', 'bar_charts', 'scatter', 'BMI_and_age', 'bp_boxplots', 'BMI_and_sleep'")
parser.add_argument("--analysis", type=str, choices=["all", "correlation", "scatter_regression", "regression"], default="all",
                    help="Specify which analysis to run: 'all', 'correlation', 'scatter_regression', 'regression'")
args = parser.parse_args()

if __name__ == "__main__":
    # Step 1: Preprocess the dataset
    dataset_manager = DatasetManager(file_path="/Users/mayademeure/Desktop/sleep/data/Sleep_health_and_lifestyle_dataset.csv")
    dataset_manager.preprocess(categorical_columns=["Gender", "Occupation", "BMI Category", "Sleep Disorder"])
    data = dataset_manager.data  # Load preprocessed data into memory

    # Check the --run argument to determine which part to execute
    if args.run in ["all", "visualization"]:
        # Step 2: Create a visualizer instance and generate visualizations
        visualizer = Visualizer(data)

        # Generate a heatmap for correlations if specified
        if args.viz in ["all", "heatmap"]:
            visualizer.plot_correlation_heatmap()

        # Generate histograms for numeric columns if specified
        if args.viz in ["all", "histograms"]:
            visualizer.plot_histograms(["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps"])

        # Generate bar charts for categorical columns if specified
        if args.viz in ["all", "bar_charts"]:
            visualizer.plot_bar_charts(["Gender", "Occupation", "BMI Category", "Sleep Disorder"])

        # Generate scatter plots for specific pairs of variables if specified
        if args.viz in ["all", "scatter"]:
            visualizer.plot_scatter("Sleep Duration", "Quality of Sleep", hue="Gender")
            visualizer.plot_scatter("Stress Level", "Heart Rate")
            visualizer.plot_scatter("Stress Level", "Quality of Sleep")
            visualizer.plot_scatter("Sleep Duration", "Stress Level")
            visualizer.plot_scatter("Quality of Sleep", "Heart Rate")

        # Generate boxplots for blood pressure grouped by BMI category if specified
        if args.viz in ["all", "bp_boxplots"]:
            visualizer.plot_bp_boxplots(bp_cols=["Systolic BP", "Diastolic BP"], group_col="BMI Category")

        # Visualize relationships between BMI category and age if specified
        if args.viz in ["all", "BMI_and_age"]:
            visualizer.plot_bmi_age_relationship(age_col="Age", bmi_col="BMI Category")

        # Visualize the relationship between BMI and sleep disorders if specified
        if args.viz in ["all", "BMI_and_sleep"]:
            bmi_mapping = {0: "Normal", 1: "Overweight", 2: "Obese"}
            sleep_disorder_mapping = {0: "No Disorder", 1: "Insomnia", 2: "Apnea"}

            # Plot Sleep Disorder vs. BMI relationship
            visualizer.plot_sleep_disorder_bmi(
                bmi_col="BMI Category",
                sleep_disorder_col="Sleep Disorder",
                bmi_mapping=bmi_mapping,
                sleep_disorder_mapping=sleep_disorder_mapping
            )

    if args.run in ["all", "analysis"]:
        # Step 3: Perform data analysis
        analysis = Analysis(data)

        # Calculate correlations between specific variables if specified
        if args.analysis in ["all", "correlation"]:
            analysis.calculate_correlation("Sleep Duration", "Quality of Sleep")
            analysis.calculate_correlation("Stress Level", "Heart Rate")
            analysis.calculate_correlation("Stress Level", "Quality of Sleep")
            analysis.calculate_correlation("Sleep Duration", "Stress Level")
            analysis.calculate_correlation("Quality of Sleep", "Heart Rate")
            
        # Generate scatter plots with regression lines if specified
        if args.analysis in ["all", "scatter_regression"]:
            analysis.scatter_plot_with_regression("Sleep Duration", "Quality of Sleep")
            analysis.scatter_plot_with_regression("Stress Level", "Heart Rate")
            analysis.scatter_plot_with_regression("Stress Level", "Quality of Sleep")
            analysis.scatter_plot_with_regression("Sleep Duration", "Stress Level")
            analysis.scatter_plot_with_regression("Quality of Sleep", "Heart Rate")

        # Perform regression analysis between specific variable pairs if specified
        if args.analysis in ["all", "regression"]:
            analysis.perform_regression("Sleep Duration", "Quality of Sleep")
            analysis.perform_regression("Stress Level", "Heart Rate")
            analysis.perform_regression("Stress Level", "Quality of Sleep")
            analysis.perform_regression("Sleep Duration", "Stress Level")
            analysis.perform_regression("Quality of Sleep", "Heart Rate")
