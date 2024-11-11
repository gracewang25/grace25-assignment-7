from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "new_random_secret_key"  # Replace with your own secret key for session management

# Helper function to generate data and save plots
def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate initial random dataset X and Y
    X = np.random.rand(N)
    Y = beta0 + beta1 * X + np.random.normal(mu, np.sqrt(sigma2), N)

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, label="Data points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Scatter Plot and Regression Line\ny = {slope:.2f}x + {intercept:.2f}")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + np.random.normal(mu, np.sqrt(sigma2), N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Calculate proportions of simulated values more extreme than observed for p-value approximation
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_more_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    # Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    return X, Y, slope, intercept, plot1_path, plot2_path, slopes, intercepts, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract parameters from form and generate data
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and store results in session
        X, Y, slope, intercept, plot1, plot2, slopes, intercepts, slope_extreme, intercept_extreme = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store in session
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["plot1"] = plot1
        session["plot2"] = plot2
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme

        # Pass results to render_template
        return render_template(
            "results.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme
        )

    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    return index()  # Calls the index function to handle form submission and render results

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve stored session variables
    slope = session.get("slope", 0.0)
    intercept = session.get("intercept", 0.0)
    slopes = session.get("slopes", [])
    intercepts = session.get("intercepts", [])
    beta0 = session.get("beta0", 0.0)
    beta1 = session.get("beta1", 0.0)

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the stored simulations for hypothesis testing
    if parameter == "slope":
        simulated_stats = np.array(slopes)  # Distribution of simulated slopes
        observed_stat = slope
        hypothesized_value = beta1  # Null hypothesis slope
    else:
        simulated_stats = np.array(intercepts)  # Distribution of simulated intercepts
        observed_stat = intercept
        hypothesized_value = beta0  # Null hypothesis intercept

    # Calculate p-value based on the type of test
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == "!=":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
    else:
        p_value = None

    # Fun message for rare events
    fun_message = None
    if p_value is not None and p_value <= 0.0001:
        fun_message = "Wow! That's a rare event!"

    plot3_path = "static/plot3.png"
    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=20, alpha=0.5, label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="--", linewidth=2, label="Observed Statistic")
    plt.axvline(hypothesized_value, color="green", linestyle="--", linewidth=2, label="Hypothesized Value")
    plt.xlabel("Statistic Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Simulated {parameter.capitalize()}s with p-value = {p_value:.4f}")
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "results.html",
        plot1=session.get("plot1"),
        plot2=session.get("plot2"),
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        p_value=p_value,
        fun_message=fun_message,
        slope_extreme=session.get("slope_extreme"),
        intercept_extreme=session.get("intercept_extreme"),
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = np.array(session.get("slopes"))
    intercepts = np.array(session.get("intercepts"))
    slope_extreme = session.get("slope_extreme")
    intercept_extreme = session.get("intercept_extreme")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = slopes
        observed_stat = slope
        true_param = beta1
    else:
        estimates = intercepts
        observed_stat = intercept
        true_param = beta0

    # Calculate mean and standard error of the estimates from simulations
    mean_estimate = np.mean(estimates)
    std_error = np.std(estimates, ddof=1) / np.sqrt(S)

    # Calculate the user-specified confidence interval
    t_score = t.ppf((1 + confidence_level / 100) / 2, df=S - 1)
    ci_lower = mean_estimate - t_score * std_error
    ci_upper = mean_estimate + t_score * std_error

    # Calculate the 50% confidence interval
    t_score_50 = t.ppf((1 + 0.5) / 2, df=S - 1)
    ci_lower_50 = mean_estimate - t_score_50 * std_error
    ci_upper_50 = mean_estimate + t_score_50 * std_error

    # Check if the specified confidence interval includes the true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # Define colors for the confidence interval line based on whether it includes the true parameter
    ci_color = "green" if includes_true else "red"

    # Generate the confidence interval plot
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(12, 4))
    plt.scatter(estimates, [0] * S, color="gray", alpha=0.5, label="Simulated Estimates")
    plt.plot([ci_lower, ci_upper], [0, 0], color=ci_color, linewidth=4, label=f"{confidence_level}% Confidence Interval")
    plt.plot([ci_lower_50, ci_upper_50], [0, 0], color="orange", linewidth=4, linestyle=":", label="50% Confidence Interval")
    plt.scatter([mean_estimate], [0], color="blue", s=100, zorder=5, label="Mean Estimate")
    plt.axvline(true_param, color="green", linestyle="--", linewidth=2, label="True Parameter")

    # Set x-axis ticks with increments of 0.5 and adjust x-axis limits to cover all data points
    min_x = min(np.min(estimates), ci_lower, ci_lower_50, true_param) - 0.5
    max_x = max(np.max(estimates), ci_upper, ci_upper_50, true_param) + 0.5
    plt.xticks(np.arange(round(min_x * 2) / 2, round(max_x * 2) / 2 + 0.5, 0.5))
    plt.xlim(min_x, max_x)

    # Formatting the plot to have only the x-axis
    plt.xlabel(f"{parameter.capitalize()} Estimate")
    plt.yticks([])
    plt.title(f"{confidence_level}% and 50% Confidence Intervals for {parameter.capitalize()} (Mean Estimate)")
    plt.legend(loc="upper right")
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template, including slope_extreme and intercept_extreme
    return render_template(
        "results.html",
        plot1=session.get("plot1"),
        plot2=session.get("plot2"),
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_lower_50=ci_lower_50,
        ci_upper_50=ci_upper_50,
        includes_true=includes_true,
        observed_stat=observed_stat,
        slope_extreme=slope_extreme,
        intercept_extreme=intercept_extreme,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

if __name__ == "__main__":
    app.run(debug=True)
