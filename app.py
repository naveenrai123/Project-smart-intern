from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)

# Load the trained XGBoost model
with open('model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Collect input from form
        quarter = request.form['quarter']
        department = request.form['department']
        day = request.form['day']
        team = request.form['team']
        targeted_productivity = request.form['targeted_productivity']
        smv = request.form['smv']
        over_time = request.form['over_time']
        incentive = request.form['incentive']
        idle_time = request.form['idle_time']
        idle_men = request.form['idle_men']
        no_of_style_change = request.form['no_of_style_change']
        no_of_workers = request.form['no_of_workers']
        month = request.form['month']

        # Prepare input
        total = [[
            int(quarter), int(department), int(day), int(team),
            float(targeted_productivity), float(smv), int(over_time), int(incentive),
            float(idle_time), int(idle_men), int(no_of_style_change), float(no_of_workers), int(month)
        ]]

        # Prediction
        prediction = model.predict(total)[0]
        productivity_score = round(float(prediction), 2)

        if prediction <= 0.3:
            text = 'The employee is Averagely Productive.'
        elif 0.3 < prediction <= 0.8:
            text = 'The employee is Medium Productive.'
        else:
            text = 'The employee is Highly Productive.'

        # Visualization
        graphs = []

        def generate_graph(fig_func):
            buf = io.BytesIO()
            fig_func()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()
            return graph

        # Bar Chart
        def bar_chart():
            categories = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
            values = [targeted_productivity, smv, over_time, idle_time]
            plt.bar(categories, [float(v) for v in values], color=['blue', 'green', 'red', 'orange'])
            plt.title('Employee Productivity Parameters')

        # Scatter Plot
        def scatter_plot():
            plt.scatter([1, 2, 3, 4], [float(targeted_productivity), float(smv), float(over_time), float(idle_time)])
            plt.title('Scatter Plot of Employee Parameters')

        # Line Plot
        def line_plot():
            plt.plot([1, 2, 3, 4], [float(targeted_productivity), float(smv), float(over_time), float(idle_time)], marker='o')
            plt.title('Line Plot of Employee Parameters')

        # Pie Chart
        def pie_chart():
            sizes = [float(targeted_productivity), float(smv), float(over_time), float(idle_time)]
            labels = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.title('Pie Chart of Employee Parameters')

        # Histogram
        def histogram():
            data = np.random.randn(100)
            plt.hist(data, bins=20, color='purple')
            plt.title('Histogram of Random Data')

        # Boxplot
        def boxplot():
            data = np.random.rand(100, 5)
            plt.boxplot(data)
            plt.title('Boxplot of Random Data')

        # Append all graphs
        graphs.append(generate_graph(bar_chart))
        graphs.append(generate_graph(scatter_plot))
        graphs.append(generate_graph(line_plot))
        graphs.append(generate_graph(pie_chart))
        graphs.append(generate_graph(histogram))
        graphs.append(generate_graph(boxplot))

        return render_template('submit.html', prediction_text=text, prediction_score=productivity_score, graphs=graphs)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
