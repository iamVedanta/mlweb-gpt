from flask import Flask, render_template, request,redirect,url_for
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Store the data in memory
data_points = {'x': [], 'y': []}
regression_line = {'slope': None, 'intercept': None}

def update_regression_line():
    if len(data_points['x']) >= 2:
        X = np.array(data_points['x']).reshape(-1, 1)
        y = np.array(data_points['y'])
        model = LinearRegression().fit(X, y)
        regression_line['slope'] = model.coef_[0]
        regression_line['intercept'] = model.intercept_

# ... (previous code)

def plot_regression_line():
    if len(data_points['x']) >= 2:
        update_regression_line()
        X = np.array(data_points['x']).reshape(-1, 1)
        y = np.array(data_points['y'])
        model = LinearRegression().fit(X, y)

        plt.switch_backend('Agg')  # Use Agg backend to avoid main thread issues

        plt.scatter(data_points['x'], data_points['y'], color='blue')
        plt.plot(data_points['x'], model.predict(X), color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')

        # Save the plot as a base64-encoded image
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()

        return img_str

# ... (rest of the code)


@app.route('/')
def index():
    return render_template('index.html', data_points=data_points, regression_line=regression_line)

@app.route('/add_data', methods=['POST'])
def add_data():
    x_value = float(request.form['x'])
    y_value = float(request.form['y'])
    data_points['x'].append(x_value)
    data_points['y'].append(y_value)
    return render_template('index.html', data_points=data_points, regression_line=regression_line)

@app.route('/plot')
def plot():
    img_str = plot_regression_line()
    return render_template('plot.html', img_str=img_str)

@app.route('/delete_data', methods=['POST'])
def delete_data():
    index_to_delete = int(request.form['index'])
    data_points['x'].pop(index_to_delete)
    data_points['y'].pop(index_to_delete)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
