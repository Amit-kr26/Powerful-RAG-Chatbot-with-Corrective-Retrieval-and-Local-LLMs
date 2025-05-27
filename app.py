from flask import Flask, request, render_template_string, jsonify
import time
import lang # Make sure to import your refactored lang.py module

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    output = None
    start_time = None
    end_time = None
    if request.method == 'POST':
        question = request.form['question']
        start_time = time.time()  # Start the timer
        
        # Run the CRAG framework from lang.py
        output = lang.run_crag_workflow(question)  
        end_time = time.time()  # End the timer

    # Calculate the time taken
    time_taken = None
    if start_time and end_time:
        time_taken = end_time - start_time

    # Advanced HTML template with enhanced design and animations
    html_template = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>RAG Chatbot with CRAG</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
          body {
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            font-family: 'Roboto', sans-serif;
            color: #333;
          }
          .container {
            max-width: 700px;
            margin-top: 60px;
            background-color: #ffffff;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
          }
          h1 {
            color: #333;
            text-align: center;
            font-weight: bold;
            margin-bottom: 25px;
          }
          .lead {
            text-align: center;
            font-size: 1.15rem;
            margin-bottom: 30px;
            color: #555;
          }
          .form-control {
            border-radius: 50px;
            padding: 20px;
            font-size: 1rem;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
            transition: box-shadow 0.3s ease;
          }
          .form-control:focus {
            box-shadow: 0px 4px 15px rgba(0, 123, 255, 0.5);
          }
          .btn-primary {
            background-color: #007bff;
            border: none;
            border-radius: 50px;
            padding: 12px 25px;
            font-size: 1.1rem;
            transition: background-color 0.3s ease, transform 0.3s ease;
          }
          .btn-primary:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
          }
          .output {
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
            font-family: 'Courier New', monospace;
            font-size: 1.2rem;
            color: #495057;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
          }
          .progress {
            height: 30px;
            margin-top: 25px;
            border-radius: 50px;
            overflow: hidden;
            background-color: #e9ecef;
            box-shadow: inset 0px 2px 5px rgba(0, 0, 0, 0.1);
          }
          .progress-bar {
            background-color: #28a745;
            height: 100%;
            transition: width 0.4s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.1rem;
          }
        </style>
        <script>
          function startProgressBar() {
            var elem = document.getElementById("progress-bar");
            var width = 0;
            var id = setInterval(frame, 20); // Update every 20ms
            function frame() {
              if (width >= 100) {
                clearInterval(id);
              } else {
                width++;
                elem.style.width = width + '%';
                elem.innerHTML = width + '%';
              }
            }
          }
        </script>
      </head>
      <body>
        <div class="container">
          <h1>RAG Chatbot with CRAG</h1>
          <p class="lead">Enhance your answers with our advanced CRAG framework by entering your question below.</p>
          <form method="POST" onsubmit="startProgressBar()">
            <div class="form-group">
              <label for="question">Your Question:</label>
              <input type="text" class="form-control" id="question" name="question" placeholder="e.g., What is CRAG?" required>
            </div>
            <button type="submit" class="btn btn-primary btn-block">Generate Answer</button>
          </form>
          
          <!-- Progress Bar -->
          <div class="progress">
            <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%">0%</div>
          </div>

          {% if output %}
          <h2 class="mt-4">Generated Output:</h2>
          <div class="output">
            {{ output }}
          </div>
          <p class="mt-3"><strong>Time Taken:</strong> {{ time_taken }} seconds</p>
          {% endif %}
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.2/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
      </body>
    </html>
    '''
    return render_template_string(html_template, output=output, time_taken=time_taken)

if __name__ == '__main__':
    app.run(port=11434)
