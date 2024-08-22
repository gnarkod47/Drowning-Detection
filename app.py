from flask import Flask, request, jsonify, redirect, url_for, render_template
import mysql.connector
import os
import subprocess

app = Flask(__name__, template_folder='templates', static_folder='static')

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="appu1234",
    database="userDB"
)

cursor = db.cursor()

@app.route('/')
def home():
    # Render the home page  
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print("Inside the post method within the login()")
        username = request.form.get('username')
        password = request.form.get('password')

        # Query the database to check if the username and password match
        query = "SELECT * FROM user WHERE name = %s AND password = %s"
        cursor.execute(query, (username, password))
        user = cursor.fetchone()

        if user:
            print("Inside user if condition")
            # If credentials are valid, redirect to functional.html
            # return redirect(url_for('functional'))
            return {'success':True}
        else:
            # If credentials are invalid, return error response
            return jsonify({'success': False, 'message': 'Invalid username or password'})

    # Render the login page
    return render_template('login.html')

@app.route('/functional')
def functional():
    # Render the functional page
    return render_template('functional.html')

@app.route('/rg')
def rg():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    # Get form data from the registration form
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')

    # Check if the username is already taken
    query = "SELECT * FROM user WHERE name = %s"
    cursor.execute(query, (username,))
    existing_user = cursor.fetchone()
    if existing_user:
        return jsonify({'success': False, 'message': 'Username already exists'})

    # Insert the new user into the database
    insert_query = "INSERT INTO user (name, password, email) VALUES (%s, %s, %s)"
    cursor.execute(insert_query, (username, password, email))
    db.commit()  # Commit the transaction

    return jsonify({'success': True, 'message': 'User registered successfully'})


@app.route('/detect', methods=['POST'])
def detect():
    # print("Inside detect function")
    # Get the uploaded video file
    video_file = request.files['video']

    # Save the video file to the videos folder
    video_path = os.path.join('videos', video_file.filename)
    video_file.save(video_path)

    # Call detect.py script with the video file as argument
    # print("It should call the detect.py")
    result = subprocess.run(['python', 'detect.py', '--source', video_path], capture_output=True)
    # print("The subprocess command should have run by now")

    # Return the output of detect.py
    return result.stdout.decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
