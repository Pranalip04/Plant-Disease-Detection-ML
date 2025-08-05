from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables
predicted_class = ""
predicted_class_idx = -1
class_labels = [
    "Leafsmut",
    "Brownspot",
    "Bacterialblight",
    "Rust",
    "Powdery",
    "Healthy",
    "Normal"
]

# Example dataset (replace with actual dataset)
true_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
predictions = np.array([0, 1, 2, 1, 0, 1, 2, 2, 0])
X = np.random.rand(100, 224, 224, 3)  # Example features
y = np.random.randint(0, len(class_labels), 100)  # Example labels

# Load the machine learning model globally
loaded_model = tf.keras.models.load_model('keras_model.h5')
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to train and get accuracy of traditional ML models
def get_ml_model_accuracies():
    X_flattened = X.reshape(len(X), -1)  # Flatten the image data

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_predictions = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    return knn_accuracy, dt_accuracy, rf_accuracy

# Function to display selected image and its predicted class
def display_image(file_path):
    global predicted_class, predicted_class_idx, loaded_model
    
    # Open and resize image
    image = Image.open(file_path)
    image = image.resize((300, 300), Image.LANCZOS)
    
    # Convert image for tkinter
    tk_image = ImageTk.PhotoImage(image)
    
    # Display image
    image_label.config(image=tk_image)
    image_label.image = tk_image
    
    # Display image name
    image_name_label.config(text="Image Name: " + os.path.basename(file_path))
    
    # Preprocess the image for prediction
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict using the loaded model
    predictions = loaded_model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    
    # Check if the predicted class index is within the range of class_labels
    if predicted_class_idx >= len(class_labels):
        messagebox.showerror("Prediction Error", "Predicted class index is out of range.")
        return
    
    # Map predicted class index to class label
    predicted_class = class_labels[predicted_class_idx]
    prediction_label.config(text="Predicted Class: " + predicted_class)
    
    # Enable buttons
    show_causes_btn.config(state=NORMAL)
    show_confusion_matrix_btn.config(state=NORMAL)
    show_accuracy_btn.config(state=NORMAL)
    show_graph_btn.config(state=NORMAL)
    show_reasons_btn.config(state=NORMAL)
    
    # Modify button text and actions based on the predicted class
    if predicted_class in ["Leafsmut", "Brownspot", "Bacterialblight", "Rust", "Powdery"]:
        show_causes_btn.config(text="Show Treatment")
    else:
        show_causes_btn.config(text="Show Causes")

# Function to show causes based on the predicted class
def show_causes():
    causes_text = ""
    if predicted_class == "Leafsmut":
        causes_text = ("Caused by fungal infection, spreads through infected seeds or soil, favored by high humidity and poor crop rotation.")
    elif predicted_class == "Brownspot":
        causes_text = ("Fungal disease prevalent in warm, humid conditions, spreads through wind and rain, affecting leaves and reducing photosynthesis.")
    elif predicted_class == "Bacterialblight":
        causes_text = ("Bacteria infects through wounds or natural openings, thrives in moist conditions, causing leaf lesions and reduced plant vigor.")
    elif predicted_class == "Rust":
        causes_text = ("Fungal disease spread by wind, favored by high humidity and moderate temperatures, causing orange-red pustules on leaves and stems.")
    elif predicted_class == "Powdery":
        causes_text = ("Fungal infection thriving in warm, dry conditions, covering leaves with white powdery growth, impairing photosynthesis and weakening plants.")
    elif predicted_class == "Healthy":
        causes_text = ("Plants unaffected by diseases, exhibiting normal growth and development without visible symptoms of fungal or bacterial infections.")
    elif predicted_class == "Normal":
        causes_text = ("You have no eye disease, your eye is normal.")
    
    messagebox.showinfo("Causes", causes_text)

# Function to show treatment or reasons based on the predicted class
def show_reasons():
    reasons_text = ""
    if predicted_class == "Leafsmut":
        reasons_text = ("Use certified disease-free seeds, practice crop rotation, and apply fungicides before symptoms appear to prevent spread.")
    elif predicted_class == "Brownspot":
        reasons_text = ("Remove and destroy infected leaves, improve air circulation, use fungicides preventatively, and avoid overhead watering to reduce humidity.")
    elif predicted_class == "Bacterialblight":
        reasons_text = ("Prune affected areas, use copper-based fungicides, practice crop rotation, and promote plant health with balanced nutrition and proper spacing.")
    elif predicted_class == "Rust":
        reasons_text = ("Remove and destroy infected leaves, improve air circulation, apply fungicides preventatively, and avoid overhead watering to reduce leaf wetness.")
    elif predicted_class == "Powdery":
        reasons_text = ("Remove and destroy infected plant parts, improve air circulation, apply fungicides early in the season, and avoid overhead watering to reduce humidity.")
    elif predicted_class == "Healthy":
        reasons_text = ("Maintain good plant hygiene, monitor for early signs of diseases, practice proper irrigation and nutrition, and promptly remove and destroy any infected plants or plant parts to prevent spread.")
    elif predicted_class == "Normal":
        reasons_text = ("You have no eye disease, your eye is normal.")
    
    messagebox.showinfo("Treatment" if show_causes_btn.cget("text") == "Show Treatment" else "Reasons", reasons_text)

# Function to show confusion matrix based on the predicted class
def show_confusion_matrix():
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix with seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_labels, yticklabels=class_labels)

    # Set labels and title
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    # Show plot
    plt.show()

# Function to calculate and display accuracy score
def show_accuracy():
    knn_acc, dt_acc, rf_acc = get_ml_model_accuracies()
    acc_score = accuracy_score(true_labels, predictions)
    messagebox.showinfo("Accuracy Score", f"Model Accuracy: {acc_score:.2f}\nKNN Accuracy: {knn_acc:.2f}\nDecision Tree Accuracy: {dt_acc:.2f}\nRandom Forest Accuracy: {rf_acc:.2f}")

# Function to show a graph (e.g., bar chart) in a new window
def show_graph():
    knn_acc, dt_acc, rf_acc = get_ml_model_accuracies()
    
    # Bar chart for accuracies
    plt.figure(figsize=(8, 6))
    classifiers = ['TensorFlow Model', 'KNN', 'Decision Tree', 'Random Forest']
    accuracies = [accuracy_score(true_labels, predictions), knn_acc, dt_acc, rf_acc]
    plt.bar(classifiers, accuracies, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Classifiers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to open file dialog and select an image
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        display_image(file_path)

def handle_login():
    username = username_entry.get()  # Get username from entry widget
    password = password_entry.get()  # Get password from entry widget
    
    # Perform login logic here (e.g., validate credentials)
    # For simplicity, let's assume a basic check
    if username == "admin" and password == "password":
        login_window.destroy()  # Close login window
        open_main_window()  # Open the main application window
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

def open_main_window():
    global root, image_label, image_name_label, prediction_label, show_causes_btn, show_confusion_matrix_btn, show_accuracy_btn, show_graph_btn, show_reasons_btn
    
    root = Tk()
    root.title("Image Classification and Information")
    root.geometry("2000x1000")
    
    # Add background image to the main window
    background_image_path = r"C:\Users\pawar\OneDrive\Desktop\Plant Disease Project\407898.jpg"
    background_image = Image.open(background_image_path)
    background_image = background_image.resize((1500, 1500), Image.LANCZOS)
    background_photo = ImageTk.PhotoImage(background_image)
    
    background_label = Label(root, image=background_photo)
    background_label.image = background_photo
    background_label.place(relwidth=1, relheight=1)
    
    # Title label
    title_label = Label(root, text="Plant Disease Detection", font=("Helvetica" , 40, "bold"))
    title_label.pack(pady=10)
    
    # Image display label
    image_label = Label(root)
    image_label.pack(pady=10)
    
    # Image name label
    image_name_label = Label(root, text="Image Name: None", font=("Helvetica", 12))
    image_name_label.pack()
    
    # Prediction label
    prediction_label = Label(root, text="Predicted Class: None", font=("Helvetica", 12))
    prediction_label.pack()
    
    # Open image button
    open_image_btn = Button(root, text="Open Image", command=open_image, font=("Helvetica", 14), width=20)
    open_image_btn.pack(pady=10)

    # Show causes button
    show_causes_btn = Button(root, text="Show Causes", command=show_causes, state=DISABLED, font=("Helvetica", 14), width=20)
    show_causes_btn.pack(pady=5)

    # Show reasons button
    show_reasons_btn = Button(root, text="Show Reasons", command=show_reasons, state=DISABLED, font=("Helvetica", 14), width=20)
    show_reasons_btn.pack(pady=5)
    
    # Show confusion matrix button
    show_confusion_matrix_btn = Button(root, text="Show Confusion Matrix", command=show_confusion_matrix, state=DISABLED, font=("Helvetica", 14), width=20)
    show_confusion_matrix_btn.pack(pady=5)

    # Show accuracy button
    show_accuracy_btn = Button(root, text="Show Accuracy", command=show_accuracy, state=DISABLED, font=("Helvetica", 14), width=20)
    show_accuracy_btn.pack(pady=5)

    # Show graph button
    show_graph_btn = Button(root, text="Show Graph", command=show_graph, state=DISABLED, font=("Helvetica", 14), width=20)
    show_graph_btn.pack(pady=5)

    root.mainloop()

# Create login window
login_window = Tk()
login_window.title("Login")
login_window.geometry("2000x1000")

background_login_image_path = r"C:\Users\pawar\OneDrive\Desktop\Plant Disease Project\desktop-wallpaper-natural-leaves-nature-leaves.jpg"
background_image_login = Image.open(background_login_image_path)
background_image_login = background_image_login.resize((1500, 1500), Image.LANCZOS)
background_photo_login = ImageTk.PhotoImage(background_image_login)
    
background_label = Label(login_window, image=background_photo_login)
background_label.image = background_photo_login
background_label.place(relwidth=1, relheight=1)


title_label = Label(login_window, text="Welcome to the login page!", font=("Helvetica" , 40, "bold"))
title_label.pack(pady=10)
# Username label and entry
username_label = Label(login_window, text="Username:", font=("Helvetica", 20))
username_label.place(x=500,y=150)
username_entry = Entry(login_window, font=("Helvetica", 20), width=20)
username_entry.place(x=700,y=150)

# Password label and entry
password_label = Label(login_window, text="Password:", font=("Helvetica", 20))
password_label.place(x=500,y=220)
password_entry = Entry(login_window, font=("Helvetica", 20), width=20, show='*')
password_entry.place(x=700,y=220)

# Login button
login_button = Button(login_window, text="Login", command=handle_login, font=("Helvetica", 25), width=10)
login_button.place(x=650,y=300)

login_window.mainloop()

