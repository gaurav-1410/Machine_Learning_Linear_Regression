import os

# Define the directory structure
project_name = "mlr_project"
folders = [
    os.path.join(project_name, "data")  # Data folder
]

# Define files to be created
files = {
    os.path.join(project_name, "data", "generate_data.py"): "# Script to generate synthetic dataset\n",
    os.path.join(project_name, "data", "dataset.csv"): "",  # Empty file; will be generated later
    os.path.join(project_name, "app.py"): "# Streamlit app for Multiple Linear Regression\n",
    os.path.join(project_name, "requirements.txt"): "streamlit\npandas\nnumpy\nscikit-learn\nmatplotlib\n",
    os.path.join(project_name, "README.md"): "# Multiple Linear Regression Streamlit App\n"
}

# Create directories
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file_path, content in files.items():
    with open(file_path, "w") as f:
        f.write(content)

print(f"✅ Project structure created successfully in '{project_name}'")
