import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
print("Project root contents:", os.listdir("."))
print("Inference directory contents:", os.listdir("inference"))

from app.main import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
