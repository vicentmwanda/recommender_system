# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy everything from repo into container
COPY . .

# Upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install your recommender package from GitHub
RUN pip install git+https://github.com/vicentmwanda/recommender_system.git

# Expose port for Streamlit
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "demo/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
