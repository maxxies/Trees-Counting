FROM python:3.8

# Set the working directory 
WORKDIR /app

# Copy the current directory into /app
COPY . /app

# Install needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available
EXPOSE 80

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]