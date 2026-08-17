#!/bin/bash

# Generate a secret key
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Update .env file
if grep -q "AIRFLOW__CORE__SECRET_KEY=" .env; then
    # Replace existing secret key
    sed -i '' "s/AIRFLOW__CORE__SECRET_KEY=.*/AIRFLOW__CORE__SECRET_KEY=${SECRET_KEY}/" .env
else
    # Append new secret key
    echo "AIRFLOW__CORE__SECRET_KEY=${SECRET_KEY}" >> .env
fi

echo "SECRET_KEY set successfully in .env file."
echo "KEY: ${SECRET_KEY}"