#!/bin/bash
# This script keeps the Render app awake by pinging it every 10 minutes
# Schedule this with cron-job.org or UptimeRobot

APP_URL="${1:-https://predictive-maintenance.onrender.com}"

echo "Pinging $APP_URL to keep it awake..."
curl -s "$APP_URL/docs" > /dev/null

if [ $? -eq 0 ]; then
    echo "✓ Ping successful at $(date)"
else
    echo "✗ Ping failed at $(date)"
fi
