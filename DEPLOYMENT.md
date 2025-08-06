# StockInsightAI Deployment Guide

## Prerequisites

1. Python 3.8 or higher
2. pip (Python package installer)
3. Git (for deployment)
4. A server with SSH access (for manual deployment)

## Deployment Options

### Option 1: Deploy to Render (Recommended)

1. **Create a Render account**
   - Go to [render.com](https://render.com/) and sign up
   - Verify your email address

2. **Create a new Web Service**
   - Click "New" and select "Web Service"
   - Connect your GitHub/GitLab repository or deploy manually

3. **Configure the Web Service**
   - **Name**: stockinsightai (or your preferred name)
   - **Region**: Select the closest to your users
   - **Branch**: main (or your deployment branch)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**:
     - `DATABASE_URL`: Your database connection string
     - `PYTHON_VERSION`: 3.9 (or your preferred version)

4. **Deploy**
   - Click "Create Web Service"
   - Wait for the build to complete

### Option 2: Deploy to Heroku

1. **Install Heroku CLI**
   ```bash
   # On Windows
   winget install -e --id Heroku.HerokuCLI
   
   # On macOS
   brew tap heroku/brew && brew install heroku
   
   # On Linux
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create a new Heroku app**
   ```bash
   heroku create stockinsightai
   ```

4. **Set up the database**
   ```bash
   heroku addons:create heroku-postgresql:hobby-dev
   ```

5. **Deploy the application**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push heroku main
   ```

### Option 3: Manual Deployment on a VPS

1. **Server Setup**
   ```bash
   # Update system packages
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and required system dependencies
   sudo apt install -y python3-pip python3-venv nginx
   
   # Install and configure PostgreSQL
   sudo apt install -y postgresql postgresql-contrib
   sudo -u postgres createuser --superuser $USER
   createdb stockinsightai
   ```

2. **Setup Application**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/StockInsightAI.git
   cd StockInsightAI
   
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set environment variables
   export DATABASE_URL=postgresql://localhost/stockinsightai
   export SECRET_KEY=your-secret-key-here
   ```

3. **Configure Nginx**
   ```bash
   sudo nano /etc/nginx/sites-available/stockinsightai
   ```
   
   Add the following configuration:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   ```
   
   Enable the site and restart Nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/stockinsightai /etc/nginx/sites-enabled
   sudo nginx -t
   sudo systemctl restart nginx
   ```

4. **Run the Application**
   ```bash
   # Install Gunicorn
   pip install gunicorn
   
   # Run with Gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
   ```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
DATABASE_URL=postgresql://username:password@localhost/stockinsightai
SECRET_KEY=your-secret-key-here
DEBUG=False
```

## Post-Deployment

1. **Set up SSL/TLS** (Recommended)
   - Use Let's Encrypt with Certbot for free SSL certificates
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

2. **Set up a process manager** (For production)
   ```bash
   # Install PM2
   npm install -g pm2
   
   # Start the application with PM2
   pm2 start "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000" --name stockinsightai
   
   # Save PM2 process list
   pm2 save
   
   # Set up PM2 to start on boot
   pm2 startup
   ```

## Monitoring

1. **Access Logs**
   ```bash
   # Application logs
   pm2 logs stockinsightai
   
   # Nginx access logs
   tail -f /var/log/nginx/access.log
   
   # Nginx error logs
   tail -f /var/log/nginx/error.log
   ```

2. **Performance Monitoring**
   ```bash
   # Install htop for process monitoring
   sudo apt install htop
   htop
   ```

## Maintenance

- **Backup Database**
  ```bash
  pg_dump stockinsightai > backup_$(date +%Y%m%d).sql
  ```

- **Update Application**
  ```bash
  git pull origin main
  pip install -r requirements.txt
  pm2 restart stockinsightai
  ```

## Troubleshooting

1. **Port already in use**
   ```bash
   sudo lsof -i :8000
   kill -9 <PID>
   ```

2. **Database connection issues**
   - Verify database is running: `sudo systemctl status postgresql`
   - Check connection string in environment variables

3. **Check service status**
   ```bash
   pm2 status
   sudo systemctl status nginx
   sudo systemctl status postgresql
   ```
