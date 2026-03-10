# Lead Ranking Agent

A Python-based agent that listens for new leads in Supabase, ranks them using scoring rules, and updates the CRM via API.

## Features

- Real-time lead processing via polling
- Dynamic scoring based on Supabase rules
- Integration with NestJS CRM API
- Docker-ready for easy deployment

## Local Setup

1. Create a `.env` file with your environment variables:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_service_key
   SUPABASE_ANON_KEY=your_anon_key
   CRM_API_BASE=your_crm_api_base
   POLL_INTERVAL_SECONDS=5
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the agent:
   ```bash
   python lead_ranking_agent.py
   ```

## Deployment

### On Render (Recommended)

1. Push this code to a GitHub repository.

2. In Render, create a new "Background Worker" service.

3. Connect your GitHub repo and select the branch.

4. Set the build command (if using Docker):
   - Build Command: (leave empty or `docker build -t lead-agent .`)
   - Docker Command: `docker run lead-agent`

5. Set environment variables in Render's dashboard:
   - SUPABASE_URL
   - SUPABASE_SERVICE_KEY
   - SUPABASE_ANON_KEY
   - CRM_API_BASE
   - POLL_INTERVAL_SECONDS

6. Deploy.

### On AWS

1. Use Elastic Beanstalk with Docker platform.

2. Push code to GitHub.

3. In EB, create a new application and environment.

4. Upload the source code or connect to GitHub.

5. Set environment variables in EB configuration.

6. Deploy.

### Using Docker Locally

```bash
docker build -t lead-ranking-agent .
docker run --env-file .env lead-ranking-agent
```

## Environment Variables

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_SERVICE_KEY`: Service role key for Supabase
- `SUPABASE_ANON_KEY`: Anon key for Realtime
- `CRM_API_BASE`: Base URL for CRM API
- `POLL_INTERVAL_SECONDS`: Polling interval (default: 5)

## Security

- Never commit `.env` files to version control.
- Use service keys only in secure environments.
- The `.env` file is ignored by `.gitignore`.
