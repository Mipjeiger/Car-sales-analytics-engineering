# Car Sales Intelligence — Frontend

React 18 + TypeScript + Vite console for the FastAPI / MLflow / Airflow stack.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

The app expects the API at `http://localhost:8000` (`VITE_API_BASE_URL` in `.env`).

Demo logins:

| Role    | Email                 | Password |
|---------|-----------------------|----------|
| Admin   | admin@carsales.io     | admin    |
| Analyst | analyst@carsales.io   | analyst  |
| Viewer  | viewer@carsales.io    | viewer   |

`/models` is Admin-only.

## API mapping

| UI | Backend |
|----|---------|
| Analytics predict | `POST /predict/` |
| Model list | `GET /predict/models` |
| Chat | `POST /chat/`, `POST /chat/reset`, `GET /chat/intents` |
| Visual search | `POST /search/similar`, `GET /search/brands`, `GET /search/stats` |
| KPIs | `GET /business-metrics`, `GET /metrics`, `GET /health` |
| Damage | `POST /damage/detect` (falls back locally if the route is missing) |

Vite keeps `index.html` at the project root (required by Vite). Static assets live in `public/`.

## Docker

```bash
docker build -t car-sales-frontend .
docker run -p 3000:80 car-sales-frontend
```

From `development/docker-compose.yml`, the `frontend` service builds this folder and publishes port 3000.
