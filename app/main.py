from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="MVP Hello World")


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
    <!doctype html>
    <html lang="ru">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Hello World</title>
        <style>
          :root {
            color-scheme: light dark;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f6f7f9;
            color: #1f2933;
          }
          body {
            min-height: 100vh;
            margin: 0;
            display: grid;
            place-items: center;
          }
          main {
            width: min(92vw, 560px);
            padding: 40px;
            border: 1px solid #d9dee7;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 12px 30px rgba(31, 41, 51, 0.08);
          }
          h1 {
            margin: 0 0 12px;
            font-size: clamp(2rem, 6vw, 4rem);
            line-height: 1;
          }
          p {
            margin: 0;
            font-size: 1.125rem;
            color: #52616f;
          }
          @media (prefers-color-scheme: dark) {
            :root {
              background: #111827;
              color: #f9fafb;
            }
            main {
              background: #1f2937;
              border-color: #374151;
              box-shadow: none;
            }
            p {
              color: #d1d5db;
            }
          }
        </style>
      </head>
      <body>
        <main>
          <h1>Hello World</h1>
          <p>FastAPI приложение запущено внутри Docker Compose.</p>
        </main>
      </body>
    </html>
    """


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
