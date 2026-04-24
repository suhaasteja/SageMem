"""SageMem demo server — FastAPI backend with SSE event stream.

Run with:
    uv run python demo/server/main.py

Then open: http://localhost:8000
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from demo.server.scenario import run_scenario

app = FastAPI(title="SageMem Demo")

UI_DIR = Path(__file__).parent.parent / "ui"


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the demo UI."""
    html_path = UI_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/events")
async def events():
    """SSE stream — runs the demo scenario and emits events."""

    async def generate():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(event_type: str, data: dict):
            payload = json.dumps({"type": event_type, **data})
            await queue.put(f"data: {payload}\n\n")

        async def run():
            try:
                await run_scenario(emit)
            except Exception as e:
                await queue.put(f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n")
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(run())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        await task

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo.server.main:app", host="0.0.0.0", port=8000, reload=False)
