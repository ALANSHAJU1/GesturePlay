
import asyncio
import json
import os
from http import HTTPStatus

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

clients = set()

ALLOWED_COMMANDS = {
    "SEEK_FORWARD",
    "SEEK_BACKWARD",
    "TOGGLE_PLAYBACK",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "TOGGLE_FULLSCREEN",
    "TOGGLE_MUTE",
    "TOGGLE_GESTURE_CONTROL",
}


async def process_request(connection, request):
    """Respond to HTTP health checks while allowing WebSocket upgrades."""
    if request.path == "/health":
        body = b"GesturePlay relay is running"
        return Response(
            HTTPStatus.OK,
            "OK",
            Headers([
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ]),
            body,
        )

    return None


async def handler(websocket):
    clients.add(websocket)
    print(f"Client connected. Total: {len(clients)}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                print("Invalid JSON received")
                continue

            if not isinstance(data, dict):
                continue

            command = data.get("command")

            if (
                data.get("type") != "gesture"
                or command not in ALLOWED_COMMANDS
            ):
                print("Ignored invalid command:", data)
                continue

            print("Gesture received:", command)

            targets = [
                client for client in clients
                if client is not websocket
            ]

            if not targets:
                print("No other connected client to receive gesture.")
                continue

            results = await asyncio.gather(
                *(client.send(json.dumps(data)) for client in targets),
                return_exceptions=True,
            )

            sent = 0
            for client, result in zip(targets, results):
                if isinstance(result, Exception):
                    print("Send failed:", repr(result))
                    clients.discard(client)
                else:
                    sent += 1

            print(f"Forwarded '{command}' to {sent} client(s).")

    except websockets.exceptions.ConnectionClosed:
        print("Client connection closed.")

    except Exception as error:
        print("Handler error:", repr(error))

    finally:
        clients.discard(websocket)
        print(f"Client disconnected. Total: {len(clients)}")


async def main():
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8765"))

    async with websockets.serve(
        handler,
        host,
        port,
        process_request=process_request,
        ping_interval=20,
        ping_timeout=10,
    ):
        print(f"GesturePlay relay listening on {host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
