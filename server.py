import asyncio
import json
import http.server
import socketserver
import threading
import websockets

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


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files silently without flooding the terminal with HTTP GET logs."""
    def log_message(self, format, *args):
        pass


def start_http_server(host="127.0.0.1", port=8080):
    """Run a local HTTP server in a daemon thread so the browser can load YouTube without Error 153."""
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer((host, port), QuietHTTPHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        print(f"GesturePlay Web Server running at: http://{host}:{port}/index.html")
        return httpd
    except Exception as error:
        print(f"HTTP server note (port {port}): {error}")
        return None


async def handler(websocket):
    clients.add(websocket)
    print(f"Client connected. Total connected: {len(clients)}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print("Invalid JSON received")
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
                print("No active browser window connected to receive gesture.")
                continue

            results = await asyncio.gather(
                *[
                    client.send(json.dumps(data))
                    for client in targets
                ],
                return_exceptions=True
            )

            for client, result in zip(targets, results):
                if isinstance(result, Exception):
                    print("Send failed:", repr(result))
                    clients.discard(client)

            sent = sum(
                not isinstance(result, Exception)
                for result in results
            )

            print(f"Forwarded '{command}' to {sent} client(s).")

    except websockets.exceptions.ConnectionClosed as error:
        print("Connection closed:", error)

    except Exception as error:
        print("Handler error:", repr(error))

    finally:
        clients.discard(websocket)
        print(f"Client disconnected. Total connected: {len(clients)}")


async def main():
    start_http_server("127.0.0.1", 8080)

    async with websockets.serve(
        handler,
        "127.0.0.1",
        8765,
        ping_interval=20,   # Send a keepalive ping every 20s
        ping_timeout=10,    # Drop client if no pong within 10s
    ):
        print("GesturePlay WebSocket Server running on ws://127.0.0.1:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
