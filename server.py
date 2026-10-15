
import asyncio
import json
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


async def handler(websocket):
    clients.add(websocket)
    print(f"Client connected. Total: {len(clients)}")

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
                print("No other client connected.")
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

            print(f"Forwarded to {sent} client(s).")

    except websockets.exceptions.ConnectionClosed as error:
        print("Connection closed:", error)

    except Exception as error:
        print("Handler error:", repr(error))

    finally:
        clients.discard(websocket)
        print(f"Client disconnected. Total: {len(clients)}")


async def main():
    async with websockets.serve(
        handler, "127.0.0.1", 8765
    ):
        print("GesturePlay server running on port 8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
