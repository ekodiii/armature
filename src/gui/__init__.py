"""armature-gui — lightweight read-only graph viewer."""

import webbrowser


def main():
    import uvicorn
    from gui.server import app

    print("Armature GUI → http://localhost:7421")
    webbrowser.open("http://localhost:7421")
    uvicorn.run(app, host="127.0.0.1", port=7421, log_level="warning")


if __name__ == "__main__":
    main()
