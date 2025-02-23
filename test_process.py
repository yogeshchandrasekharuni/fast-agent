import anyio


async def main():
    # Choose command based on platform
    # command = "dir" if sys.platform == "win32" else "sdfsdf  dsfsd"

    # print(f"Running command: {command}")
    process = await anyio.open_process(["badcommand", "run", "main.py"])
    print(f"Return code: {process.returncode}")
    print(process.pid)
    print(process.returncode)
    await process.wait()


# print(f"Output:\n{process.stdout.decode()}")


if __name__ == "__main__":
    anyio.run(main)
